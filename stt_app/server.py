"""EG2 speech in the browser: live Japanese STT with the audio encoder on the Neural Engine (REPORT.md 12・13章), and
TTS through EG2's audio tokens (14章).

uv run --group coreml --group app python -m stt_app.server [--port 8443] [--reader-device cpu] [--no-tts]

Serves https on 0.0.0.0, with a self-signed certificate made on first start: browsers only give the microphone to
https pages (or localhost), and this is meant to be opened from another Mac over Tailscale.
"""

import argparse
import asyncio
import base64
import ctypes
import io
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

# Same cap as the other scripts (REPORT.md 7章).
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

ROOT = Path(__file__).resolve().parent.parent
RATE = 16000
WINDOW = 30 * RATE           # the converted encoder takes exactly 30 s (p5_coreml.py)
SAMPLES_PER_TOKEN = 640      # 25 audio tokens a second
CUT_AFTER = 24 * RATE        # a segment this long is closed at its quietest moment, before it outgrows the window
CUT_FROM = 12 * RATE         # ...looked for no earlier than this


class Transcriber:
    """Waveform -> text: EG2's feature extractor (CPU), the Core ML audio encoder (Neural Engine), the reader."""

    def __init__(self, reader_device):
        import coremltools as ct
        from transformers import AutoProcessor

        from p6_stt import EG2, OUT, Reader, char_table

        self.fe = AutoProcessor.from_pretrained(EG2).feature_extractor
        t = time.perf_counter()
        self.encoder = ct.models.MLModel(str(ROOT / "build/eg2_audio_ane.mlpackage"),
                                         compute_units=ct.ComputeUnit.CPU_AND_NE)
        self.encoder_load = time.perf_counter() - t
        ckpt = torch.load(ROOT / OUT / "reader_eg2.pt", map_location="cpu")
        all_chars, all_emb = char_table()
        self.chars = ckpt["seen"]
        self.reader = Reader("eg2", len(self.chars), all_emb[[all_chars.index(c) for c in self.chars]])
        self.reader.load_state_dict(ckpt["state"])
        self.device = reader_device
        self.reader = self.reader.to(reader_device).eval()
        self.lock = threading.Lock()   # one Neural Engine model, shared by every connection
        for _ in range(3):
            self(np.zeros(RATE, np.float32))

    def __call__(self, audio):
        from p6_stt import greedy

        audio = audio[:WINDOW]
        tokens = max(1, -(-len(audio) // SAMPLES_PER_TOKEN))
        with self.lock:
            t0 = time.perf_counter()
            # Padded with silence to the encoder's 30 s. Its attention never looks ahead, so the padding doesn't
            # change the tokens of the speech before it; only those are read.
            f = self.fe([np.pad(audio, (0, WINDOW - len(audio)))], sampling_rate=RATE, return_tensors="np")
            t1 = time.perf_counter()
            emb = self.encoder.predict({"features": f["input_features"].astype(np.float32)})["audio_embeds"][0][:tokens]
            t2 = time.perf_counter()
            with torch.no_grad():
                logits = self.reader(torch.from_numpy(emb.astype(np.float32))[None].to(self.device))
                text = greedy(logits, torch.tensor([tokens]), self.chars)[0]
            t3 = time.perf_counter()
        return text, {"features": t1 - t0, "encoder": t2 - t1, "reader": t3 - t2, "total": t3 - t0}


class Speaker:
    """Text -> waveform (24 kHz): characters' EG2 text embeddings -> audio tokens -> mel -> Vocos (p7_tts.py)."""

    MAX_CHARS = 400

    def __init__(self, device):
        import p7_tts as t

        self.t, self.device = t, device
        self.index, emb = t.text_table()
        self.tts = t.TextToTokens(emb.float()).to(device).eval()
        self.tts.load_state_dict(torch.load(ROOT / t.OUT / "tts.pt", map_location=device))
        self.decoder = t.Decoder().to(device).eval()
        self.decoder.load_state_dict(torch.load(ROOT / t.OUT / "decoder.pt", map_location=device))
        self.vocos = t.Vocos.pretrained().to(device)
        self.lock = threading.Lock()
        self("テスト")

    def sentences(self, text):
        """Trained on single sentences without punctuation: split at 。！？ and line breaks, long ones at 、 too."""
        from p6_stt import normalize

        out = []
        for part in re.split(r"[。．！？!?\n]+", text):
            pieces = re.split(r"[、，,]", part) if len(normalize(part)) > 60 else [part]
            out += [normalize(p) for p in pieces if normalize(p)]
        return out

    def __call__(self, text):
        t = self.t
        gap = np.zeros(int(0.25 * t.SR), np.float32)
        parts = self.sentences(text)
        if not parts or sum(map(len, parts)) > self.MAX_CHARS:
            raise ValueError(f"1〜{self.MAX_CHARS}文字で入力してください")
        timing = {"tokens": 0.0, "mel": 0.0, "vocos": 0.0}
        waves, unknown = [], sorted({c for p in parts for c in p if c not in self.index})
        sync = torch.mps.synchronize if self.device == "mps" else (lambda: None)
        with self.lock, torch.no_grad():
            for part in parts:
                t0 = time.perf_counter()
                tok = self.tts.generate(torch.tensor([t.encode_text(part, self.index)], device=self.device))[0]
                sync()
                t1 = time.perf_counter()
                mel = self.decoder(tok[None], t.mel_frames(len(tok))).transpose(1, 2)
                sync()
                t2 = time.perf_counter()
                waves += [self.vocos(mel)[0].numpy(), gap]
                t3 = time.perf_counter()
                timing["tokens"] += t1 - t0
                timing["mel"] += t2 - t1
                timing["vocos"] += t3 - t2
        timing["total"] = sum(timing.values())
        return np.concatenate(waves[:-1]), parts, unknown, timing


# ---- memory, as Activity Monitor counts it ---------------------------------------------------------------------

class RusageInfoV2(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in (
        "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size", "resident_size",
        "phys_footprint", "proc_start_abstime", "proc_exit_abstime", "child_user_time", "child_system_time",
        "child_pkg_idle_wkups", "child_interrupt_wkups", "child_pageins", "child_elapsed_abstime",
        "diskio_bytesread", "diskio_byteswritten")]


_libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
_peak = 0


def memory():
    """This process' physical footprint (Activity Monitor's "Memory"), its peak, and the whole Mac's use.
    Footprint covers what Core ML and Metal map into the process; what the Neural Engine driver holds is not in it."""
    global _peak
    info = RusageInfoV2()
    _libproc.proc_pid_rusage(os.getpid(), 2, ctypes.byref(info))   # RUSAGE_INFO_V2
    _peak = max(_peak, info.phys_footprint)
    page = os.sysconf("SC_PAGE_SIZE")
    vm = {}
    for line in subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout.splitlines()[1:]:
        key, _, value = line.partition(":")
        vm[key.strip()] = int(value.strip().rstrip("."))
    # Activity Monitor's "Memory Used": app memory + wired + compressed.
    used = (vm["Anonymous pages"] - vm["Pages purgeable"] + vm["Pages wired down"]
            + vm["Pages occupied by compressor"]) * page
    out = {"footprint": info.phys_footprint, "peak": _peak, "resident": info.resident_size,
           "system_used": used, "system_total": os.sysconf("SC_PHYS_PAGES") * page}
    if torch.backends.mps.is_available():
        out["mps"] = torch.mps.driver_allocated_memory()
    return out


# ---- the web app ------------------------------------------------------------------------------------------------

app = FastAPI()
model: Transcriber
speaker: Speaker | None = None
SAMPLES = ROOT / "build/tts/samples"
RECORDINGS = ROOT / "build/stt_app/recordings"


@app.get("/")
def index():
    return HTMLResponse((Path(__file__).parent / "index.html").read_text(encoding="utf-8"))


@app.get("/info")
def info():
    return {"reader_device": model.device, "encoder_load": model.encoder_load, "tts": speaker is not None,
            "tts_device": speaker.device if speaker else None, "memory": memory()}


@app.post("/tts")
async def speak(text: str = Body(..., embed=True)):
    if speaker is None:
        raise HTTPException(404, "TTS is off (--no-tts)")
    try:
        wave, parts, unknown, timing = await asyncio.to_thread(speaker, text)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    buf = io.BytesIO()
    sf.write(buf, wave, speaker.t.SR, format="WAV", subtype="PCM_16")
    return {"audio": base64.b64encode(buf.getvalue()).decode(), "seconds": len(wave) / speaker.t.SR,
            "sentences": parts, "unknown": unknown, "timing": timing, "memory": memory()}


@app.get("/samples")
def samples():
    """The FLEURS test clips p7_tts.py listen saved, each as the original and the speech made at every step."""
    import zipfile

    from p6_stt import OUT as STT
    from p6_stt import normalize

    keys = sorted({p.stem.removeprefix("orig-") for p in SAMPLES.glob("orig-*.wav")})
    with zipfile.ZipFile(ROOT / STT / "fleurs-test.npz") as z:   # just the ids and texts, not the tokens
        d = np.load(z.open("ids.npy")), np.load(z.open("texts.npy"))
    text = {Path(k).stem: normalize(t) for k, t in zip(*d)}
    kinds = ["orig", "copy", "decoder", "tts-oracle", "tts"]
    return [{"key": k, "text": text.get(k, ""), "files": {kind: f"{kind}-{k}.wav" for kind in kinds
                                                          if (SAMPLES / f"{kind}-{k}.wav").exists()}} for k in keys]


@app.get("/samples/{name}")
def sample(name: str):
    if not re.fullmatch(r"[a-z-]+-\d+\.wav", name) or not (SAMPLES / name).exists():
        raise HTTPException(404)
    return FileResponse(SAMPLES / name, media_type="audio/wav")


def quietest(audio):
    """Where to close a long segment: the middle of its quietest 100 ms after CUT_FROM."""
    frames = audio[CUT_FROM:len(audio) - len(audio) % 1600].reshape(-1, 1600)
    return CUT_FROM + int(np.argmin((frames ** 2).mean(1))) * 1600 + 800


@app.websocket("/ws")
async def transcribe(ws: WebSocket):
    """Takes 16 kHz mono int16 PCM as binary messages ("reset" as text). After new audio arrives, re-reads the open
    segment (up to 30 s) and sends back the closed text, the open segment's text, timings and memory."""
    await ws.accept()
    closed, segment, received, recorded = [], np.zeros(0, np.float32), 0, []
    fresh = asyncio.Event()

    async def receive():
        nonlocal closed, segment, received, recorded
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                raise WebSocketDisconnect
            if msg.get("bytes"):
                pcm = np.frombuffer(msg["bytes"], np.int16).astype(np.float32) / 32768
                segment = np.concatenate([segment, pcm])
                recorded.append(pcm)
                received += len(pcm)
                fresh.set()
            elif msg.get("text") == "reset":
                closed, segment, received, recorded = [], np.zeros(0, np.float32), 0, []
            elif msg.get("text") == "save" and recorded:
                # Only when asked (the page's button): what the server heard, to look into a transcript offline.
                RECORDINGS.mkdir(parents=True, exist_ok=True)
                path = RECORDINGS / time.strftime("%Y%m%d-%H%M%S.wav")
                sf.write(path, np.concatenate(recorded), RATE)
                await ws.send_text(json.dumps({"saved": str(path.relative_to(ROOT))}))

    async def work():
        nonlocal segment
        while True:
            await fresh.wait()
            fresh.clear()
            upto, cut = received, None
            if len(segment) >= CUT_AFTER:
                cut = quietest(segment[:WINDOW])
                text, _ = await asyncio.to_thread(model, segment[:cut])
                closed.append(text)
                segment = segment[cut:]
            audio = segment
            text, timing = await asyncio.to_thread(model, audio)
            await ws.send_text(json.dumps({
                "closed": "".join(closed), "open": text, "segment_seconds": len(audio) / RATE,
                "received": upto, "cut": cut is not None, "timing": timing, "memory": memory()}))

    tasks = [asyncio.create_task(receive()), asyncio.create_task(work())]
    try:
        await asyncio.gather(*tasks)
    except WebSocketDisconnect:
        pass
    finally:
        for t in tasks:
            t.cancel()


def certificate():
    """A self-signed certificate for localhost and this Mac's Tailscale address, made once into build/stt_app/."""
    out = ROOT / "build/stt_app"
    cert, key = out / "cert.pem", out / "key.pem"
    if not cert.exists():
        out.mkdir(parents=True, exist_ok=True)
        names = ["DNS:localhost", "IP:127.0.0.1"]
        try:
            ip = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, check=True).stdout.split()
            status = json.loads(subprocess.run(["tailscale", "status", "--json"], capture_output=True, text=True,
                                               check=True).stdout)
            names += [f"IP:{a}" for a in ip] + [f"DNS:{status['Self']['DNSName'].rstrip('.')}"]
        except (OSError, subprocess.CalledProcessError, KeyError, ValueError):
            pass
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "365",
                        "-subj", "/CN=eg2-stt", "-addext", "subjectAltName=" + ",".join(names),
                        "-keyout", str(key), "-out", str(cert)], check=True, capture_output=True)
    return str(cert), str(key)


def main():
    global model, speaker
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8443)
    p.add_argument("--reader-device", default="cpu", choices=["cpu", "mps"])
    p.add_argument("--tts-device", default="mps", choices=["cpu", "mps"])
    p.add_argument("--no-tts", action="store_true")
    args = p.parse_args()
    t = time.perf_counter()
    model = Transcriber(args.reader_device)
    if not args.no_tts:
        speaker = Speaker(args.tts_device)
    print(f"ready in {time.perf_counter() - t:.1f}s (encoder load {model.encoder_load:.1f}s); "
          f"memory {memory()['footprint'] / 2**20:.0f} MiB")
    cert, key = certificate()
    uvicorn.run(app, host=args.host, port=args.port, ssl_certfile=cert, ssl_keyfile=key)


if __name__ == "__main__":
    main()
