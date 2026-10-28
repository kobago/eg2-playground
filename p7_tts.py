"""Japanese text-to-speech through EG2's audio tokens (REPORT.md 14章).

Step 1: can EG2's audio tokens be turned back into speech? A small decoder maps them to a mel spectrogram and a
pretrained Vocos turns that into a waveform.

Step 2: text to EG2's audio tokens. Each character's frozen EG2 text embedding goes in; a small model predicts how
many audio tokens it lasts and the tokens themselves, which step 1 turns into speech.

uv run python p7_tts.py mels {fleurs-train,fleurs-dev,fleurs-test}
uv run python p7_tts.py train-decoder
uv run python p7_tts.py align {fleurs-train,fleurs-dev,fleurs-test}
uv run python p7_tts.py train-tts
uv run --group whisper python p7_tts.py listen {copy,decoder,tts-oracle,tts} fleurs-test
"""

import glob
import os
import random
import sys
import time
from pathlib import Path

# Same cap as the other scripts (REPORT.md 7章).
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch import nn

from p6_stt import (
    DEVICE,
    HUB,
    SOURCES,
    WHISPER,
    Reader,
    char_table,
    edit_distance,
    load_split,
    normalize,
    sinusoid,
)
from p6_stt import OUT as STT

OUT = Path("build/tts")
SR = 24000
HOP = 256
MEL_PER_TOKEN = SR / HOP / 25   # 3.75: Vocos' 93.75 mel frames a second over EG2's 25 tokens


# ---- Vocos (gemelo-ai/vocos, MIT): the inference half, to load charactr/vocos-mel-24khz without the package --------

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, intermediate_dim, layer_scale):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim))

    def forward(self, x):   # [B, C, T]
        h = self.dwconv(x).transpose(1, 2)
        h = self.gamma * self.pwconv2(F.gelu(self.pwconv1(self.norm(h))))
        return x + h.transpose(1, 2)


class VocosBackbone(nn.Module):
    def __init__(self, input_channels=100, dim=512, intermediate_dim=1536, num_layers=8):
        super().__init__()
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.convnext = nn.ModuleList(ConvNeXtBlock(dim, intermediate_dim, 1 / num_layers) for _ in range(num_layers))
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):   # [B, mels, T] -> [B, T, dim]
        x = self.norm(self.embed(x).transpose(1, 2)).transpose(1, 2)
        for block in self.convnext:
            x = block(x)
        return self.final_layer_norm(x.transpose(1, 2))


class ISTFTHead(nn.Module):
    def __init__(self, dim=512, n_fft=1024):
        super().__init__()
        self.n_fft = n_fft
        self.out = nn.Linear(dim, n_fft + 2)
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, x):
        mag, phase = self.out(x).transpose(1, 2).chunk(2, dim=1)
        spec = torch.clip(torch.exp(mag), max=1e2) * torch.exp(1j * phase)
        # MPS has no complex istft.
        return torch.istft(spec.cpu(), self.n_fft, HOP, self.n_fft, self.window.cpu(), center=True)


class Vocos(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone, self.head = VocosBackbone(), ISTFTHead()
        self.mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=1024, hop_length=HOP, n_mels=100, center=True,
                                                        power=1)

    def features(self, audio):   # 24 kHz waveform -> log mel, as Vocos was trained on
        return torch.log(torch.clip(self.mel(audio), min=1e-7))

    def forward(self, mel):
        return self.head(self.backbone(mel))

    @classmethod
    def pretrained(cls):
        path = glob.glob(str(HUB / "models--charactr--vocos-mel-24khz/snapshots/*/pytorch_model.bin"))[0]
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = {k: v for k, v in state.items() if not k.startswith(("feature_extractor.", "head.istft."))}
        v = cls()
        missing, unexpected = v.load_state_dict(state, strict=False)
        assert not unexpected and all(k.startswith(("mel.", "head.window")) for k in missing), (missing, unexpected)
        return v.eval()


# ---- targets: the mel spectrogram of each clip -----------------------------------------------------------------

def to_24k(audio_16k, tokens):
    """The clip padded to its tokens' length (EG2 makes a token per 640 samples, the last one partial), at 24 kHz."""
    audio = np.pad(audio_16k, (0, tokens * 640 - len(audio_16k)))
    return torchaudio.functional.resample(torch.from_numpy(audio), 16000, SR)


def mels(name):
    vocos = Vocos()
    ids, _, tokens = load_split(name)
    length = dict(zip(ids, map(len, tokens)))
    out = {}
    for key, audio, _ in SOURCES[name]():
        out[key] = vocos.features(to_24k(audio, length[key])).T.numpy().astype(np.float16)   # [T, 100]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"mel-{name}.npz", **{k: out[k] for k in ids})
    print(f"mels {name}: {len(out)} clips, {sum(len(m) for m in out.values())} frames")


def load_pairs(name):
    ids, texts, tokens = load_split(name)
    m = np.load(OUT / f"mel-{name}.npz")
    return ids, texts, tokens, [m[k] for k in ids]


# ---- the decoder: audio tokens -> mel --------------------------------------------------------------------------

class Decoder(nn.Module):
    """ConvNeXt blocks at the token rate, a linear stretch to the mel rate (x3.75), more blocks, a mel per frame."""

    def __init__(self, d=512, token_layers=4, mel_layers=6):
        super().__init__()
        self.inp = nn.Linear(512, d)
        self.token_blocks = nn.ModuleList(ConvNeXtBlock(d, 3 * d, 0.1) for _ in range(token_layers))
        self.mel_blocks = nn.ModuleList(ConvNeXtBlock(d, 3 * d, 0.1) for _ in range(mel_layers))
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 100)

    def forward(self, x, frames):   # [B, tokens, 512] -> [B, frames, 100]
        h = self.inp(x).transpose(1, 2)
        for block in self.token_blocks:
            h = block(h)
        h = F.interpolate(h, size=frames, mode="linear", align_corners=False)
        for block in self.mel_blocks:
            h = block(h)
        return self.out(self.norm(h.transpose(1, 2)))


def mel_frames(tokens):
    return int(tokens * MEL_PER_TOKEN) + 1   # center=True adds one frame


def pair_batches(tokens, mel, max_tokens=6000, shuffle=True):
    order = sorted(range(len(tokens)), key=lambda i: len(tokens[i]))
    groups, cur = [], []
    for i in order:
        if cur and len(tokens[i]) * (len(cur) + 1) > max_tokens:
            groups.append(cur)
            cur = []
        cur.append(i)
    groups.append(cur)
    if shuffle:
        random.shuffle(groups)
    for g in groups:
        x = nn.utils.rnn.pad_sequence([torch.from_numpy(tokens[i].astype(np.float32)) for i in g], batch_first=True)
        y = nn.utils.rnn.pad_sequence([torch.from_numpy(mel[i].astype(np.float32)) for i in g], batch_first=True)
        n = torch.tensor([len(mel[i]) for i in g])
        yield g, x, y, n


def masked_l1(pred, target, lengths):
    mask = (torch.arange(target.shape[1], device=target.device)[None] < lengths[:, None].to(target.device))[..., None]
    return ((pred - target).abs() * mask).sum() / (mask.sum() * target.shape[-1])


def train_decoder(epochs=60):
    torch.manual_seed(0)
    random.seed(0)
    _, _, train_tok, train_mel = load_pairs("fleurs-train")
    _, _, dev_tok, dev_mel = load_pairs("fleurs-dev")
    assert all(len(m) == mel_frames(len(t)) for t, m in zip(train_tok, train_mel))
    model = Decoder().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    steps = epochs * sum(1 for _ in pair_batches(train_tok, train_mel))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
    print(f"train decoder: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    best, start = 1e9, time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for _, x, y, n in pair_batches(train_tok, train_mel):
            y = y.to(DEVICE)
            loss = masked_l1(model(x.to(DEVICE), y.shape[1]), y, n)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        if epoch % 10 == 9 or epoch == epochs - 1:
            dev = dev_loss(model, dev_tok, dev_mel)
            print(f"  epoch {epoch + 1}: dev L1 {dev:.3f}, {time.perf_counter() - start:.0f}s")
            if dev < best:
                best = dev
                torch.save(model.state_dict(), OUT / "decoder.pt")


def dev_loss(model, tokens, mel):
    model.eval()
    total = count = 0
    with torch.no_grad():
        for g, x, y, n in pair_batches(tokens, mel, shuffle=False):
            y = y.to(DEVICE)
            total += masked_l1(model(x.to(DEVICE), y.shape[1]), y, n).item() * len(g)
            count += len(g)
    return total / count


# ---- step 2: text -> audio tokens -------------------------------------------------------------------------------

def align(name):
    """How many audio tokens each character lasts, from the STT reader (REPORT.md 13章) by CTC forced alignment.
    A character keeps the blanks after it; the blanks before the first one go to a start symbol, so the durations
    of [start, characters..., end] add up to the clip's tokens."""
    ckpt = torch.load(STT / "reader_eg2.pt", map_location="cpu")
    reader = Reader("eg2", len(ckpt["seen"]), torch.zeros(len(ckpt["seen"]), 512)).to(DEVICE).eval()
    reader.load_state_dict(ckpt["state"])
    # Over all of EG2's characters, not just the training ones: the EG2 head scores any of them (13章), so test texts
    # with characters training never had can be aligned too.
    chars, emb = char_table()
    index = {c: i + 1 for i, c in enumerate(chars)}
    emb = F.normalize(emb.float(), dim=-1).to(DEVICE)
    ids, texts, tokens = load_split(name)
    out, skipped = {}, 0
    for key, text, tok in zip(ids, texts, tokens):
        if not text or any(c not in index for c in text):
            skipped += 1
            continue
        with torch.no_grad():
            logp = reader(torch.from_numpy(tok.astype(np.float32))[None].to(DEVICE), emb).log_softmax(-1).cpu()
        try:
            path, scores = torchaudio.functional.forced_align(logp, torch.tensor([[index[c] for c in text]]), blank=0)
        except RuntimeError:   # fewer tokens than the text needs
            skipped += 1
            continue
        starts = [span.start for span in torchaudio.functional.merge_tokens(path[0], scores[0].exp())]
        bounds = [0, *starts, len(tok)]
        durations = np.diff(bounds)
        out[key] = np.concatenate([durations, [0]]).astype(np.int16)   # the end symbol lasts nothing
    np.savez(OUT / f"align-{name}.npz", **out)
    total = np.concatenate(list(out.values()))
    print(f"align {name}: {len(out)} clips, {skipped} skipped, {total[1:-1].mean():.2f} tokens per character "
          f"(start {np.mean([d[0] for d in out.values()]):.1f})")


class TextToTokens(nn.Module):
    """FastSpeech-like: characters (frozen EG2 text embeddings) -> Transformer -> durations; each character's state
    repeated for its duration -> Transformer -> EG2 audio tokens (512-d)."""

    def __init__(self, emb, d=384, layers=4):
        super().__init__()
        self.register_buffer("emb", emb)                 # [characters, 512], EG2's, never trained
        self.special = nn.Parameter(torch.randn(3, 512) * emb.std())   # start, end, unknown character
        self.inp = nn.Sequential(nn.Linear(512, d), nn.LayerNorm(d))
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.1, batch_first=True, norm_first=True), layers,
            enable_nested_tensor=False)
        self.duration = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d), nn.Linear(d, 1))
        self.dec = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.1, batch_first=True, norm_first=True), layers,
            enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 512)

    def encode(self, idx, pad):   # idx: [B, L] (characters, then len(emb) + 0/1/2 for the specials)
        n = len(self.emb)
        x = torch.where((idx < n)[..., None], self.emb[idx.clamp(max=n - 1)], self.special[(idx - n).clamp(min=0)])
        h = self.inp(x)
        h = self.enc(h + sinusoid(h.shape[1], h.shape[2], h.device), src_key_padding_mask=pad)
        return h, self.duration(h).squeeze(-1)   # log(1 + tokens)

    def decode(self, h, durations):
        """durations: [B, L] whole tokens; padding has 0."""
        frames = [torch.repeat_interleave(hb, db, dim=0) for hb, db in zip(h, durations)]
        lengths = torch.tensor([len(f) for f in frames], device=h.device)
        x = nn.utils.rnn.pad_sequence(frames, batch_first=True)
        pad = torch.arange(x.shape[1], device=h.device)[None] >= lengths[:, None]
        y = self.dec(x + sinusoid(x.shape[1], x.shape[2], x.device), src_key_padding_mask=pad)
        return self.out(self.norm(y)), lengths

    def generate(self, idx):
        h, logd = self.encode(idx, torch.zeros_like(idx, dtype=torch.bool))
        durations = torch.round(torch.expm1(logd)).clamp(min=0).long()
        return self.decode(h, durations)[0]


def text_table():
    chars, emb = char_table()
    return {c: i for i, c in enumerate(chars)}, emb


def encode_text(text, index):
    n = len(index)
    return [n] + [index.get(c, n + 2) for c in text] + [n + 1]


def load_tts(name, index):
    ids, texts, tokens = load_split(name)
    a = np.load(OUT / f"align-{name}.npz")
    keep = [i for i, k in enumerate(ids) if k in a.files]
    return ([ids[i] for i in keep], [texts[i] for i in keep], [encode_text(texts[i], index) for i in keep],
            [a[ids[i]] for i in keep], [tokens[i] for i in keep])


def tts_batches(chars, durations, tokens, max_tokens=6000, shuffle=True):
    order = sorted(range(len(tokens)), key=lambda i: len(tokens[i]))
    groups, cur = [], []
    for i in order:
        if cur and len(tokens[i]) * (len(cur) + 1) > max_tokens:
            groups.append(cur)
            cur = []
        cur.append(i)
    groups.append(cur)
    if shuffle:
        random.shuffle(groups)
    for g in groups:
        c = nn.utils.rnn.pad_sequence([torch.tensor(chars[i]) for i in g], batch_first=True, padding_value=-1)
        d = nn.utils.rnn.pad_sequence([torch.from_numpy(durations[i].astype(np.int64)) for i in g], batch_first=True)
        y = nn.utils.rnn.pad_sequence([torch.from_numpy(tokens[i].astype(np.float32)) for i in g], batch_first=True)
        yield g, c, d, y


def tts_loss(model, c, d, y):
    c, d, y = c.to(DEVICE), d.to(DEVICE), y.to(DEVICE)
    pad = c < 0
    h, logd = model.encode(c.clamp(min=0), pad)
    keep = ~pad
    dur_loss = F.mse_loss(logd[keep], torch.log1p(d[keep].float()))
    pred, lengths = model.decode(h, d)
    mask = (torch.arange(y.shape[1], device=DEVICE)[None] < lengths[:, None])[..., None]
    token_loss = ((pred - y).abs() * mask).sum() / (mask.sum() * y.shape[-1])
    return token_loss, dur_loss


def train_tts(epochs=60):
    torch.manual_seed(0)
    random.seed(0)
    index, emb = text_table()
    _, _, train_c, train_d, train_y = load_tts("fleurs-train", index)
    _, _, dev_c, dev_d, dev_y = load_tts("fleurs-dev", index)
    model = TextToTokens(emb.float()).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=0.01)
    steps = epochs * sum(1 for _ in tts_batches(train_c, train_d, train_y))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-4, total_steps=steps, pct_start=0.1)
    print(f"train tts: {sum(p.numel() for p in params) / 1e6:.1f}M params, {len(train_c)} clips")
    best, start = 1e9, time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for _, c, d, y in tts_batches(train_c, train_d, train_y):
            token_loss, dur_loss = tts_loss(model, c, d, y)
            opt.zero_grad()
            (token_loss + dur_loss).backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
        if epoch % 10 == 9 or epoch == epochs - 1:
            model.eval()
            losses = []
            with torch.no_grad():
                for g, c, d, y in tts_batches(dev_c, dev_d, dev_y, shuffle=False):
                    losses.append([v.item() * len(g) for v in tts_loss(model, c, d, y)])
            tok, dur = np.sum(losses, 0) / len(dev_c)
            print(f"  epoch {epoch + 1}: dev token L1 {tok:.3f}, duration MSE {dur:.3f}, {time.perf_counter() - start:.0f}s")
            if tok + dur < best:
                best = tok + dur
                torch.save(model.state_dict(), OUT / "tts.pt")


# ---- listening: Whisper reads the speech back -----------------------------------------------------------------

def to_16k(wave_24k):
    return torchaudio.functional.resample(wave_24k, SR, 16000).numpy().astype(np.float32)


def listen(kind, split):
    """Speech made for each test clip, transcribed by Whisper: the CER says how much of the words survive.
    copy: the clip's own mel through Vocos (Vocos' share of the loss). decoder: the clip's EG2 audio tokens through the
    decoder. tts-oracle: tokens made from the text, with the durations aligned to the real clip. tts: from the text
    alone (predicted durations): text-to-speech."""
    import mlx_whisper

    ids, texts, tokens, mel = load_pairs(split)
    vocos = Vocos.pretrained().to(DEVICE)
    decoder = tts = None
    if kind != "copy":
        decoder = Decoder().to(DEVICE).eval()
        decoder.load_state_dict(torch.load(OUT / "decoder.pt"))
    if kind.startswith("tts"):
        index, emb = text_table()
        tts = TextToTokens(emb.float()).to(DEVICE).eval()
        tts.load_state_dict(torch.load(OUT / "tts.pt"))
        aligned = np.load(OUT / f"align-{split}.npz")
        keep = [i for i, k in enumerate(ids) if k in aligned.files]
        ids, texts, tokens, mel = ([v[i] for i in keep] for v in (ids, texts, tokens, mel))
    samples = OUT / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    opts = {"path_or_hf_repo": WHISPER, "language": "ja", "temperature": 0.0, "condition_on_previous_text": False,
            "verbose": None}
    errors, seconds, compute = 0, 0.0, 0.0
    for i, (key, text, tok, m) in enumerate(zip(ids, texts, tokens, mel)):
        t = time.perf_counter()
        with torch.no_grad():
            if kind == "copy":
                spec = torch.from_numpy(m.astype(np.float32)).T[None]
            else:
                if kind == "tts":
                    tok = tts.generate(torch.tensor([encode_text(text, index)], device=DEVICE))[0]
                elif kind == "tts-oracle":
                    d = torch.from_numpy(aligned[key].astype(np.int64))[None].to(DEVICE)
                    c = torch.tensor([encode_text(text, index)], device=DEVICE)
                    h, _ = tts.encode(c, torch.zeros_like(c, dtype=torch.bool))
                    tok = tts.decode(h, d)[0][0]
                else:
                    tok = torch.from_numpy(tok.astype(np.float32)).to(DEVICE)
                spec = decoder(tok[None], mel_frames(len(tok))).transpose(1, 2)
            wave = vocos(spec.to(DEVICE))[0]
        compute += time.perf_counter() - t
        seconds += len(wave) / SR
        if i < 3:
            sf.write(samples / f"{kind}-{Path(key).stem}.wav", wave.numpy(), SR)
        hyp = normalize(mlx_whisper.transcribe(to_16k(wave), **opts)["text"])
        errors += edit_distance(hyp, text)
        if i < 3:
            print(f"  ref: {text}\n  hyp: {hyp}")
    print(f"listen {kind} {split}: Whisper CER {errors / sum(len(t) for t in texts):.3f} over {len(ids)} clips; "
          f"{seconds / 3600:.2f}h of speech made in {compute:.0f}s ({seconds / compute:.0f}x realtime, clip by clip)")
    if not kind.startswith("tts"):
        speed(vocos, decoder)


def speed(vocos, decoder):
    """30 s of audio tokens (750) to a waveform."""
    x = torch.randn(1, 750, 512, device=DEVICE)
    frames = mel_frames(750)

    def run():
        with torch.no_grad():
            spec = decoder(x, frames).transpose(1, 2) if decoder else torch.randn(1, 100, frames, device=DEVICE)
            return vocos(spec)

    for _ in range(3):
        run()
    n, t = 0, time.perf_counter()
    while time.perf_counter() - t < 3:
        run()
        n += 1
    sec = (time.perf_counter() - t) / n
    print(f"  {'decoder + ' if decoder else ''}vocos: {sec * 1000:.0f}ms per 30s ({30 / sec:.0f}x realtime)")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    mode, *args = sys.argv[1:]
    {"mels": mels, "train-decoder": train_decoder, "align": align, "train-tts": train_tts, "listen": listen}[mode](*args)
