import os
import resource
import sys
import time

# Uncapped, MPS grows past the M1's 16GB into swap and freezes the desktop instead of raising OOM.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

import numpy as np
import soundfile as sf
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

EG2 = "google/embeddinggemma-2"
TOWERS = {
    "text": {"vision_config": None, "audio_config": None},
    "text+image": {"audio_config": None},
    "text+audio": {"vision_config": None},
    "full": {},
}
JA = "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。" * 2000
AUDIO = sf.read("data/say/ja_long.wav")[0].astype(np.float32)


def sync(device):
    if device == "mps":
        torch.mps.synchronize()


def bench(fn, device, budget_s=8.0):
    fn()
    sync(device)
    n, t = 0, time.perf_counter()
    while time.perf_counter() - t < budget_s or n < 2:
        fn()
        n += 1
    sync(device)
    return (time.perf_counter() - t) / n


def mem(device):
    gb = 2**30
    mps = f"mps_alloc={torch.mps.current_allocated_memory() / gb:.2f}GB mps_driver={torch.mps.driver_allocated_memory() / gb:.2f}GB" if device == "mps" else ""
    return f"maxrss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / gb:.2f}GB {mps}"


def load(name, device, dtype, config_kwargs=None):
    return SentenceTransformer(name, device=device, config_kwargs=config_kwargs, model_kwargs={"dtype": dtype})


def run_mem(tower):
    m = load(EG2, "mps", torch.bfloat16, TOWERS[tower])
    n = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"mem {tower}: params={n:.0f}M after_load {mem('mps')}")
    m.max_seq_length = 512
    m.encode([JA] * 16, batch_size=16)
    print(f"mem {tower}: after_encode(16x512tok) {mem('mps')}")


def run_text(name, device, dtype, lengths):
    m = load(name, device, dtype, TOWERS["text"] if name == EG2 else None)
    for length in lengths:
        m.max_seq_length = length
        for bs in [1, 16] if length <= 512 else [1]:
            sec = bench(lambda: m.encode([JA] * bs, batch_size=bs), device)
            print(f"text {name} {device} {dtype} len={length} bs={bs}: {sec * 1000:.0f}ms/batch {bs / sec:.1f}docs/s {bs * length / sec:.0f}tok/s {mem(device)}")


def run_audio(chunk_s, bs):
    m = load(EG2, "mps", torch.bfloat16, TOWERS["text+audio"])
    am, fe = m[0].auto_model, m[0].processor.feature_extractor
    # The feature extractor silently cuts input to 30s unless told otherwise.
    m[0].processing_kwargs["audio"] = {"truncation": False}
    # ja_long.wav is ~65s, so longer chunks repeat it; only the length matters for speed.
    chunk = np.resize(AUDIO, 16000 * chunk_s)
    clips = [{"array": chunk, "sampling_rate": 16000}] * bs

    def tower_only():
        f = fe([chunk] * bs, sampling_rate=16000, return_tensors="pt", truncation=False)
        with torch.no_grad():
            am.get_audio_features(f["input_features"].to("mps", torch.bfloat16), f["input_features_mask"].to("mps"))

    for label, fn in [("tower_only", tower_only), ("full_encode", lambda: m.encode(clips, batch_size=bs))]:
        sec = bench(fn, "mps")
        rt = chunk_s * bs / sec
        print(f"audio {label} chunk={chunk_s}s bs={bs}: {sec * 1000:.0f}ms/batch {rt:.0f}x realtime 1h_audio={3600 / rt:.0f}s {mem('mps')}")


def run_image(budget, bs):
    m = load(EG2, "mps", torch.bfloat16, TOWERS["text+image"])
    m[0].processing_kwargs["image"] = {"max_soft_tokens": budget}
    img = Image.fromarray((np.random.default_rng(0).random((480, 640, 3)) * 255).astype("uint8"))
    sec = bench(lambda: m.encode([img] * bs, batch_size=bs), "mps")
    print(f"image budget={budget} bs={bs}: {sec * 1000:.0f}ms/batch {bs / sec:.1f}img/s {mem('mps')}")


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    if mode == "mem":
        run_mem(args[0])
    elif mode == "text":
        dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args[2]]
        run_text(args[0], args[1], dtype, [int(x) for x in args[3].split(",")])
    elif mode == "audio":
        run_audio(int(args[0]), int(args[1]))
    elif mode == "image":
        run_image(int(args[0]), int(args[1]))
