import glob
import os

import numpy as np
import soundfile as sf
import torch
from sentence_transformers import SentenceTransformer

MODEL = "google/embeddinggemma-2"
TEXT_ONLY = {"vision_config": None, "audio_config": None}
TEXTS = [
    "What causes the northern lights?",
    "オーロラはなぜ発生するのですか？",
    "The northern lights are caused by charged particles from the sun.",
]


def load(device, dtype, config_kwargs=TEXT_ONLY):
    return SentenceTransformer(MODEL, device=device, config_kwargs=config_kwargs, model_kwargs={"dtype": dtype})


def precision_check():
    ref = load("cpu", torch.float32).encode(TEXTS, prompt_name="SearchQuery")
    print("== dtype/device vs cpu-fp32 (cosine, min over texts)")
    for device, dtype in [("mps", torch.float32), ("mps", torch.bfloat16), ("cpu", torch.bfloat16), ("mps", torch.float16)]:
        emb = load(device, dtype).encode(TEXTS, prompt_name="SearchQuery")
        nan = np.isnan(emb).any()
        cos = (emb * ref).sum(1) / (np.linalg.norm(emb, axis=1) * np.linalg.norm(ref, axis=1))
        print(f"{device:4} {str(dtype):15} nan={nan} cos_min={np.nanmin(cos):.5f}")


def audio_check():
    m = load("mps", torch.bfloat16, {"vision_config": None})
    words = sorted({os.path.basename(p)[3:-4] for p in glob.glob("data/say/*.wav")})
    ja = {"apple": "りんご", "banana": "バナナ", "dog": "犬", "cat": "猫", "airplane": "飛行機"}
    for lang in ["en", "ja"]:
        audio = [{"array": sf.read(f"data/say/{lang}_{w}.wav")[0].astype(np.float32), "sampling_rate": 16000} for w in words]
        a = m.encode(audio)
        for label, cands in [("en-text", words), ("ja-text", [ja[w] for w in words])]:
            t = m.encode(cands, prompt_name="Document")
            sim = m.similarity(a, t)
            acc = (sim.argmax(1) == torch.arange(len(words))).float().mean().item()
            print(f"== {lang}-audio -> {label}: top1={acc:.2f}")
            print(np.array2string(sim.numpy(), precision=3))


if __name__ == "__main__":
    precision_check()
    audio_check()
