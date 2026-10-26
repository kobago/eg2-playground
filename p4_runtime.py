import base64
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

# Same cap as the other scripts: only the PyTorch reference here uses MPS, but keep it bounded.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

EG2 = "google/embeddinggemma-2"
# A llama-server running the Q8_0 GGUF + mmproj with --embeddings --pooling mean (command in REPORT.md).
LLAMA = os.environ.get("LLAMA_URL", "http://127.0.0.1:8089")
QUERY = "task: search result | query: 海辺で走る犬の写真"
JA = "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。" * 2000
NOISE = Image.fromarray((np.random.default_rng(0).random((480, 640, 3)) * 255).astype("uint8"))


def doc_image():
    img = Image.new("RGB", (900, 1200), "white")
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", 36)
    for i, line in enumerate(["請求書", "請求番号: INV-2026-0042", "お支払い期限: 2026年11月30日", "合計金額: 128,000円"]):
        d.text((60, 80 + i * 90), line, fill="black", font=font)
    d.ellipse((200, 650, 700, 950), fill="orange")
    return img


def fit(img, budget):
    # transformers' Gemma 4 image processor: fill `budget` soft tokens (3x3 patches of 16px), sides rounded down to
    # 48px. llama.cpp resizes differently (cosine 0.97 on a document page), so images go to it already at this size.
    w, h = img.size
    f = (budget * 9 * 256 / (h * w)) ** 0.5
    return img.resize((int(f * w // 48) * 48, int(f * h // 48) * 48), Image.Resampling.BICUBIC)


def bench(fn, budget_s=8.0):
    fn()
    n, t = 0, time.perf_counter()
    while time.perf_counter() - t < budget_s or n < 2:
        fn()
        n += 1
    return (time.perf_counter() - t) / n


class Mlx:
    def __init__(self):
        import mlx.core as mx
        from mlx_vlm import load

        self.mx = mx
        self.model, self.proc = load(EG2)
        self.files = {}

    def texts(self, texts, length=None):
        kw = {"truncation": True, "max_length": length} if length else {}
        inp = self.proc.tokenizer(texts, padding=True, return_tensors="np", **kw)
        e = self.model(**{k: self.mx.array(v) for k, v in inp.items()}).text_embeds
        self.mx.eval(e)
        return np.array(e)

    def images(self, imgs, budget):
        conv = [[{"role": "user", "content": [{"type": "image", "url": self._file(img)}]}] for img in imgs]
        inp = self.proc.apply_chat_template(conv, tokenize=True, return_dict=True, return_tensors="mlx",
                                            max_soft_tokens=budget)
        e = self.model(**inp).text_embeds
        self.mx.eval(e)
        return np.array(e)

    def _file(self, img):
        # The processor takes paths, so each test image is written once.
        if id(img) not in self.files:
            path = os.path.join(tempfile.gettempdir(), f"p4_{id(img)}.png")
            img.save(path)
            self.files[id(img)] = path
        return self.files[id(img)]

    def mem(self):
        return f"peak={self.mx.get_peak_memory() / 2**30:.2f}GB"


class Llama:
    def __init__(self):
        self.marker = self._get("/props")["media_marker"]
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(EG2)

    def _get(self, path):
        return json.load(urllib.request.urlopen(LLAMA + path))

    def _embed(self, content):
        req = urllib.request.Request(LLAMA + "/embeddings", json.dumps({"content": content}).encode(),
                                     {"Content-Type": "application/json"})
        return np.array([np.asarray(r["embedding"]).reshape(-1) for r in json.load(urllib.request.urlopen(req))])

    def texts(self, texts, length=None):
        if length:
            # Token ids, so every runtime sees the same `length` tokens.
            return self._embed([self.tok(t, truncation=True, max_length=length)["input_ids"] for t in texts])
        return self._embed(texts)

    def images(self, imgs, budget):
        items = []
        for img in imgs:
            buf = io.BytesIO()
            fit(img, budget).save(buf, "PNG")
            items.append({"prompt_string": self.marker, "multimodal_data": [base64.b64encode(buf.getvalue()).decode()]})
        return self._embed(items)

    def mem(self):
        pid = subprocess.run(["pgrep", "-f", "llama-server.*8089"], capture_output=True, text=True, check=True).stdout.split()[0]
        rss = subprocess.run(["ps", "-o", "rss=", "-p", pid], capture_output=True, text=True, check=True).stdout
        return f"server_rss={int(rss) / 2**20:.2f}GB"


def reference():
    import torch
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(EG2, device="cpu", config_kwargs={"audio_config": None}, model_kwargs={"dtype": torch.float32})
    out = {"query": st.encode([QUERY])[0]}
    for budget in [70, 280]:
        st[0].processing_kwargs["image"] = {"max_soft_tokens": budget}
        out.update({f"noise{budget}": st.encode([NOISE])[0], f"doc{budget}": st.encode([doc_image()])[0]})
    return out


def run_parity(rt):
    ref = reference()
    got = {"query": rt.texts([QUERY])[0]}
    for budget in [70, 280]:
        got[f"noise{budget}"] = rt.images([NOISE], budget)[0]
        got[f"doc{budget}"] = rt.images([doc_image()], budget)[0]
    cos = {k: float(np.dot(ref[k], v) / np.linalg.norm(ref[k]) / np.linalg.norm(v)) for k, v in got.items()}
    print("parity " + " ".join(f"{k}={v:.5f}" for k, v in cos.items()))


def run_bench(rt, name):
    for length in [128, 512]:
        for bs in [1, 16]:
            sec = bench(lambda: rt.texts([JA] * bs, length))
            print(f"text {name} len={length} bs={bs}: {sec * 1000:.0f}ms/batch {bs / sec:.1f}docs/s {rt.mem()}")
    for budget in [70, 140, 280]:
        for bs in [1, 8]:
            sec = bench(lambda: rt.images([NOISE] * bs, budget))
            print(f"image {name} budget={budget} bs={bs}: {sec * 1000:.0f}ms/batch {bs / sec:.1f}img/s {rt.mem()}")


if __name__ == "__main__":
    mode, name = sys.argv[1:]
    rt = {"mlx": Mlx, "llama": Llama}[name]()
    {"parity": run_parity, "bench": lambda r: run_bench(r, name)}[mode](rt)
