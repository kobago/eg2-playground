import mlx.core as mx
import numpy as np
from mlx_vlm.embedding_loader import load_embedding_model
from mlx_vlm.models.embedding_gemma2 import EmbeddingGemma2Processor
from mlx_vlm.utils import get_model_path, load_config

EG2 = "google/embeddinggemma-2"
DIM = 768
TEXT_BATCH = 16
IMAGE_BATCH = 8   # 280 soft tokens x 8 peaked at 5.0GB in p4_runtime.py
# MLX's limit is a soft cap; it keeps the index run from growing into swap while the Mac is in use.
mx.set_memory_limit(8 << 30)


def query_prompt(text):
    return f"task: search result | query: {text}"


def document_prompt(title, text):
    return f"title: {title or 'none'} | text: {text}"


class Embedder:
    """EG2 on MLX (REPORT.md 10章). Vectors are L2-normalized float32."""

    def __init__(self, vision=True):
        path = get_model_path(EG2)
        # mlx-vlm's own processor: transformers' one (what load_processor returns) needs torch.
        self.proc = EmbeddingGemma2Processor.from_pretrained(path)
        config = load_config(path)
        config["audio_config"] = None
        if not vision:
            config["vision_config"] = None
        self.model = load_embedding_model(path, config=config)

    def texts(self, texts):
        out = []
        for i in range(0, len(texts), TEXT_BATCH):
            inp = self.proc.tokenizer(texts[i:i + TEXT_BATCH], padding=True, truncation=True, max_length=2048,
                                      return_tensors="np")
            out.append(self._run({k: mx.array(v) for k, v in inp.items()}))
        return np.concatenate(out) if out else np.zeros((0, DIM), np.float32)

    def images(self, images, budget):
        out = []
        for i in range(0, len(images), IMAGE_BATCH):
            inp = self.proc(images=[[img] for img in images[i:i + IMAGE_BATCH]], return_tensors="mlx",
                            images_kwargs={"max_soft_tokens": budget})
            out.append(self._run(inp))
        return np.concatenate(out) if out else np.zeros((0, DIM), np.float32)

    def _run(self, inputs):
        e = self.model(**inputs).text_embeds
        mx.eval(e)
        return np.array(e, dtype=np.float32)
