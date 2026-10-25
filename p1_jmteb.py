import os
import sys
import time

# Uncapped, MPS grows past the M1's 16GB into swap and freezes the desktop instead of raising OOM.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

import mteb
from mteb.cache import ResultCache
from mteb.models import SentenceTransformerEncoderWrapper
from sentence_transformers import SentenceTransformer

EG2 = "google/embeddinggemma-2"
TASKS = [
    "JSTS",
    "JSICK",
    "LivedoorNewsClustering.v2",
    "JapaneseSentimentClassification",
    "NLPJournalTitleAbsRetrieval.V2",
    "JaGovFaqsRetrieval",
]


def no_prefix():
    st = SentenceTransformer(
        EG2,
        config_kwargs={"vision_config": None, "audio_config": None},
        model_kwargs={"dtype": "bfloat16"},
    )
    st.prompts = {}
    st.default_prompt_name = None
    return SentenceTransformerEncoderWrapper(st)


VARIANTS = {
    "eg2-768": lambda: mteb.get_model(EG2),
    "eg2-256": lambda: mteb.get_model(EG2, embed_dim=256),
    "eg2-128": lambda: mteb.get_model(EG2, embed_dim=128),
    "eg2-noprefix": no_prefix,
    "ruri-v3-310m": lambda: mteb.get_model("cl-nagoya/ruri-v3-310m"),
    "me5-large": lambda: mteb.get_model("intfloat/multilingual-e5-large"),
}

if __name__ == "__main__":
    variant = sys.argv[1]
    tasks = mteb.get_tasks(tasks=sys.argv[2:] or TASKS, languages=["jpn"])
    model = VARIANTS[variant]()
    t = time.time()
    mteb.evaluate(
        model,
        tasks,
        # Per-variant cache dirs: eg2 variants share one model name/revision and would overwrite each other.
        cache=ResultCache(cache_path=f"results/{variant}"),
        # 16 OOMs on 8K-token docs (Livedoor, NLPJournal articles) within the M1's 16GB unified memory.
        encode_kwargs={"batch_size": int(os.environ.get("BATCH_SIZE", 4))},
        raise_error=False,
        show_progress_bar=False,
    )
    print(f"{variant} done in {time.time() - t:.0f}s")
