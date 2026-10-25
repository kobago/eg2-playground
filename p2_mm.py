import os
import sys
import time

# Uncapped, MPS grows past the M1's 16GB into swap and freezes the desktop instead of raising OOM.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

import mteb
from mteb.cache import ResultCache

EG2 = "google/embeddinggemma-2"
IMAGE_TASKS = ["XFlickr30kCoT2IRetrieval", "XFlickr30kCoI2TRetrieval", "JinaVDRJDocQARetrieval"]
AUDIO_TASKS = [
    "FleursT2ARetrieval.v2",
    "FleursA2TRetrieval.v2",
    "CommonVoiceMini21T2ARetrieval",
    "CommonVoiceMini21A2TRetrieval",
]
BUDGET_TASKS = ["XFlickr30kCoT2IRetrieval"]


def image_budget(soft_tokens):
    def load():
        model = mteb.get_model(EG2, use_image=True)
        model.model[0].processing_kwargs["image"] = {"max_soft_tokens": soft_tokens}
        # The wrapper's fixed 380 would truncate image blocks larger than the default 280 budget.
        model.model.max_seq_length = soft_tokens + 100
        return model

    return load, BUDGET_TASKS


VARIANTS = {
    "eg2-image": (lambda: mteb.get_model(EG2, use_image=True), IMAGE_TASKS),
    "eg2-audio": (lambda: mteb.get_model(EG2, use_audio=True), AUDIO_TASKS),
    "eg2-image-70": image_budget(70),
    "eg2-image-1120": image_budget(1120),
}

if __name__ == "__main__":
    variant = sys.argv[1]
    load, task_names = VARIANTS[variant]
    tasks = mteb.get_tasks(tasks=task_names, languages=["jpn", "eng"])
    model = load()
    t = time.time()
    mteb.evaluate(
        model,
        tasks,
        cache=ResultCache(cache_path=f"results/{variant}"),
        encode_kwargs={"batch_size": int(os.environ.get("BATCH_SIZE", 4))},
        raise_error=False,
        show_progress_bar=False,
    )
    print(f"{variant} done in {time.time() - t:.0f}s")
