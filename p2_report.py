import glob
import json
import os

from p2_mm import VARIANTS

print("| variant | task | subset | main_score | eval time (s) |")
print("|---|---|---|---|---|")
for v in VARIANTS:
    for path in sorted(glob.glob(f"results/{v}/**/*.json", recursive=True)):
        task = os.path.basename(path)[:-5]
        if task == "model_meta":
            continue
        d = json.load(open(path))
        for split in d["scores"].values():
            for s in split:
                print(f"| {v} | {task} | {s['hf_subset']} | {s['main_score'] * 100:.2f} | {d['evaluation_time']:.0f} |")
