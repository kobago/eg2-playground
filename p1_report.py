import glob
import json
import os

from p1_jmteb import TASKS, VARIANTS

rows = {}
for v in VARIANTS:
    for path in glob.glob(f"results/{v}/**/*.json", recursive=True):
        task = os.path.basename(path)[:-5]
        if task not in TASKS:
            continue
        d = json.load(open(path))
        scores = [s["main_score"] for split in d["scores"].values() for s in split]
        rows.setdefault(v, {})[task] = (sum(scores) / len(scores) * 100, d["evaluation_time"])

print("| task | " + " | ".join(rows) + " |")
print("|---|" + "---|" * len(rows))
for t in TASKS:
    print(f"| {t} | " + " | ".join(f"{rows[v][t][0]:.2f}" if t in rows[v] else "-" for v in rows) + " |")
done = [t for t in TASKS if all(t in rows[v] for v in rows)]
print("| **mean** | " + " | ".join(f"**{sum(rows[v][t][0] for t in done) / len(done):.2f}**" for v in rows) + " |")
print("| eval time (s) | " + " | ".join(f"{sum(x[1] for x in rows[v].values()):.0f}" for v in rows) + " |")
