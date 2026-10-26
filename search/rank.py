import numpy as np

# Following DigUp's Searcher: a file counts as a meaning match when its best segment stands out by `Z_GATE`
# standard deviations for this query, or as a keyword match when it holds at least half of the query's words.
Z_GATE = 2.0
WORD_WEIGHT = 2.0
MIN_GROUP = 20      # a modality with fewer vectors than this takes the overall mean
MIN_DEVIATION = 0.02


def meaning(scores, modalities):
    """Cosine scores on one scale per query. Text vectors score higher than image vectors for the same topic, so
    each modality's mean is taken out before the z-score."""
    scores = np.asarray(scores, np.float64)
    modalities = np.asarray(modalities)
    centered = scores.copy()
    overall = scores.mean() if len(scores) else 0.0
    for m in set(modalities.tolist()):
        group = modalities == m
        centered[group] -= scores[group].mean() if group.sum() >= MIN_GROUP else overall
    deviation = max(centered.std(), MIN_DEVIATION) if len(scores) else 1.0
    return (centered - centered.mean()) / deviation


def best_per_file(z, files):
    """{file: (z, row)} for each file's best-scoring row."""
    best = {}
    for row in np.argsort(-z):
        best.setdefault(int(files[row]), (float(z[row]), int(row)))
    return best


def fuse(best, coverage):
    """{file: score} for the files that qualify. `coverage`: {file: share of the query's words it contains}."""
    candidates = {f for f, (z, _) in best.items() if z >= Z_GATE} | {f for f, c in coverage.items() if c >= 0.5}
    return {f: max(best.get(f, (0.0, None))[0], 0.0) + WORD_WEIGHT * coverage.get(f, 0.0) ** 1.5 for f in candidates}


def terms(query):
    """The words of a query. Japanese isn't split: "請求書 INV-2026-0042" gives two terms, a sentence gives one."""
    return [t for t in query.split() if t]
