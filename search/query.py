"""Searches the index in Japanese (or any language): uv run python -m search.query "海辺で走る犬の写真" """

import argparse
import time

from search import rank
from search.embed import Embedder, query_prompt
from search.index import DEFAULT_DB
from search.store import Store

KINDS = ["image", "screenshot", "pdf", "doc"]


def search(store, embedder, query, limit=10, kinds=None):
    """[(score, z or None, path, kind, segment kind, loc, excerpt)], best first."""
    mat, segs, files, modalities = store.vectors()
    best = {}
    if len(segs):
        q = embedder.texts([query_prompt(query)])[0]
        best = rank.best_per_file(rank.meaning(mat @ q, modalities), files)
    words = rank.terms(query)
    coverage, word_seg = {}, {}
    for term in words:
        for file, seg in store.keyword_files(term).items():
            coverage[file] = coverage.get(file, 0) + 1 / len(words)
            word_seg.setdefault(file, seg)

    hits = []
    for file, score in sorted(rank.fuse(best, coverage).items(), key=lambda x: -x[1]):
        path, kind = store.file(file)
        if kinds and kind not in kinds:
            continue
        z, row = best.get(file, (None, None))
        # Point at the segment that matched in meaning if it stands out, else at the one that holds the words.
        seg = segs[row] if row is not None and (z >= rank.Z_GATE or file not in word_seg) else word_seg[file]
        seg_kind, loc, text = store.segment(seg)
        hits.append((score, z, path, kind, seg_kind, loc, excerpt(text, words)))
        if len(hits) == limit:
            break
    return hits


def excerpt(text, words, width=60):
    if not text:
        return ""
    text = " ".join(text.split())
    at = next((i for w in words if (i := text.lower().find(w.lower())) >= 0), 0)
    start = max(at - width // 3, 0)
    return ("…" if start else "") + text[start:start + width] + ("…" if start + width < len(text) else "")


def main():
    ap = argparse.ArgumentParser(description="Search what search.index read")
    ap.add_argument("query")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--kind", choices=KINDS, action="append", help="only these kinds of file (repeatable)")
    args = ap.parse_args()

    store = Store(args.db)
    embedder = Embedder(vision=False)   # a query is text: the text part alone loads in ~1s
    start = time.perf_counter()
    hits = search(store, embedder, args.query, args.n, args.kind)
    print(f"{len(hits)} results, {(time.perf_counter() - start) * 1000:.0f}ms")
    for score, z, path, kind, seg_kind, loc, text in hits:
        where = f" p.{loc}" if seg_kind in ("page", "page_image") else ""
        meaning = f"z={z:.1f}" if z is not None else "z=-"
        print(f"{score:5.2f} {meaning:7} {kind:10} {path}{where}")
        if text and seg_kind != "name":
            print(f"      {text}")


if __name__ == "__main__":
    main()
