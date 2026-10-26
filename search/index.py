"""Indexes folders: uv run python -m search.index ~/Pictures ~/Documents"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

from search import extract
from search.embed import Embedder, document_prompt
from search.store import Store

DEFAULT_DB = "search.sqlite"
FLUSH_IMAGES = 16   # embed once this many images are waiting, so batches stay full across small files
FLUSH_TEXTS = 64


class Indexer:
    def __init__(self, store, embedder):
        self.store, self.embedder = store, embedder
        self.pending = []   # (path, kind, mtime, size, segments)
        self.done = 0

    def add(self, path, kind, mtime, size, segments):
        self.pending.append((path, kind, mtime, size, segments))
        images = sum(s.image is not None for p in self.pending for s in p[4])
        texts = sum(s.image is None for p in self.pending for s in p[4])
        if images >= FLUSH_IMAGES or texts >= FLUSH_TEXTS:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        segs = [s for p in self.pending for s in p[4]]
        vecs = {}
        by_budget = defaultdict(list)
        for i, s in enumerate(segs):
            if s.image is not None:
                by_budget[s.budget].append(i)
        for budget, idx in by_budget.items():
            for i, v in zip(idx, self.embedder.images([segs[i].image for i in idx], budget)):
                vecs[i] = v
        text_idx = [i for i, s in enumerate(segs) if s.image is None]
        titles = [p[0].stem for p in self.pending for _ in p[4]]
        prompts = [document_prompt(titles[i], segs[i].text) for i in text_idx]
        for i, v in zip(text_idx, self.embedder.texts(prompts)):
            vecs[i] = v

        n = 0
        for path, kind, mtime, size, file_segs in self.pending:
            rows = [("name", None, None, None, extract.name_text(path))]
            for s in file_segs:
                rows.append((s.kind, s.loc, "image" if s.image is not None else "text", vecs[n], s.text))
                n += 1
            self.store.replace(path, kind, mtime, size, rows)
        self.store.commit()
        self.done += len(self.pending)
        self.pending = []


def main():
    ap = argparse.ArgumentParser(description="Index photos, screenshots, PDFs and text notes for search.query")
    ap.add_argument("folders", nargs="+", type=Path)
    ap.add_argument("--db", default=DEFAULT_DB)
    args = ap.parse_args()

    store = Store(args.db)
    indexer = Indexer(store, Embedder())
    start, skipped, empty, read = time.perf_counter(), 0, 0, 0
    for root in args.folders:
        root = root.expanduser().resolve()
        seen = set()
        for path in extract.walk(root):
            seen.add(str(path))
            st = path.stat()
            if store.unchanged(path, st.st_mtime, st.st_size):
                skipped += 1
                continue
            kind = extract.file_kind(path)
            segs = extract.segments(path, kind)
            empty += not segs   # still indexed, by its name, so it isn't read again until it changes
            indexer.add(path, kind, st.st_mtime, st.st_size, segs)
            read += 1
            if read % 50 == 0:
                print(f"{read} files, {time.perf_counter() - start:.0f}s", file=sys.stderr)
        indexer.flush()
        gone = [p for p in store.paths_under(root) if p not in seen]
        for p in gone:
            store.remove(p)
        store.commit()
        if gone:
            print(f"{root}: removed {len(gone)} deleted files", file=sys.stderr)
    print(f"indexed {indexer.done} files ({empty} by name only), {skipped} unchanged, "
          f"{time.perf_counter() - start:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
