import sqlite3

import numpy as np

DIM = 768   # search.embed.DIM, repeated so the index can be read without loading MLX

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, kind TEXT NOT NULL, mtime REAL NOT NULL, size INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY, file INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    kind TEXT NOT NULL, loc INTEGER, modality TEXT, vec BLOB
);
CREATE INDEX IF NOT EXISTS segments_file ON segments(file);
-- trigram: Japanese has no spaces between words, so the default tokenizer would see a sentence as one word.
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(body, tokenize = 'trigram');
"""
# What made the vectors. An index never mixes settings: change this and index from scratch.
FINGERPRINT = "eg2-914f7f89 mlx-bf16 photo70 shot280 page280 chunk1800/200"


class Store:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        found = self.db.execute("SELECT value FROM meta WHERE key = 'fingerprint'").fetchone()
        if found is None:
            self.db.execute("INSERT INTO meta VALUES ('fingerprint', ?)", (FINGERPRINT,))
            self.db.commit()
        elif found[0] != FINGERPRINT:
            raise SystemExit(f"{path} was made with '{found[0]}', not '{FINGERPRINT}': delete it and index again")

    def unchanged(self, path, mtime, size):
        row = self.db.execute("SELECT mtime, size FROM files WHERE path = ?", (str(path),)).fetchone()
        return row is not None and row == (mtime, size)

    def replace(self, path, kind, mtime, size, rows):
        """Replaces a file's segments. `rows`: (kind, loc, modality, vector or None, keyword text or None)."""
        self.remove(path)
        file = self.db.execute("INSERT INTO files (path, kind, mtime, size) VALUES (?, ?, ?, ?)",
                               (str(path), kind, mtime, size)).lastrowid
        for seg_kind, loc, modality, vec, text in rows:
            blob = None if vec is None else np.asarray(vec, np.float16).tobytes()
            seg = self.db.execute("INSERT INTO segments (file, kind, loc, modality, vec) VALUES (?, ?, ?, ?, ?)",
                                  (file, seg_kind, loc, modality, blob)).lastrowid
            if text:
                self.db.execute("INSERT INTO fts (rowid, body) VALUES (?, ?)", (seg, text))

    def remove(self, path):
        self.db.execute("DELETE FROM fts WHERE rowid IN (SELECT s.id FROM segments s JOIN files f ON f.id = s.file "
                        "WHERE f.path = ?)", (str(path),))
        self.db.execute("DELETE FROM files WHERE path = ?", (str(path),))

    def paths_under(self, root):
        prefix = str(root).rstrip("/") + "/"
        return [r[0] for r in self.db.execute("SELECT path FROM files WHERE substr(path, 1, ?) = ?",
                                              (len(prefix), prefix))]

    def commit(self):
        self.db.commit()

    def vectors(self):
        """All vectors as float32 (rows x 768), with each row's segment id, file id and modality."""
        rows = self.db.execute("SELECT id, file, modality, vec FROM segments WHERE vec IS NOT NULL ORDER BY id").fetchall()
        if not rows:
            return np.zeros((0, DIM), np.float32), np.zeros(0, int), np.zeros(0, int), []
        mat = np.frombuffer(b"".join(r[3] for r in rows), np.float16).reshape(len(rows), DIM).astype(np.float32)
        return mat, np.array([r[0] for r in rows]), np.array([r[1] for r in rows]), [r[2] for r in rows]

    def keyword_files(self, term):
        """Files whose name or text contains `term` (case-insensitive for ASCII): {file: first matching segment}."""
        like = "%" + term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        out = {}
        for seg, file in self.db.execute("SELECT s.id, s.file FROM fts JOIN segments s ON s.id = fts.rowid "
                                         "WHERE fts.body LIKE ? ESCAPE '\\' ORDER BY s.id", (like,)):
            out.setdefault(file, seg)
        return out

    def file(self, file_id):
        return self.db.execute("SELECT path, kind FROM files WHERE id = ?", (int(file_id),)).fetchone()

    def segment(self, seg_id):
        """(kind, loc, keyword text) of a segment."""
        return self.db.execute("SELECT s.kind, s.loc, fts.body FROM segments s LEFT JOIN fts ON fts.rowid = s.id "
                               "WHERE s.id = ?", (int(seg_id),)).fetchone()
