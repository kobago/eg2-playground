"""uv run python -m unittest discover tests  (no model needed)"""

import tempfile
import unittest
from itertools import pairwise
from pathlib import Path

import numpy as np

from search import extract, rank
from search.store import DIM, Store


class ChunksTest(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        self.assertEqual(extract.chunks("  吾輩は猫である。 "), ["吾輩は猫である。"])

    def test_pieces_overlap_and_cover_the_text(self):
        text = "".join(f"{i:04d}番目の文です。" for i in range(1000))
        pieces = extract.chunks(text, size=300, overlap=50)
        self.assertTrue(all(len(p) <= 300 for p in pieces))
        self.assertTrue(all(p.endswith("。") for p in pieces[:-1]), "cut after 。 where there is one")
        self.assertTrue(pieces[-1].endswith("0999番目の文です。"))
        for a, b in pairwise(pieces):
            self.assertIn(b[:20], a, "consecutive pieces overlap")

    def test_text_without_breaks_is_still_cut(self):
        pieces = extract.chunks("あ" * 1000, size=300, overlap=50)
        self.assertEqual([len(p) for p in pieces[:3]], [300, 300, 300])
        self.assertEqual(sum(len(p) for p in pieces) - 50 * (len(pieces) - 1), 1000)


class FileKindTest(unittest.TestCase):
    def test_kinds(self):
        cases = {
            "スクリーンショット 2026-10-26 15.00.00.png": "screenshot",
            "Screenshot 2026-10-26 at 15.00.00.png": "screenshot",
            "IMG_0001.HEIC": "image",
            "請求書.pdf": "pdf",
            "メモ.md": "doc",
            "data.csv": None,
        }
        for name, kind in cases.items():
            self.assertEqual(extract.file_kind(Path(name)), kind, name)

    def test_walk_skips_hidden_and_packages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel in ["a.txt", ".hidden.txt", ".git/b.txt", "Photos Library.photoslibrary/c.jpg", "sub/d.pdf", "e.csv"]:
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_text("x")
            self.assertEqual([p.relative_to(root).as_posix() for p in extract.walk(root)], ["a.txt", "sub/d.pdf"])


class RankTest(unittest.TestCase):
    def test_modality_means_are_taken_out(self):
        # Text scores sit higher than image scores; the best image should still stand out as much as the best text.
        rng = np.random.default_rng(0)
        text = 0.8 + 0.01 * rng.standard_normal(50)
        image = 0.6 + 0.01 * rng.standard_normal(50)
        text[0] += 0.1
        image[0] += 0.1
        z = rank.meaning(np.concatenate([text, image]), ["text"] * 50 + ["image"] * 50)
        self.assertGreater(z[0], rank.Z_GATE)
        self.assertGreater(z[50], rank.Z_GATE)
        self.assertLess(abs(z[0] - z[50]), 1.0)

    def test_small_groups_take_the_overall_mean(self):
        z = rank.meaning([0.9, 0.5, 0.5], ["text", "image", "image"])
        self.assertGreater(z[0], z[1])

    def test_fuse(self):
        best = {1: (3.0, 0), 2: (1.0, 1), 3: (0.5, 2)}
        coverage = {2: 1.0, 3: 0.4}
        scores = rank.fuse(best, coverage)
        self.assertEqual(set(scores), {1, 2}, "3 neither stands out nor holds half the words")
        self.assertAlmostEqual(scores[1], 3.0)
        self.assertAlmostEqual(scores[2], 1.0 + rank.WORD_WEIGHT)

    def test_terms(self):
        self.assertEqual(rank.terms("請求書  INV-2026-0042"), ["請求書", "INV-2026-0042"])


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(str(Path(self.tmp.name) / "index.sqlite"))

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def vec(self, i):
        v = np.zeros(DIM, np.float32)
        v[i] = 1
        return v

    def test_replace_vectors_and_keywords(self):
        self.store.replace("/x/請求書.pdf", "pdf", 1.0, 10, [
            ("name", None, None, None, "請求書 x"),
            ("page", 1, "text", self.vec(0), "請求番号 INV-2026-0042 合計 128,000円"),
            ("page_image", 2, "image", self.vec(1), None),
        ])
        self.store.commit()
        mat, segs, files, modalities = self.store.vectors()
        self.assertEqual(mat.shape, (2, DIM))
        self.assertEqual(modalities, ["text", "image"])
        np.testing.assert_array_equal(mat[1], self.vec(1))
        file = int(files[0])
        self.assertEqual(set(self.store.keyword_files("inv-2026")), {file}, "case-insensitive for ASCII")
        self.assertEqual(set(self.store.keyword_files("請求書")), {file}, "found by its name")
        self.assertEqual(self.store.keyword_files("合計"), {file: int(segs[0])}, "two characters: below a trigram")
        self.assertEqual(self.store.keyword_files("50%"), {}, "% is literal")
        self.assertTrue(self.store.unchanged(Path("/x/請求書.pdf"), 1.0, 10))
        self.assertFalse(self.store.unchanged(Path("/x/請求書.pdf"), 2.0, 10))

    def test_replace_and_remove_leave_nothing_behind(self):
        for _ in range(2):
            self.store.replace("/x/a.txt", "doc", 1.0, 1, [("chunk", 0, "text", self.vec(0), "本文")])
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM segments").fetchone(), (1,))
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM fts").fetchone(), (1,))
        self.assertEqual(self.store.paths_under("/x"), ["/x/a.txt"])
        self.assertEqual(self.store.paths_under("/x/a"), [], "a folder prefix, not a string prefix")
        self.store.remove("/x/a.txt")
        for table in ("files", "segments", "fts"):
            self.assertEqual(self.store.db.execute(f"SELECT count(*) FROM {table}").fetchone(), (0,), table)

    def test_other_settings_are_refused(self):
        self.store.db.execute("UPDATE meta SET value = 'other' WHERE key = 'fingerprint'")
        self.store.commit()
        with self.assertRaises(SystemExit):
            Store(str(Path(self.tmp.name) / "index.sqlite"))


if __name__ == "__main__":
    unittest.main()
