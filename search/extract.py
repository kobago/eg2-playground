import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
DOC_EXT = {".txt", ".md"}
PDF_EXT = {".pdf"}
SKIP_DIRS = {"node_modules", "__pycache__", "venv", "Library", "Applications"}
# macOS packages look like folders; a Photos library would be read as thousands of loose files.
SKIP_SUFFIXES = (".app", ".photoslibrary", ".bundle", ".framework")
# macOS names screenshots like this in Japanese and English.
SCREENSHOT = re.compile(r"^(スクリーンショット|Screenshot|Screen Shot)")

# Vision soft tokens. Photos: 70 beat the default 280 on XFlickr (REPORT.md 5.3) and is 4.5x faster.
# Screenshots and PDF pages hold small text, so they keep 280 (JDocQA, 5.3).
PHOTO_BUDGET = 70
SCREENSHOT_BUDGET = 280
PAGE_BUDGET = 280
MIN_IMAGE_SIDE = 256      # icons and thumbnails aren't worth a vector
MIN_PAGE_TEXT = 200       # fewer characters: a scan or a slide, so the rendered page is embedded instead
CHUNK_CHARS = 1800        # DigUp's size: ~450 tokens of English; Japanese runs closer to a token per 1-2 characters
CHUNK_OVERLAP = 200
MAX_PAGES = 300
MAX_CHUNKS = 2000


@dataclass
class Segment:
    kind: str                 # "image", "page" (a PDF page's text), "page_image", "chunk"
    loc: int | None = None    # PDF page (from 1) or chunk number (from 0)
    text: str | None = None   # what's embedded for text segments, and what keyword search reads
    image: Image.Image | None = None
    budget: int | None = None


def file_kind(path):
    ext = path.suffix.lower()
    if ext in IMAGE_EXT:
        return "screenshot" if SCREENSHOT.match(path.name) else "image"
    if ext in PDF_EXT:
        return "pdf"
    if ext in DOC_EXT:
        return "doc"
    return None


def walk(root):
    """Files under `root` that can be indexed. Hidden files and folders are skipped."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if not d.startswith(".") and d not in SKIP_DIRS and not d.endswith(SKIP_SUFFIXES))
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if not name.startswith(".") and file_kind(path) and path.is_file():
                yield path


def name_text(path):
    """The file name and its folders, for keyword search only."""
    return " ".join([path.stem, *path.parent.parts[-3:]])


def segments(path, kind):
    if kind in ("image", "screenshot"):
        img = open_image(path)
        if img is None or min(img.size) < MIN_IMAGE_SIDE:
            return []
        return [Segment("image", image=img, budget=SCREENSHOT_BUDGET if kind == "screenshot" else PHOTO_BUDGET)]
    if kind == "pdf":
        return pdf_segments(path)
    if kind == "doc":
        text = path.read_text(encoding="utf-8", errors="replace")
        return [Segment("chunk", loc=i, text=c) for i, c in enumerate(chunks(text)[:MAX_CHUNKS])]
    return []


def open_image(path):
    try:
        if path.suffix.lower() in (".heic", ".heif"):
            # Pillow can't read HEIC; macOS's sips can, without another dependency.
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "image.jpg"
                subprocess.run(["sips", "-s", "format", "jpeg", str(path), "--out", str(out)],
                               capture_output=True, check=True)
                return load_rgb(out)
        return load_rgb(path)
    except (OSError, subprocess.CalledProcessError, Image.DecompressionBombError):
        return None


def load_rgb(path):
    with Image.open(path) as img:
        img.seek(0)   # the first frame of a GIF
        return ImageOps.exif_transpose(img).convert("RGB")


def pdf_segments(path):
    import pypdfium2 as pdfium

    out = []
    try:
        pdf = pdfium.PdfDocument(path)
    except pdfium.PdfiumError:
        return []
    try:
        for i in range(min(len(pdf), MAX_PAGES)):
            page = pdf[i]
            text = page.get_textpage().get_text_range().strip()
            if len(text) >= MIN_PAGE_TEXT:
                out += [Segment("page", loc=i + 1, text=c) for c in chunks(text)]
            else:
                # At 72dpi an A4 page is smaller than the 280-token budget (~800x800px) and would be upscaled.
                img = page.render(scale=1.5).to_pil().convert("RGB")
                out.append(Segment("page_image", loc=i + 1, text=text or None, image=img, budget=PAGE_BUDGET))
    finally:
        pdf.close()
    return out


def chunks(text, size=CHUNK_CHARS, overlap=CHUNK_OVERLAP):
    """~`size`-character pieces that overlap by ~`overlap`, cut after a line break or "。" where there is one."""
    text = re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n")).strip()
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Look back over the last quarter of the piece for a natural break.
            window = text[start + size * 3 // 4:end]
            cut = max(window.rfind("\n"), window.rfind("。"), window.rfind(". "))
            if cut >= 0:
                end = start + size * 3 // 4 + cut + 1
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return out
