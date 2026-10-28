"""Japanese speech-to-text on a frozen EG2 audio encoder (REPORT.md 13章).

uv run python p6_stt.py features {fleurs-train,fleurs-dev,fleurs-test,reazon-test,reazon-train}
uv run python p6_stt.py zeroshot fleurs-dev
uv run python p6_stt.py train {eg2,free} [fraction]       # e.g. 0.25: a quarter of the clips (REPORT 16, how far data helps)
uv run python p6_stt.py train {eg2,free} [fraction] reazon   # plus ReazonSpeech's 100 hours (REPORT 17)
uv run python p6_stt.py eval {eg2,free}[+reazon][@fraction] {fleurs-test,reazon-test}
uv run --group whisper python p6_stt.py whisper {fleurs-test,reazon-test}
"""

import csv
import glob
import io
import math
import os
import random
import sys
import tarfile
import time
import unicodedata
from itertools import pairwise
from pathlib import Path

# Same cap as the other scripts (REPORT.md 7章).
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch import nn

EG2 = "google/embeddinggemma-2"
OUT = Path("build/stt")
HUB = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
DEVICE = "mps"


# ---- data ---------------------------------------------------------------------------------------------------------

def normalize(text):
    """What CER is measured on: NFKC, no spaces, no punctuation or symbols (FLEURS' normalized column keeps some)."""
    text = unicodedata.normalize("NFKC", text)
    return "".join(c for c in text if not c.isspace() and unicodedata.category(c)[0] not in "PSZ")


def fleurs(split):
    """(id, waveform, text) for FLEURS ja_jp, read straight out of the downloaded tar.gz."""
    root = Path(glob.glob(str(HUB / "datasets--google--fleurs/snapshots/*/data/ja_jp"))[0])
    with open(root / f"{split}.tsv", encoding="utf-8") as f:
        # The normalized transcription (no punctuation).
        text = {r[1]: r[3] for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)}
    with tarfile.open(root / "audio" / f"{split}.tar.gz") as tar:
        for member in tar:
            name = Path(member.name).name
            if member.isfile() and name in text:
                audio, sr = sf.read(io.BytesIO(tar.extractfile(member).read()), dtype="float32")
                assert sr == 16000
                yield name, audio, text[name]


def reazon_test():
    import pyarrow.parquet as pq

    files = sorted(glob.glob(str(HUB / "datasets--japanese-asr--ja_asr.reazonspeech_test/snapshots/*/data/*.parquet")))
    for path in files:
        for i, row in enumerate(pq.read_table(path).to_pylist()):
            audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            assert sr == 16000
            if audio.ndim > 1:
                audio = audio.mean(1)
            yield f"{Path(path).stem}-{i}", audio[: 16000 * 30], row["transcription"]


def reazon_train_files():
    """ReazonSpeech's 100-hour "small" set, as parquet (japanese-asr/whisper_transcriptions.reazonspeech.small)."""
    return sorted(glob.glob(str(HUB / "datasets--japanese-asr--whisper_transcriptions.reazonspeech.small/snapshots/*/small/*.parquet")))


def reazon_train(path):
    import pyarrow.parquet as pq

    for row in pq.read_table(path, columns=["name", "audio", "transcription"]).to_pylist():
        try:
            audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        except sf.LibsndfileError:   # one clip in the set (00a/cd6fd726264c3.flac) does not decode
            print(f"skipped {row['name']}: not readable")
            continue
        assert sr == 16000
        if audio.ndim > 1:
            audio = audio.mean(1)
        yield row["name"], audio[: 16000 * 30], row["transcription"]


SOURCES = {
    "fleurs-train": lambda: fleurs("train"),
    "fleurs-dev": lambda: fleurs("dev"),
    "fleurs-test": lambda: fleurs("test"),
    "reazon-test": reazon_test,
    # 100 hours of audio does not fit in memory at once: one shard per parquet file (reazon-train-00, ...).
    **{f"reazon-train-{i:02}": (lambda p=p: reazon_train(p)) for i, p in enumerate(reazon_train_files())},
}


# ---- features: EG2's audio tokens ------------------------------------------------------------------------------

def features(name):
    """Runs the frozen audio encoder over a dataset and saves its tokens (25 per second, 512-d, EG2's text space)."""
    from sentence_transformers import SentenceTransformer

    if name == "reazon-train":
        for shard in sorted(k for k in SOURCES if k.startswith("reazon-train-")):
            if not (OUT / f"{shard}.npz").exists():   # resumable
                features(shard)
        return

    m = SentenceTransformer(EG2, device=DEVICE, config_kwargs={"vision_config": None},
                            model_kwargs={"dtype": torch.float32})
    am, fe = m[0].auto_model.eval(), m[0].processor.feature_extractor
    items = sorted(SOURCES[name](), key=lambda x: len(x[1]))   # similar lengths batch with little padding
    ids, texts, tokens, start = [], [], [], time.perf_counter()
    for i in range(0, len(items), 8):
        batch = items[i:i + 8]
        f = fe([a for _, a, _ in batch], sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = am.get_audio_features(f["input_features"].to(DEVICE), f["input_features_mask"].to(DEVICE))
        emb, mask = out.pooler_output.float().cpu().numpy(), out.attention_mask.cpu().numpy()
        for (key, _, text), e, keep in zip(batch, emb, mask):
            ids.append(key)
            texts.append(text)
            tokens.append(e[keep].astype(np.float16))
    seconds = sum(len(a) for _, a, _ in items) / 16000
    print(f"features {name}: {len(items)} clips, {seconds / 3600:.2f}h audio, {time.perf_counter() - start:.0f}s")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{name}.npz", ids=np.array(ids), texts=np.array(texts), lengths=np.array([len(t) for t in tokens]),
             tokens=np.concatenate(tokens))


def load_split(name):
    if name == "reazon-train":
        shards = [load_split(Path(p).stem) for p in sorted(glob.glob(str(OUT / "reazon-train-*.npz")))]
        return tuple([x for s in shards for x in s[k]] for k in range(3))
    d = np.load(OUT / f"{name}.npz")
    all_tokens = d["tokens"]   # each d[...] reads the array out of the file again: read it once
    bounds = np.concatenate([[0], np.cumsum(d["lengths"])])
    tokens = [all_tokens[a:b] for a, b in pairwise(bounds)]
    return list(d["ids"]), [normalize(t) for t in d["texts"]], tokens


def reazon_split():
    """ReazonSpeech train minus 300 clips held out for dev (the same ones every run): (train, dev), each (texts, tokens)."""
    _, texts, tokens = load_split("reazon-train")
    dev = set(random.Random(1).sample(range(len(texts)), 300))
    train = [i for i in range(len(texts)) if i not in dev]
    return ([texts[i] for i in train], [tokens[i] for i in train]), ([texts[i] for i in sorted(dev)], [tokens[i] for i in sorted(dev)])


# ---- the reader -------------------------------------------------------------------------------------------------

def char_table():
    """Every single-character Gemma token in Japanese scripts, digits or Latin letters, with EG2's text embedding."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(EG2)
    chars = sorted(c for c in tok.get_vocab() if len(c) == 1 and normalize(c) == c and is_japanese_or_alnum(c))
    ids = tok.convert_tokens_to_ids(chars)
    with safe_open(hf_hub_download(EG2, "model.safetensors"), "pt") as f:
        emb = f.get_slice("language_model.embed_tokens.weight")[:]
    return chars, emb[ids].float()


def is_japanese_or_alnum(c):
    o = ord(c)
    return (0x3041 <= o <= 0x309F or 0x30A0 <= o <= 0x30FF or 0x4E00 <= o <= 0x9FFF or c.isascii() and c.isalnum()
            or c in "々〆ヶ")


class Reader(nn.Module):
    """Audio tokens -> per-frame character scores, for CTC (index 0 is the blank).

    head="eg2": a character's score is the cosine between the frame (projected back into 512-d) and that character's
    frozen EG2 text embedding: the post's "nearest word in the model's own text space", with characters for words.
    Any character in `emb` can be read, including ones the training data never had.
    head="free": an ordinary linear layer over the training characters, the baseline."""

    def __init__(self, head, n_out, emb=None, d=512, layers=2):
        super().__init__()
        self.head = head
        self.conv = nn.Conv1d(512, d, kernel_size=5, padding=2)
        layer = nn.TransformerEncoderLayer(d, 8, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        if head == "eg2":
            self.proj = nn.Linear(d, 512)
            self.blank = nn.Linear(d, 1)
            self.scale = nn.Parameter(torch.tensor(10.0))
            self.register_buffer("emb", F.normalize(emb, dim=-1))
        else:
            self.out = nn.Linear(d, n_out + 1)

    def forward(self, x, emb=None):
        h = F.gelu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        h = self.norm(self.enc(h + sinusoid(h.shape[1], h.shape[2], h.device)))
        if self.head == "eg2":
            emb = self.emb if emb is None else emb
            chars = self.scale * F.normalize(self.proj(h), dim=-1) @ emb.T
            return torch.cat([self.blank(h), chars], -1)
        return self.out(h)


def sinusoid(length, d, device):
    pos = torch.arange(length, device=device)[:, None]
    div = torch.exp(torch.arange(0, d, 2, device=device) * (-math.log(10000.0) / d))
    pe = torch.zeros(length, d, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


def batches(tokens, labels, max_frames=6000, shuffle=True):
    order = sorted(range(len(tokens)), key=lambda i: len(tokens[i]))
    groups, cur = [], []
    for i in order:
        if cur and len(tokens[i]) * (len(cur) + 1) > max_frames:
            groups.append(cur)
            cur = []
        cur.append(i)
    groups.append(cur)
    if shuffle:
        random.shuffle(groups)
    for g in groups:
        x = nn.utils.rnn.pad_sequence([torch.from_numpy(tokens[i].astype(np.float32)) for i in g], batch_first=True)
        yield g, x, torch.tensor([len(tokens[i]) for i in g]), [labels[i] for i in g]


def time_mask(x, lengths, spans=2, width=0.1):
    """SpecAugment's time masking on the audio tokens: a little help with 7 hours of data."""
    x = x.clone()
    for b, n in enumerate(lengths.tolist()):
        for _ in range(spans):
            w = random.randint(0, int(n * width))
            s = random.randint(0, max(n - w, 0))
            x[b, s:s + w] = 0
    return x


def train(head, fraction="1", extra=None):
    """fraction < 1 trains on that share of the clips (picked at random), for as many steps as the full set gets, so
    that only the amount of data differs. It is saved as reader_{head}@{fraction}.pt.

    extra="reazon" adds ReazonSpeech's 100 hours to all of FLEURS train, and fraction then picks from ReazonSpeech.
    The full set gets 12 epochs: 14 times the data, so a similar number of steps per clip would take days.
    It is saved as reader_{head}+reazon@{fraction}.pt; dev is FLEURS dev plus 300 held-out ReazonSpeech clips."""
    fraction = float(fraction)
    name = head + (f"+{extra}" if extra else "") + ("" if fraction == 1 else f"@{fraction:g}")
    epochs = round((12 if extra else 80) / fraction)
    torch.manual_seed(0)
    random.seed(0)
    _, train_text, train_tok = load_split("fleurs-train")
    _, dev_text, dev_tok = load_split("fleurs-dev")
    more_text, more_tok = [], []
    if extra == "reazon":
        (more_text, more_tok), (rdev_text, rdev_tok) = reazon_split()
        dev_text, dev_tok = dev_text + rdev_text, dev_tok + rdev_tok
    elif extra:
        raise SystemExit(f"unknown extra data: {extra}")
    if fraction < 1:
        if extra:
            keep = sorted(random.Random(0).sample(range(len(more_tok)), round(len(more_tok) * fraction)))
            more_text, more_tok = [more_text[i] for i in keep], [more_tok[i] for i in keep]
        else:
            keep = sorted(random.Random(0).sample(range(len(train_tok)), round(len(train_tok) * fraction)))
            train_text, train_tok = [train_text[i] for i in keep], [train_tok[i] for i in keep]
    all_chars, all_emb = char_table()
    # Clips with a character outside EG2's single-character tokens (rare kanji, symbols) or with no text at all:
    # left out of training, for both heads alike.
    vocab = set(all_chars)
    pairs = [(t, x) for t, x in zip(train_text + more_text, train_tok + more_tok) if t and set(t) <= vocab]
    dropped = len(train_tok) + len(more_tok) - len(pairs)
    train_text, train_tok = [t for t, _ in pairs], [x for _, x in pairs]
    seen = sorted(set("".join(train_text)))
    index = {c: i + 1 for i, c in enumerate(seen)}   # 0 is the blank
    emb = all_emb[[all_chars.index(c) for c in seen]] if head == "eg2" else None
    model = Reader(head, len(seen), emb).to(DEVICE)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.01)
    steps = epochs * sum(1 for _ in batches(train_tok, train_text))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
    hours = sum(len(t) for t in train_tok) / 25 / 3600
    print(f"train {name}: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M params, "
          f"{len(train_tok)} clips ({hours:.2f}h; {dropped} left out), {epochs} epochs, "
          f"{len(seen)} characters seen in training, {len(all_chars)} in EG2's vocabulary")
    best, start = 1e9, time.perf_counter()
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for _, x, lengths, texts in batches(train_tok, train_text):
            targets = [torch.tensor([index[c] for c in t]) for t in texts]
            logp = model(time_mask(x, lengths).to(DEVICE)).log_softmax(-1).transpose(0, 1)
            loss = F.ctc_loss(logp, torch.cat(targets).to(DEVICE), lengths, torch.tensor([len(t) for t in targets]),
                              zero_infinity=True)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item()
        if (epoch + 1) % (epochs // 8) == 0 or epoch == epochs - 1:
            cer, hyps = evaluate(model, dev_tok, dev_text, seen)
            n = len(dev_text) - 300   # with extra data, FLEURS dev comes first and ReazonSpeech's 300 clips last
            parts = f", fleurs {corpus_cer(hyps[:n], dev_text[:n]):.3f}, reazon {corpus_cer(hyps[n:], dev_text[n:]):.3f}"
            print(f"  epoch {epoch + 1}: loss {total:.1f}, dev CER {cer:.3f}{parts if extra else ''}, "
                  f"{time.perf_counter() - start:.0f}s")
            if cer < best:
                best = cer
                torch.save({"state": model.state_dict(), "seen": seen}, OUT / f"reader_{name}.pt")


def greedy(logits, lengths, chars):
    out = []
    for row, n in zip(logits.argmax(-1).cpu().numpy(), lengths.tolist()):
        prev, text = 0, []
        for k in row[:n]:
            if k != prev and k != 0:
                text.append(chars[k - 1])
            prev = k
        out.append("".join(text))
    return out


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def evaluate(model, tokens, texts, chars, emb=None):
    """Corpus CER (edit distance over reference length) and the hypotheses, in input order."""
    model.eval()
    hyps = [""] * len(tokens)
    with torch.no_grad():
        for g, x, lengths, _ in batches(tokens, texts, shuffle=False):
            for i, h in zip(g, greedy(model(x.to(DEVICE), emb), lengths, chars)):
                hyps[i] = h
    return corpus_cer(hyps, texts), hyps


def corpus_cer(hyps, texts):
    return sum(edit_distance(h, t) for h, t in zip(hyps, texts)) / sum(len(t) for t in texts)


def run_eval(name, split):
    head = name.split("@")[0].split("+")[0]
    ckpt = torch.load(OUT / f"reader_{name}.pt")
    seen = ckpt["seen"]
    all_chars, all_emb = char_table()
    model = Reader(head, len(seen), all_emb[[all_chars.index(c) for c in seen]] if head == "eg2" else None).to(DEVICE)
    model.load_state_dict(ckpt["state"])
    _, texts, tokens = load_split(split)
    unseen = sum(c not in set(seen) for t in texts for c in t) / sum(len(t) for t in texts)
    cer, hyps = evaluate(model, tokens, texts, seen)
    print(f"eval {name} {split}: CER {cer:.3f} (training characters only; {unseen:.1%} of reference characters unseen)")
    if split == "reazon-test" and glob.glob(str(OUT / "reazon-train-*.npz")):
        # TV commercials air again and again: 526 test sentences are in ReazonSpeech train word for word.
        seen_text = {normalize(t) for p in glob.glob(str(OUT / "reazon-train-*.npz")) for t in np.load(p)["texts"]}
        new = [i for i, t in enumerate(texts) if t not in seen_text]
        print(f"eval {name} {split}: CER {corpus_cer([hyps[i] for i in new], [texts[i] for i in new]):.3f} "
              f"on the {len(new)} clips whose sentence is not in ReazonSpeech train")
    if head == "eg2":
        cer_all, _ = evaluate(model, tokens, texts, all_chars, F.normalize(all_emb, dim=-1).to(DEVICE))
        print(f"eval {name} {split}: CER {cer_all:.3f} (all {len(all_chars)} characters of EG2's vocabulary)")
    for h, t in list(zip(hyps, texts))[:3]:
        print(f"  ref: {t}\n  hyp: {h}")
    # Reader speed on 30 s of audio tokens (750), after the encoder (REPORT.md 12章: 98 ms on the Neural Engine).
    x = torch.randn(1, 750, 512, device=DEVICE)
    with torch.no_grad():
        for _ in range(3):
            model(x).argmax(-1).cpu()
        n, t = 0, time.perf_counter()
        while time.perf_counter() - t < 3:
            model(x).argmax(-1).cpu()
            n += 1
    print(f"eval {name}: reader {(time.perf_counter() - t) / n * 1000:.1f}ms per 30s on {DEVICE}")


def zeroshot(split):
    """No training: each audio token's nearest EG2 character embedding, repeats collapsed."""
    chars, emb = char_table()
    _, texts, tokens = load_split(split)
    emb = F.normalize(emb, dim=-1)
    hyps = []
    for t in tokens:
        idx = (F.normalize(torch.from_numpy(t.astype(np.float32)), dim=-1) @ emb.T).argmax(-1).tolist()
        hyps.append("".join(chars[k] for j, k in enumerate(idx) if j == 0 or k != idx[j - 1]))
    errors = sum(edit_distance(h, t) for h, t in zip(hyps, texts))
    print(f"zeroshot {split}: CER {errors / sum(len(t) for t in texts):.3f}")
    for h, t in list(zip(hyps, texts))[:3]:
        print(f"  ref: {t}\n  hyp: {h[:120]}")


WHISPER = "mlx-community/whisper-large-v3-turbo"


def whisper(split):
    """The same test sets and CER with Whisper large-v3-turbo (MLX), a ready-made STT to compare against."""
    import mlx_whisper

    items = list(SOURCES[split]())
    opts = {"path_or_hf_repo": WHISPER, "language": "ja", "temperature": 0.0, "condition_on_previous_text": False,
            "verbose": None}
    mlx_whisper.transcribe(items[0][1], **opts)   # loads the model, so it isn't timed
    hyps, start = [], time.perf_counter()
    for _, audio, _ in items:
        hyps.append(normalize(mlx_whisper.transcribe(audio, **opts)["text"]))
    took = time.perf_counter() - start
    texts = [normalize(t) for _, _, t in items]
    seconds = sum(len(a) for _, a, _ in items) / 16000
    errors = sum(edit_distance(h, t) for h, t in zip(hyps, texts))
    print(f"whisper {split}: CER {errors / sum(len(t) for t in texts):.3f}, {took:.0f}s for {seconds / 3600:.2f}h "
          f"({seconds / took:.0f}x realtime, clip by clip)")
    for h, t in list(zip(hyps, texts))[:3]:
        print(f"  ref: {t}\n  hyp: {h}")
    # One 30 s clip, as the EG2 numbers are per 30 s (REPORT.md 12-13章).
    audio = np.concatenate([a for _, a, _ in items])[: 16000 * 30]
    n, t = 0, time.perf_counter()
    while time.perf_counter() - t < 8 or n < 3:
        mlx_whisper.transcribe(audio, **opts)
        n += 1
    sec = (time.perf_counter() - t) / n
    print(f"whisper: {sec * 1000:.0f}ms per 30s ({30 / sec:.0f}x realtime)")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    mode, *args = sys.argv[1:]
    {"features": features, "zeroshot": zeroshot, "train": train, "eval": run_eval, "whisper": whisper}[mode](*args)
