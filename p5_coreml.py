"""EG2's audio tower on Core ML: uv run --group coreml python p5_coreml.py {convert,bench} {plain,ane}"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers.masking_utils import create_bidirectional_mask
from transformers.models.gemma4 import modeling_gemma4 as g4

EG2 = "google/embeddinggemma-2"
SECONDS = 30   # the feature extractor's cut (REPORT.md 6.3); the converted model takes exactly this much
BUILD = Path("build")


class Tower(torch.nn.Module):
    """Gemma4AudioModel.forward + embed_audio for one unpadded input of a fixed length.

    The attention mask only depends on the length, so it is computed once here: building it inside the trace (vmap)
    is what torch.jit.trace can't do. With `ane`, attention, RMSNorm and the clamps are rewritten into ops the Neural
    Engine runs (AneAttention, ane_rms_norm, ane_clippable_linear); the result is the same up to rounding.
    """

    def __init__(self, am, features, mask, ane):
        super().__init__()
        self.at, self.embed, self.ane = am.audio_tower, am.embed_audio, ane
        self.register_buffer("in_mask", mask)
        c = self.at.config
        with torch.no_grad():
            hs, out_mask = self.at.subsample_conv_projection(features, mask)
            attn = create_bidirectional_mask(
                config=c, inputs_embeds=hs, attention_mask=out_mask,
                and_mask_function=g4.sliding_window_mask_function((c.attention_context_left - 1, c.attention_context_right)))
            blocked = self.at._convert_4d_mask_to_blocked_5d(attn)
        self.register_buffer("attn", blocked)
        if ane:
            self.register_buffer("bias", block_mask_bias(blocked))
            for layer in self.at.layers:
                layer.self_attn.forward = AneAttention(layer.self_attn, self.bias)
            for module in self.at.modules():
                if hasattr(module, "gradient_clipping"):
                    # 1e10 isn't an fp16 number, so its clamps convert to fp32 casts the Neural Engine can't run.
                    # fp16's largest value clips nothing these activations reach (the largest is a few hundred).
                    module.gradient_clipping = float(torch.finfo(torch.float16).max)

    def forward(self, features):
        hs, _ = self.at.subsample_conv_projection(features, self.in_mask)
        pos = self.at.rel_pos_enc(hs)
        for layer in self.at.layers[: self.at.config.num_hidden_layers]:
            hs = layer(hs, attention_mask=self.attn, position_embeddings=pos)
        return self.embed(inputs_embeds=self.at.output_proj(hs))


class AneAttention:
    """Gemma4AudioAttention.forward for batch 1 in rank-4 tensors, without unfold or fp32 casts.

    The original works on [batch, heads, blocks, rows, keys] tensors and builds each block's keys with unfold; the
    Neural Engine runs neither rank-5 matmuls nor unfold (sliding_windows). With one input the batch axis can go, and
    since each block of `chunk` queries sees the `past` keys before it and its own (past == chunk here), its keys are
    the previous block and its own, put side by side."""

    def __init__(self, attn, bias):
        assert attn.max_past_horizon == attn.chunk_size and attn.max_future_horizon == 0
        self.a, self.bias = attn, bias

    def __call__(self, hidden_states, position_embeddings, attention_mask=None):
        a, chunk = self.a, self.a.chunk_size
        length = hidden_states.shape[1]
        blocks = (length + chunk - 1) // chunk

        def split(x):
            # [1, L, H*D] -> [H, blocks, chunk, D], zero-padded at the end like _convert_to_block.
            x = F.pad(x.view(1, length, a.num_heads, a.head_dim), (0, 0, 0, 0, 0, blocks * chunk - length))
            return x.view(blocks, chunk, a.num_heads, a.head_dim).permute(2, 0, 1, 3)

        def context(x):
            # [H, blocks, chunk, D] -> [H, blocks, 2 * chunk, D]: each block's keys are the block before it and itself.
            # (Shifting along the first axis made the Neural Engine compiler fail.)
            return torch.cat([F.pad(x, (0, 0, 0, 0, 1, 0))[:, :-1], x], dim=2)

        q = split(a.q_proj(hidden_states)) * (a.q_scale * F.softplus(a.per_dim_scale))
        k = context(split(a.k_proj(hidden_states) * a.k_scale))
        v = context(split(a.v_proj(hidden_states)))

        rel_k = a.relative_k_proj(position_embeddings).view(-1, a.num_heads, a.head_dim).permute(1, 2, 0)   # [H, D, P]
        bd = q.reshape(a.num_heads, blocks * chunk, a.head_dim) @ rel_k                                     # [H, B*C, P]
        # _rel_shift, one rank lower.
        ctx = a.context_size
        bd = F.pad(bd.view(a.num_heads, blocks, chunk, -1), (0, ctx + 1 - bd.shape[-1]))
        bd = bd.reshape(a.num_heads, blocks, chunk * (ctx + 1))[..., : chunk * ctx].reshape(a.num_heads, blocks, chunk, ctx)

        scores = torch.tanh((q @ k.transpose(-1, -2) + bd) / a.softcap) * a.softcap + self.bias
        out = F.softmax(scores, dim=-1) @ v                                                  # [H, blocks, chunk, D]
        out = out.permute(1, 2, 0, 3).reshape(1, blocks * chunk, -1)[:, :length]
        return a.post(out), None


def block_mask_bias(blocked):
    """[1, blocks, chunk, keys]: 0 where the blocked mask lets a query see a key, a large negative number elsewhere.
    Scores are soft-capped to +-50 before this is added, so -1e4 (still finite in fp16) zeroes them in the softmax."""
    return torch.where(blocked[0] == 0, 0.0, -1e4)


def ane_rms_norm(self, hidden_states):
    """Gemma4RMSNorm without the fp32 cast: layer_norm over [x, -x] is x / sqrt(mean(x^2) + eps) (the mean is 0), and
    the Neural Engine has a layer_norm op, where squaring x in fp16 directly would overflow past |x| = 256."""
    d = hidden_states.shape[-1]
    normed = F.layer_norm(torch.cat([hidden_states, -hidden_states], -1), (2 * d,), eps=self.eps)[..., :d]
    return normed * self.weight if self.with_scale else normed


def ane_clippable_linear(self, hidden_states):
    """Gemma4ClippableLinear with its bounds as Python floats. As 0-d tensors they convert to a cast and a clip with
    runtime bounds, which only the CPU runs; constant bounds keep the clip on the Neural Engine (all are within +-212)."""
    if self.use_clipped_linears:
        hidden_states = torch.clamp(hidden_states, self.input_min.item(), self.input_max.item())
    hidden_states = self.linear(hidden_states)
    if self.use_clipped_linears:
        hidden_states = torch.clamp(hidden_states, self.output_min.item(), self.output_max.item())
    return hidden_states


def load():
    m = SentenceTransformer(EG2, device="cpu", config_kwargs={"vision_config": None}, model_kwargs={"dtype": torch.float32})
    am, fe = m[0].auto_model.eval(), m[0].processor.feature_extractor
    audio = sf.read("data/say/ja_long.wav")[0].astype(np.float32)[: 16000 * SECONDS]
    f = fe([audio], sampling_rate=16000, return_tensors="pt")
    return am, f["input_features"], f["input_features_mask"]


def register_movedim():
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch.ops import _get_inputs
    from coremltools.converters.mil.frontend.torch.torch_op_registry import (
        register_torch_op,
    )

    @register_torch_op
    def movedim(context, node):
        # torch.movedim(x, source, destination) is a transpose; coremltools 9.1.dev1 has no converter for it.
        x, src, dst = _get_inputs(context, node, expected=3)
        src = [int(v) % x.rank for v in np.atleast_1d(src.val)]
        dst = [int(v) % x.rank for v in np.atleast_1d(dst.val)]
        perm = [None] * x.rank
        for s, d in zip(src, dst):
            perm[d] = s
        rest = iter(d for d in range(x.rank) if d not in src)
        context.add(mb.transpose(x=x, perm=[p if p is not None else next(rest) for p in perm], name=node.name))


def convert(variant):
    import coremltools as ct

    am, features, mask = load()
    with torch.no_grad():
        # Before the rewrites below: they patch the model in place.
        ref = am.get_audio_features(features, mask).pooler_output.numpy()
    ane = variant == "ane"
    if ane:
        g4.Gemma4RMSNorm.forward = ane_rms_norm
        g4.Gemma4ClippableLinear.forward = ane_clippable_linear
    tower = Tower(am, features, mask, ane).eval()
    with torch.no_grad():
        mine = tower(features).numpy()
    print(f"{variant}: torch wrapper vs original: max abs diff {np.abs(ref - mine).max():.2e}, "
          f"max |activation| {np.abs(mine).max():.1f}")
    BUILD.mkdir(exist_ok=True)
    np.save(BUILD / "audio_features.npy", features.numpy())
    np.save(BUILD / "audio_ref.npy", ref)
    traced = torch.jit.trace(tower, (features,), strict=False)
    register_movedim()
    ml = ct.convert(traced, inputs=[ct.TensorType("features", shape=features.shape, dtype=np.float32)],
                    outputs=[ct.TensorType("audio_embeds")], minimum_deployment_target=ct.target.macOS15,
                    compute_precision=ct.precision.FLOAT16, convert_to="mlprogram")
    ml.save(BUILD / f"eg2_audio_{variant}.mlpackage")


def bench(variant):
    import coremltools as ct

    x, ref = np.load(BUILD / "audio_features.npy"), np.load(BUILD / "audio_ref.npy")[0].astype(np.float64)
    for unit in ["CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY"]:
        t = time.perf_counter()
        m = ct.models.MLModel(str(BUILD / f"eg2_audio_{variant}.mlpackage"), compute_units=getattr(ct.ComputeUnit, unit))
        load_s = time.perf_counter() - t
        out = m.predict({"features": x})["audio_embeds"][0].astype(np.float64)
        n, t = 0, time.perf_counter()
        while time.perf_counter() - t < 8 or n < 3:
            m.predict({"features": x})
            n += 1
        sec = (time.perf_counter() - t) / n
        cos = (out * ref).sum(-1) / np.linalg.norm(out, axis=-1) / np.linalg.norm(ref, axis=-1)
        pooled = out.mean(0) @ ref.mean(0) / np.linalg.norm(out.mean(0)) / np.linalg.norm(ref.mean(0))
        print(f"coreml {variant} {unit}: load={load_s:.1f}s {sec * 1000:.0f}ms/{SECONDS}s {SECONDS / sec:.0f}x realtime "
              f"1h_audio={3600 * sec / SECONDS:.0f}s nan={bool(np.isnan(out).any())} "
              f"token_cos_mean={np.nanmean(cos):.5f} token_cos_min={np.nanmin(cos):.5f} pooled_cos={pooled:.5f}")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    mode, variant = sys.argv[1:]
    {"convert": convert, "bench": bench}[mode](variant)
