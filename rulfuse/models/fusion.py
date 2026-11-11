"""RUL-Fuse's two fusion candidates, F1 and F0 (PROJECT_PLAN.md §6.3, §6.4, D13, D21;
part P11).

**Token encoders** (§6.3). They run once per forward on the units' base tokens, before the
tokens are gathered into the augmented sequences.

* Signal, per channel: a learned per-feature affine on the (fixed-normalized) inputs, then a
  rank-``rank`` linear map shared by all channels, then a linear map to d, plus the channel's
  descriptor embedding (sensor + axis + derivation). Per sequence it is averaged over the kept
  channels (channel dropout), then an MLP. A missing segment or a dropped token gets a learned
  mask vector.
* Image: the same affine and shared low-rank projection per encoder token, plus kind and
  token-position embeddings, then attention pooling to one token. The HI readout is added
  through a linear map.

**Usage features U** are added to every signal token when ``usage`` is on: Fourier features
of τ/H and of the seconds into the recording.

**Relative time.** Each attention head adds a learned bias b_h(Δτ), a piecewise-linear
function of log(1 + Δτ/10 s) on ``time_buckets`` knots. This is a cheaper parametrisation
of the plan's MLP_h, and it is the only time the sensing streams see.

**Causality** (P11), with τ_q the time of the query:

* a signal token at τ_q reads signal tokens with τ ≤ τ_q and images available strictly
  before τ_q (D9);
* an image token reads tokens up to its own availability time.

So no output at τ depends on anything after τ: changing a recording's image changes nothing
up to the end of that recording. ``history_recordings`` = k restricts every query to the
last k recordings and turns the memory off (the Gate-2b control, §7.5).

* **F1**: one causal transformer over the signal and image tokens, with the memory state
  (D21) concatenated before the heads.
* **F0**: three experts, U (usage MLP), S (causal transformer over signal tokens, plus the
  memory) and I (a GRU over image tokens whose mean falls with the time since the last
  image), fused by a generalized product of experts. The gate weights α_m form a softmax
  over the present experts.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rulfuse.config import FitConfig
from rulfuse.models.heads import StudentTHead, student_t_variance
from rulfuse.models.memory import CausalMemory

TOL = 1e-6
USAGE_DIM = 12


def usage_features(tau: Tensor, elapsed: Tensor, horizon_s: float) -> Tensor:
    x = tau / horizon_s
    e = elapsed / (horizon_s / 15.0)
    parts = [x, e]
    for j in range(4):
        w = math.pi / 2**j
        parts += [torch.sin(w * x), torch.cos(w * x)]
    parts += [torch.sin(math.pi / 2 * e), torch.cos(math.pi / 2 * e)]
    return torch.stack(parts, dim=-1)


class Affine(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.weight + self.bias


class SignalEncoder(nn.Module):
    def __init__(self, in_dim: int, channels: list[tuple[str, str, str]], config: FitConfig):
        super().__init__()
        d = config.d_model
        self.affine = Affine(in_dim)
        self.down = nn.Linear(in_dim, config.rank, bias=False)
        self.up = nn.Linear(config.rank, d)
        vocab = [sorted({c[i] for c in channels}) for i in range(3)]
        self.register_buffer(
            "channel_ids",
            torch.tensor([[vocab[i].index(c[i]) for i in range(3)] for c in channels]),
            persistent=False,
        )
        self.embed = nn.ModuleList(nn.Embedding(len(v), d) for v in vocab)
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.mask = nn.Parameter(torch.zeros(d))
        self.noise = config.feature_noise if config.regularize else 0.0

    def base(self, x: Tensor) -> Tensor:
        """(N, C, F) -> (N, C, d)."""
        if self.training and self.noise > 0:
            x = x + self.noise * torch.randn_like(x)
        z = self.up(self.down(self.affine(x)))
        ids = self.channel_ids
        return z + sum(self.embed[i](ids[:, i]) for i in range(3))[None]  # type: ignore[index]

    def pool(self, z: Tensor, keep: Tensor, present: Tensor) -> Tensor:
        """(B, L, C, d) gathered, (B, C) kept channels, (B, L) present -> (B, L, d)."""
        w = keep.to(z.dtype)[:, None, :, None]
        pooled = (z * w).sum(2) / w.sum(2).clamp_min(1.0)
        out = self.mlp(pooled)
        return torch.where(present[..., None], out, self.mask.expand_as(out))


class ImageEncoder(nn.Module):
    def __init__(self, in_dim: int, tokens: int, kinds: int, has_hi: bool, config: FitConfig):
        super().__init__()
        d = config.d_model
        self.affine = Affine(in_dim)
        self.down = nn.Linear(in_dim, config.rank, bias=False)
        self.up = nn.Linear(config.rank, d)
        self.position = nn.Parameter(torch.zeros(tokens, d))
        self.kind = nn.Embedding(max(kinds, 1), d)
        self.query = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.readout = nn.Linear(1, d) if has_hi else None
        self.noise = config.feature_noise if config.regularize else 0.0

    def base(self, x: Tensor, kind: Tensor, readout: Tensor) -> Tensor:
        """(N, P, G) -> (N, d)."""
        if x.shape[0] == 0:
            return x.new_zeros(0, self.up.out_features)
        if self.training and self.noise > 0:
            x = x + self.noise * torch.randn_like(x)
        h = self.up(self.down(self.affine(x))) + self.position[None] + self.kind(kind)[:, None]
        scores = torch.softmax(h @ self.query / math.sqrt(h.shape[-1]), dim=1)
        out = (scores[..., None] * h).sum(1)
        if self.readout is not None:
            out = out + self.readout(readout[:, None])
        return out


class RelativeTimeBias(nn.Module):
    def __init__(self, heads: int, buckets: int, max_s: float) -> None:
        super().__init__()
        self.table = nn.Parameter(torch.zeros(buckets, heads))
        self.buckets = buckets
        self.scale = (buckets - 1) / math.log1p(max_s / 10.0)

    def forward(self, delta: Tensor) -> Tensor:
        """(B, Lq, Lk) seconds >= 0 -> (B, heads, Lq, Lk)."""
        u = (torch.log1p(delta.clamp_min(0.0) / 10.0) * self.scale).clamp(0, self.buckets - 1)
        lo = u.floor().long()
        hi = (lo + 1).clamp_max(self.buckets - 1)
        frac = (u - lo.to(u.dtype))[..., None]
        bias = self.table[lo] * (1 - frac) + self.table[hi] * frac
        return bias.permute(0, 3, 1, 2)


class Block(nn.Module):
    def __init__(self, d: int, heads: int, ffn: int, dropout: float) -> None:
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, bias: Tensor) -> Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, n, 3, self.heads, d // self.heads).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        att = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=self.dropout if self.training else 0.0
        )
        x = x + self.drop(self.out(att.transpose(1, 2).reshape(b, n, d)))
        return x + self.drop(self.ff(self.ln2(x)))


class CausalTransformer(nn.Module):
    def __init__(self, config: FitConfig, horizon_s: float) -> None:
        super().__init__()
        d = config.d_model
        self.blocks = nn.ModuleList(
            Block(d, config.heads, config.ffn, config.dropout) for _ in range(config.layers)
        )
        self.bias = RelativeTimeBias(config.heads, config.time_buckets, 8.0 * horizon_s)
        self.ln = nn.LayerNorm(d)

    def forward(self, x: Tensor, delta: Tensor, allowed: Tensor) -> Tensor:
        bias = self.bias(delta)
        bias = bias.masked_fill(~allowed[:, None], float("-inf"))
        for block in self.blocks:
            x = block(x, bias)
        return self.ln(x)


def attention_mask(batch: dict[str, Tensor], history: int, images: bool) -> tuple[Tensor, Tensor]:
    """(Δτ, allowed) over [signal tokens, image tokens] as queries and keys (P11)."""
    tau, valid, entry = batch["tau"], batch["valid"], batch["entry"]
    if images:
        t = torch.cat([tau, batch["img_time"]], 1)
        is_img = torch.cat([torch.zeros_like(valid), torch.ones_like(batch["img_valid"])], 1)
        ok = torch.cat([valid, batch["img_valid"]], 1)
        ent = torch.cat([entry, batch["img_entry"] + 1], 1)  # an image belongs to the next one
    else:
        t, is_img, ok, ent = tau, torch.zeros_like(valid), valid, entry
    tq, tk = t[:, :, None], t[:, None, :]
    qi, ki = is_img[:, :, None], is_img[:, None, :]
    signal_key = tk <= tq + TOL
    image_key = torch.where(qi, tk <= tq + TOL, tk < tq - TOL)
    allowed = torch.where(ki, image_key, signal_key) & ok[:, None, :]
    if history > 0:
        allowed &= ent[:, None, :] >= ent[:, :, None] - (history - 1)
    eye = torch.eye(t.shape[1], dtype=torch.bool, device=t.device)[None]
    allowed = allowed | eye
    return (tq - tk), allowed


class RULFuse(nn.Module):
    def __init__(
        self,
        config: FitConfig,
        horizon_s: float,
        signal_dim: int,
        channels: list[tuple[str, str, str]],
        image_dim: int,
        image_tokens: int,
        kinds: int,
        has_hi: bool,
    ) -> None:
        super().__init__()
        self.config, self.horizon = config, horizon_s
        d = config.d_model
        self.signal = SignalEncoder(signal_dim, channels, config)
        self.image = ImageEncoder(image_dim, image_tokens, kinds, has_hi, config)
        self.modality = nn.Parameter(torch.zeros(2, d))
        self.usage = nn.Linear(USAGE_DIM, d)
        self.oracle = nn.Linear(1, d) if config.oracle_hi else None
        self.use_memory = config.memory and config.history_recordings == 0
        self.memory = CausalMemory(d, 64) if self.use_memory else None
        width = d + (64 if self.use_memory else 0)
        if config.fusion == "F1":
            self.backbone = CausalTransformer(config, horizon_s)
            self.head = StudentTHead(width)
            self.hi_head = nn.Linear(width, 1)
        else:
            self.backbone = CausalTransformer(config, horizon_s)  # the S expert
            self.u_mlp = nn.Sequential(nn.Linear(USAGE_DIM, d), nn.GELU(), nn.Linear(d, d))
            self.u_head = StudentTHead(d)
            self.s_head = StudentTHead(width)
            self.hi_head = nn.Linear(width, 1)
            self.i_cell = nn.GRUCell(d, config.image_hidden)
            self.i_head = StudentTHead(config.image_hidden)
            self.gate = nn.Sequential(nn.Linear(3 + USAGE_DIM, 32), nn.GELU(), nn.Linear(32, 3))

    # ------------------------------------------------------------------ shared pieces
    def tokens(self, batch: dict[str, Tensor], tensors) -> tuple[Tensor, Tensor | None, Tensor]:
        z = self.signal.base(tensors.signal)[batch["seg"]]
        s = self.signal.pool(z, batch["channel_keep"], batch["present"])
        u = usage_features(batch["tau"], batch["elapsed"], self.horizon)
        img = None
        if self.config.image_kinds:
            base = self.image.base(tensors.images, tensors.img_kind, tensors.img_readout)
            if base.shape[0]:
                img = base[batch["img"]] + self.modality[1]
        return s + self.modality[0], img, u

    def _memory(self, x: Tensor, batch: dict[str, Tensor]) -> Tensor | None:
        if self.memory is None:
            return None
        return self.memory(x, batch["lengths"])

    # ------------------------------------------------------------------ forward
    def forward(self, batch: dict[str, Tensor], tensors) -> dict[str, Tensor]:
        s, img, u = self.tokens(batch, tensors)
        x = s
        if self.config.usage and self.config.fusion == "F1":
            x = x + self.usage(u)
        if self.oracle is not None:
            x = x + self.oracle(batch["oracle"][..., None])
        n = x.shape[1]
        if self.config.fusion == "F1":
            seq = x if img is None else torch.cat([x, img], 1)
            delta, allowed = attention_mask(batch, self.config.history_recordings, img is not None)
            h = self.backbone(seq, delta, allowed)[:, :n]
            m = self._memory(x, batch)
            h = h if m is None else torch.cat([h, m], -1)
            mu, sigma, nu = self.head(h)
            return {"mu": mu, "sigma": sigma, "nu": nu, "hi": self.hi_head(h)[..., 0]}
        return self._f0(x, img, u, batch)

    def _f0(self, x: Tensor, img: Tensor | None, u: Tensor, batch: dict[str, Tensor]) -> dict:
        n = x.shape[1]
        b = x.shape[0]
        uu = u if self.config.usage else torch.zeros_like(u)
        mu_u, sg_u, nu_u = self.u_head(self.u_mlp(uu))
        delta, allowed = attention_mask(batch, self.config.history_recordings, False)
        h = self.backbone(x, delta, allowed)
        m = self._memory(x, batch)
        h = h if m is None else torch.cat([h, m], -1)
        mu_s, sg_s, nu_s = self.s_head(h)
        present_s = (~batch["drop_signal"])[:, None].expand(b, n)
        # image expert: GRU over image tokens; read the state after the last visible image
        mu_i, sg_i, nu_i = torch.zeros_like(mu_u), torch.ones_like(sg_u), torch.full_like(nu_u, 4.0)
        present_i = torch.zeros(b, n, dtype=torch.bool, device=x.device)
        if img is not None:
            valid, times = batch["img_valid"], batch["img_time"]
            state = img.new_zeros(b, self.i_cell.hidden_size)
            states, last_t = [], []
            last = torch.full((b,), float("-inf"), device=x.device)
            for j in range(img.shape[1]):
                prev = state if self.config.history_recordings == 0 else torch.zeros_like(state)
                new = self.i_cell(img[:, j], prev)
                state = torch.where(valid[:, j, None], new, state)
                last = torch.where(valid[:, j], times[:, j], last)
                states.append(state)
                last_t.append(last)
            st = torch.stack(states, 1)
            lt = torch.stack(last_t, 1)
            count = (times[:, None, :] < batch["tau"][:, :, None] - TOL).sum(-1)  # (B, n)
            idx = (count - 1).clamp_min(0)
            read = torch.gather(st, 1, idx[..., None].expand(-1, -1, st.shape[-1]))
            t_img = torch.gather(lt, 1, idx)
            present_i = (count > 0) & torch.isfinite(t_img) & ~batch["drop_images"][:, None]
            mu0, sg_i, nu_i = self.i_head(read)
            age = torch.where(present_i, batch["tau"] - t_img, torch.zeros_like(t_img))
            mu_i = (mu0 - age / self.horizon).clamp_min(1e-4)
        mus = torch.stack([mu_u, mu_s, mu_i], -1)
        sgs = torch.stack([sg_u, sg_s, sg_i], -1)
        nus = torch.stack([nu_u, nu_s, nu_i], -1)
        present = torch.stack([torch.ones_like(present_s), present_s, present_i], -1)
        logits = self.gate(torch.cat([sgs, uu], -1)).masked_fill(~present, float("-inf"))
        alpha = torch.softmax(logits, -1)
        prec = 1.0 / student_t_variance(sgs, nus)
        pi = (alpha * prec).sum(-1)
        mu = (alpha * prec * mus).sum(-1) / pi
        nu = (alpha * nus).sum(-1)
        sigma = torch.sqrt((nu - 2) / (nu * pi))
        return {
            "mu": mu,
            "sigma": sigma,
            "nu": nu,
            "hi": self.hi_head(h)[..., 0],
            "alpha": alpha,
            "experts": {"mu": mus, "sigma": sgs, "nu": nus, "present": present},
        }


def build_model(config: FitConfig, data, horizon_s: float) -> RULFuse:
    return RULFuse(
        config,
        horizon_s,
        data.signal_dim,
        data.channels,
        data.image_dim,
        data.image_tokens,
        len(data.kinds),
        data.has_hi,
    )
