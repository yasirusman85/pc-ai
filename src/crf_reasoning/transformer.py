import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=8192):
        super().__init__()
        self.d_model = d_model
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, offset=0):
        seq_len = x.size(1)
        t = torch.arange(offset, offset + seq_len, device=x.device).type_as(
            self.inv_freq
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb

    @staticmethod
    def apply_rotary(x, cos, sin):
        d = x.size(-1)
        x1, x2 = x[..., : d // 2], x[..., d // 2 :]
        rx1 = x1 * cos - x2 * sin
        rx2 = x1 * sin + x2 * cos
        return torch.cat((rx1, rx2), dim=-1)


def precompute_rotary_cache(head_dim, seq_len, device):
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, head_dim, 2).float().to(device) / head_dim)
    )
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, use_rotary=False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_rotary = use_rotary

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, cos=None, sin=None):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_rotary and cos is not None and sin is not None:
            cos = cos[:T].unsqueeze(0).unsqueeze(0)
            sin = sin[:T].unsqueeze(0).unsqueeze(0)
            q = RotaryPositionalEncoding.apply_rotary(q, cos, sin)
            k = RotaryPositionalEncoding.apply_rotary(k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        y = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(y)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, activation="swiglu", dropout=0.1):
        super().__init__()
        self.activation = activation
        if activation == "swiglu":
            self.w1 = nn.Linear(d_model, d_ff * 2, bias=False)
            self.w2 = nn.Linear(d_ff, d_model, bias=False)
        else:
            self.fc1 = nn.Linear(d_model, d_ff, bias=False)
            self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.activation == "swiglu":
            x1, x2 = self.w1(x).chunk(2, dim=-1)
            x = F.silu(x1) * x2
            return self.w2(self.dropout(x))
        else:
            return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model, n_heads, d_ff, dropout=0.1, activation="swiglu", use_rotary=False
    ):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, use_rotary)
        self.ff = FeedForward(d_model, d_ff, activation, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, cos=None, sin=None):
        x = x + self.dropout(self.attn(self.norm1(x), mask, cos, sin))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=768,
        n_heads=12,
        n_layers=12,
        d_ff=None,
        max_seq_len=2048,
        dropout=0.1,
        activation="swiglu",
        use_rotary=False,
        tie_weights=False,
    ):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.use_rotary = use_rotary

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = (
            SinusoidalPositionalEncoding(d_model, max_seq_len)
            if not use_rotary
            else None
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, n_heads, d_ff, dropout, activation, use_rotary
                )
                for _ in range(n_layers)
            ]
        )

        self.norm = LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.token_embed.weight

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(
                1, 1, max_seq_len, max_seq_len
            ),
        )
        self.cos, self.sin = None, None

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        B, T = x.shape
        assert (
            T <= self.max_seq_len
        ), f"Sequence length {T} exceeds max {self.max_seq_len}"

        x = self.token_embed(x) * math.sqrt(self.d_model)

        if self.use_rotary:
            if self.cos is None or self.cos.size(0) < T:
                head_dim = self.d_model // self.blocks[0].attn.n_heads
                self.cos, self.sin = precompute_rotary_cache(
                    head_dim, self.max_seq_len, x.device
                )
            cos, sin = self.cos, self.sin
        else:
            x = self.pos_enc(x)
            cos, sin = None, None

        mask = self.causal_mask[:, :, :T, :T]
        if x.device != self.causal_mask.device:
            mask = mask.to(x.device)

        for block in self.blocks:
            x = block(x, mask, cos, sin)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, x, max_new_tokens=100, temperature=1.0, top_k=None, top_p=None):
        self.eval()
        for _ in range(max_new_tokens):
            x_cond = x[:, -self.max_seq_len :]
            logits, _ = self.forward(x_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < vals[:, -1:]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
                cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                mask = cum_probs - sorted_logits.softmax(dim=-1) > top_p
                logits.scatter_(
                    1,
                    sorted_idx,
                    torch.where(mask, float("-inf"), logits.gather(1, sorted_idx)),
                )

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token), dim=1)

        return x

    @torch.no_grad()
    def generate_beam(self, x, max_new_tokens=100, beam_width=4, temperature=1.0):
        self.eval()
        beams = [(x, 0.0)]

        for _ in range(max_new_tokens):
            candidates = []
            for seq, score in beams:
                x_cond = seq[:, -self.max_seq_len :]
                logits, _ = self.forward(x_cond)
                logits = logits[:, -1, :] / temperature
                probs = F.log_softmax(logits, dim=-1)
                vals, idxs = probs.topk(beam_width, dim=-1)
                for v, i in zip(vals[0], idxs[0]):
                    token = i.reshape(1, 1)
                    new_seq = torch.cat((seq, token), dim=1)
                    candidates.append((new_seq, score + v.item()))

            beams = sorted(candidates, key=lambda p: p[1], reverse=True)[:beam_width]

        return beams[0][0]


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    vocab_size = 50257
    model = GPT(
        vocab_size=vocab_size,
        d_model=768,
        n_heads=12,
        n_layers=12,
        max_seq_len=2048,
        use_rotary=False,
    )
    print(f"Transformer params: {count_params(model):,}")

    x = torch.randint(0, vocab_size, (2, 128))
    logits, loss = model(x, targets=x)
    print(f"Forward OK: logits {logits.shape}, loss {loss.item():.4f}")

    gen = model.generate(x[:, :10], max_new_tokens=20, temperature=0.8, top_k=50)
    print(f"Generate OK: {gen.shape}")

    gen_beam = model.generate_beam(x[:1, :10], max_new_tokens=20, beam_width=3)
    print(f"Beam search OK: {gen_beam.shape}")

    rotary_model = GPT(
        vocab_size=vocab_size,
        d_model=768,
        n_heads=12,
        n_layers=12,
        max_seq_len=2048,
        use_rotary=True,
    )
    print(f"RoFormer params: {count_params(rotary_model):,}")
    logits2, loss2 = rotary_model(x, targets=x)
    print(f"RoFormer forward OK: loss {loss2.item():.4f}")
