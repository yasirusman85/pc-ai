import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random


class CellProgram(nn.Module):
    def __init__(self, d_model, d_hidden=128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_model)
        )
        self.msg = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.energy_gate = nn.Linear(d_model, 1)

    def forward(self, state, message):
        x = torch.cat([state, message], dim=-1)
        new_state = self.norm(state + self.gate(x))
        return (
            new_state,
            self.msg(new_state),
            torch.sigmoid(self.energy_gate(new_state)),
        )

    def copy_mutated(self, noise=0.1):
        d_model = self.gate[0].in_features // 2
        clone = CellProgram(d_model, self.gate[0].out_features)
        with torch.no_grad():
            for p, cp in zip(self.parameters(), clone.parameters()):
                cp.copy_(p + torch.randn_like(p) * noise * (p.std() + 1e-6))
        return clone


class CognitiveCell(nn.Module):
    def __init__(self, state, pos, program, is_anchor=False):
        super().__init__()
        self.is_anchor = is_anchor
        self.register_buffer("state", state)
        self.register_buffer("pos", pos)
        self.register_buffer("energy", torch.tensor(1.0))
        self.program = program
        self.age = 0

    def step(self, message):
        self.age += 1
        new_state, out_msg, energy_delta = self.program(self.state, message)
        self.state.data = new_state
        self.energy.data = (self.energy * 0.95 + energy_delta.squeeze(-1) * 0.05).clamp(
            0, 5
        )
        return out_msg


class SparseFabric(nn.Module):
    def __init__(self, d_model, k=4):
        super().__init__()
        self.k = k
        self.route = nn.Linear(d_model * 2, 1)

    def forward(self, states, positions):
        B = states.size(0)
        if B <= 1:
            return states.clone() if B == 1 else states

        sim = F.normalize(states, dim=-1) @ F.normalize(states, dim=-1).T
        pos_dists = torch.cdist(positions, positions)
        sim = sim - 0.05 * pos_dists
        sim.fill_diagonal_(-1e9)

        k = min(self.k, B - 1)
        vals, idxs = sim.topk(k, dim=-1)
        messages = torch.zeros_like(states)
        for i in range(B):
            ws = torch.zeros(B, device=states.device)
            for j, v in zip(idxs[i], vals[i]):
                w = torch.sigmoid(self.route(torch.cat([states[i], states[j]])))
                ws[j] = w
            if ws.sum() > 0:
                messages[i] = (ws.unsqueeze(-1) * states).sum(0) / ws.sum()
            else:
                messages[i] = states[i]
        return messages


class CellularReasoningFabric(nn.Module):
    def __init__(self, d_model=256, n_init=64, max_cells=1024, k_neighbors=4):
        super().__init__()
        self.d_model = d_model
        self.max_cells = max_cells
        self.n_init = n_init
        self.fabric = SparseFabric(d_model, k_neighbors)
        self.input_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, token_states, n_steps=8):
        B, T, C = token_states.shape
        device = token_states.device
        in_states = self.input_proj(token_states[0])

        cells = nn.ModuleList()
        n_grid = max(4, int(math.sqrt(self.n_init)))
        n_create = max(self.n_init, T + 8)

        for i in range(n_create):
            if i < T:
                state = in_states[i]
                pos = torch.tensor(
                    [float(i % n_grid), float(i // n_grid)], device=device
                )
                anchor = True
            else:
                src = in_states[i % T]
                state = src + torch.randn(C, device=device) * 0.02
                ti = i % T
                pos = torch.tensor(
                    [
                        ti % n_grid + random.uniform(-0.3, 0.3),
                        ti // n_grid + random.uniform(-0.3, 0.3),
                    ],
                    device=device,
                )
                anchor = False
            prog = CellProgram(C)
            cells.append(CognitiveCell(state.detach(), pos, prog, is_anchor=anchor))

        for step in range(n_steps):
            if len(cells) < 2:
                break
            states = torch.stack([c.state for c in cells])
            positions = torch.stack([c.pos for c in cells])
            messages = self.fabric(states, positions)
            for i, c in enumerate(cells):
                c.step(messages[i])

            if step % 3 == 0:
                for i in range(len(cells)):
                    c = cells[i]
                    if (
                        not c.is_anchor
                        and c.energy.item() > 1.5
                        and len(cells) < self.max_cells
                    ):
                        cp = c.pos + (torch.randn(2, device=device) * 0.5).clamp(
                            -0.5, 0.5
                        )
                        cs = c.state + torch.randn(C, device=device) * 0.01
                        child = CognitiveCell(
                            cs.detach(), cp, c.program.copy_mutated(0.05)
                        )
                        child.energy.data = c.energy * 0.4
                        c.energy.data = c.energy * 0.4
                        cells.append(child)

                dead = sorted(
                    [
                        i
                        for i, c in enumerate(cells)
                        if not c.is_anchor and c.energy.item() < 0.01
                    ],
                    reverse=True,
                )
                for i in dead:
                    cells.pop(i)

            if step % 5 == 0 and step > 0 and len(cells) > 2:
                for _ in range(min(len(cells) // 4, 3)):
                    a = random.randint(0, len(cells) - 1)
                    b = random.randint(0, len(cells) - 1)
                    if a != b and not (cells[a].is_anchor and cells[b].is_anchor):
                        ca, cb = cells[a], cells[b]
                        sim = F.cosine_similarity(
                            ca.state.unsqueeze(0), cb.state.unsqueeze(0)
                        ).item()
                        if sim > 0.95:
                            ca.state.data = (ca.state + cb.state) / 2
                            ca.energy.data = ca.energy + cb.energy
                            ca.pos.data = (ca.pos + cb.pos) / 2
                            cells.pop(max(a, b))

        anchor_states = torch.stack(
            [c.state for i, c in enumerate(cells) if c.is_anchor]
        )
        if anchor_states.size(0) < T:
            pad = torch.zeros(T - anchor_states.size(0), C, device=device)
            anchor_states = torch.cat([anchor_states, pad])
        return self.output_proj(anchor_states.unsqueeze(0))


class CRFTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        n_init_cells=64,
        max_cells=1024,
        n_crf_steps=8,
        k_neighbors=4,
        max_seq_len=2048,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.crf = CellularReasoningFabric(
            d_model, n_init_cells, max_cells, k_neighbors
        )
        self.n_crf_steps = n_crf_steps
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, x, targets=None):
        B, T = x.shape
        tok = self.token_embed(x) + self.pos_enc[:, :T]
        outs = []
        for b in range(B):
            out = self.crf(tok[b : b + 1], self.n_crf_steps)
            outs.append(out)
        out = torch.cat(outs, dim=0)
        logits = self.lm_head(self.ln(out))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


if __name__ == "__main__":
    vocab_size = 5000
    model = CRFTransformer(
        vocab_size=vocab_size,
        d_model=128,
        n_init_cells=32,
        max_cells=256,
        n_crf_steps=6,
        k_neighbors=3,
    )
    print(
        f"CRF params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    x = torch.randint(0, vocab_size, (2, 16))
    logits, loss = model(x, targets=x)
    print(f"Forward OK: logits {logits.shape}, loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        gen = x[:1, :4]
        for _ in range(10):
            logits, _ = model(gen)
            probs = F.softmax(logits[:, -1] / 0.8, dim=-1)
            gen = torch.cat((gen, torch.multinomial(probs, 1)), dim=1)
        print(f"Generate OK: {gen.shape}")
