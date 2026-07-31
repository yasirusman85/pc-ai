"""
CRF Simulator: Cellular Reasoning Fabric v2
CPU-based, 100-1000 cells, bytecode DSL, dynamic graph, cell lifecycle, voting consensus.
"""

import torch
import torch.nn.functional as F
import math
import random
from collections import Counter

DEVICE = torch.device('cpu')
INT_MAX = 2**15 - 1

# ─── DSL ──────────────────────────────────────────────────────────────────

OPS = {
    0: ('NOP', 0),    # no-op
    1: ('MOV', 2),    # MOV dst src   (copy value)
    2: ('ADD', 3),    # ADD dst a b
    3: ('SUB', 3),    # SUB dst a b
    4: ('MUL', 3),    # MUL dst a b
    5: ('DIV', 3),    # DIV dst a b
    6: ('LT',  3),    # LT  dst a b   (1 if a < b else 0)
    7: ('GT',  3),    # GT  dst a b
    8: ('EQ',  3),    # EQ  dst a b
    9: ('SEND', 2),   # SEND addr val
   10: ('RECV', 2),   # RECV dst addr
   11: ('MEMW', 2),   # MEMW key val
   12: ('MEMR', 2),   # MEMR dst key
   13: ('VOTE', 1),   # VOTE val     (register output vote)
   14: ('SPLT', 0),   # SPLT        (request split)
}

class Interpreter:
    def __init__(self, registers, memory, messages_in):
        self.r = registers
        self.mem = memory
        self.msg = messages_in
        self.vote = None
        self.outbox = {}
        self.split_request = False

    def run(self, prog, steps=8):
        ip = 0
        plen = len(prog)
        for _ in range(steps):
            if ip >= plen:
                break
            opcode = int(prog[ip])
            if opcode == 0:   # NOP
                ip += 1
            elif opcode == 1 and ip + 2 < plen: # MOV
                dst, src = int(prog[ip+1]), int(prog[ip+2])
                self._wr(dst, self._val(src))
                ip += 3
            elif opcode == 2 and ip + 3 < plen: # ADD
                d, a, b = int(prog[ip+1]), int(prog[ip+2]), int(prog[ip+3])
                self._wr(d, self._val(a) + self._val(b))
                ip += 4
            elif opcode == 3 and ip + 3 < plen: # SUB
                d, a, b = int(prog[ip+1]), int(prog[ip+2]), int(prog[ip+3])
                self._wr(d, self._val(a) - self._val(b))
                ip += 4
            elif opcode == 4 and ip + 3 < plen: # MUL
                d, a, b = int(prog[ip+1]), int(prog[ip+2]), int(prog[ip+3])
                self._wr(d, self._val(a) * self._val(b))
                ip += 4
            elif opcode == 5 and ip + 3 < plen: # DIV
                d, a, b = int(prog[ip+1]), int(prog[ip+2]), int(prog[ip+3])
                bv = self._val(b)
                self._wr(d, self._val(a) / bv if bv != 0 else 0)
                ip += 4
            elif opcode == 6 and ip + 3 < plen: # LT
                d, a, b = int(prog[ip+1]), int(prog[ip+2]), int(prog[ip+3])
                self._wr(d, 1.0 if self._val(a) < self._val(b) else 0.0)
                ip += 4
            elif opcode == 7 and ip + 3 < plen: # GT
                d, a, b = int(prog[ip+1]), int(prog[ip+2]), int(prog[ip+3])
                self._wr(d, 1.0 if self._val(a) > self._val(b) else 0.0)
                ip += 4
            elif opcode == 8 and ip + 3 < plen: # EQ
                d, a, b = int(prog[ip+1]), int(prog[ip+2]), int(prog[ip+3])
                self._wr(d, 1.0 if abs(self._val(a) - self._val(b)) < 1e-6 else 0.0)
                ip += 4
            elif opcode == 9 and ip + 2 < plen: # SEND
                addr, val = int(prog[ip+1]), int(prog[ip+2])
                self.outbox[int(addr)] = self._val(val)
                ip += 3
            elif opcode == 10 and ip + 2 < plen: # RECV
                dst, addr = int(prog[ip+1]), int(prog[ip+2])
                self._wr(dst, self.msg.get(int(addr), 0.0))
                ip += 3
            elif opcode == 11 and ip + 2 < plen: # MEMW
                k, v = int(prog[ip+1]), int(prog[ip+2])
                self.mem[int(k)] = self._val(v)
                ip += 3
            elif opcode == 12 and ip + 2 < plen: # MEMR
                dst, k = int(prog[ip+1]), int(prog[ip+2])
                self._wr(dst, self.mem.get(int(k), 0.0))
                ip += 3
            elif opcode == 13 and ip + 1 < plen: # VOTE
                self.vote = self._val(int(prog[ip+1]))
                ip += 2
            elif opcode == 14: # SPLT
                self.split_request = True
                ip += 1
            else:
                ip += 1
        return self.r, self.vote, self.outbox, self.split_request

    def _val(self, src):
        if 0 <= src < len(self.r):
            return self.r[src]
        return float(src)

    def _wr(self, dst, val):
        if 0 <= dst < len(self.r):
            self.r[dst] = val


# ─── CELL ─────────────────────────────────────────────────────────────────

class Cell:
    def __init__(self, cid, program, state_dim=64, mem_size=16):
        self.cid = cid
        self.program = program  # list of ints (bytecode)
        self.state = [random.uniform(-1, 1) for _ in range(state_dim)]
        self.memory = {}
        self.registers = [0.0] * 32
        self.energy = 1.0
        self.confidence = 0.0
        self.age = 0
        self.last_vote = None
        self.inbox = {}
        self.state_dim = state_dim

    def step(self, neighbors):
        self.age += 1
        interp = Interpreter(self.registers[:], self.memory, self.inbox)
        regs, vote, outbox, split_req = interp.run(self.program)
        self.registers = regs
        self.last_vote = vote
        self.inbox = {}

        outvec = sum(outbox.values()) / max(len(outbox), 1)
        self.state = [0.9 * s + 0.1 * outvec for s in self.state]

        if vote is not None:
            self.confidence = 1 - 1 / (1 + abs(vote) + 1e-6)
            self.energy += 0.05 * self.confidence
        self.energy *= 0.97

        return outbox, split_req

    def mutate(self, rate=0.2):
        new_prog = self.program[:]
        for i in range(len(new_prog)):
            if random.random() < rate:
                new_prog[i] = random.randint(0, len(OPS) - 1)
        if random.random() < rate:
            if random.random() < 0.5 and len(new_prog) < 32:
                new_prog.append(random.randint(0, len(OPS) - 1))
            elif len(new_prog) > 2:
                new_prog.pop(random.randrange(len(new_prog)))
        return new_prog

    def merge_state_with(self, other):
        self.state = [(a + b) / 2 for a, b in zip(self.state, other.state)]
        self.energy = min(self.energy + other.energy, 5.0)
        self.confidence = max(self.confidence, other.confidence)


# ─── FABRIC (communication graph) ────────────────────────────────────────

class SparseGraph:
    def __init__(self, k=4):
        self.k = k

    def build(self, cells):
        n = len(cells)
        if n == 0:
            return {}
        states = torch.tensor([c.state for c in cells])
        sim = F.normalize(states) @ F.normalize(states).T
        sim.fill_diagonal_(-1e9)

        graph = {c.cid: [] for c in cells}
        k = min(self.k, n - 1)
        vals, idxs = sim.topk(k, dim=-1)
        for i, cell in enumerate(cells):
            for j in idxs[i]:
                graph[cell.cid].append(cells[j.item()].cid)
        return graph

    def route_messages(self, cells, graph, outboxes):
        for cell in cells:
            msg_sum = 0.0
            count = 0
            for nid in graph.get(cell.cid, []):
                if nid in outboxes and cell.cid in outboxes[nid]:
                    msg_sum += outboxes[nid][cell.cid]
                    count += 1
            if count > 0:
                cell.inbox[0] = msg_sum / count
        for cell in cells:
            if random.random() < 0.1:
                others = [c for c in cells if c.cid != cell.cid]
                if others:
                    src = random.choice(others)
                    cell.inbox[1] = src.registers[0]


# ─── LIFECYCLE SCHEDULER ─────────────────────────────────────────────────

class Scheduler:
    def __init__(self, max_cells=1000):
        self.max_cells = max_cells
        self.step = 0

    def update(self, cells):
        self.step += 1
        changes = []

        for cell in cells[:]:
            if cell.age % 5 == 0:
                if (cell.energy > 1.8 and len(cells) < self.max_cells
                        and cell.age > 3 and random.random() < 0.3):
                    child = Cell(len(cells), cell.mutate(0.3), cell.state_dim)
                    child.energy = cell.energy * 0.4
                    cell.energy *= 0.4
                    child.state = [s + random.uniform(-0.1, 0.1) for s in cell.state]
                    cells.append(child)
                    changes.append(f'split c{cell.cid} -> c{child.cid}')

        for cell in cells[:]:
            if cell.energy < 0.01 and cell.age > 5:
                cells.remove(cell)
                changes.append(f'die c{cell.cid}')

        if len(cells) > 2 and self.step % 5 == 0:
            pairs = []
            for i in range(len(cells)):
                for j in range(i + 1, len(cells)):
                    sim = sum(abs(a - b) for a, b in zip(cells[i].state, cells[j].state))
                    pairs.append((sim, i, j))
            pairs.sort()
            merged = set()
            for sim, i, j in pairs[:max(1, len(cells) // 20)]:
                if i not in merged and j not in merged and sim < 1.0:
                    cells[min(i, j)].merge_state_with(cells[max(i, j)])
                    cells.pop(max(i, j))
                    merged.add(min(i, j))
                    merged.add(max(i, j))
                    changes.append(f'merge c{i} c{j}')

        return changes


# ─── CONSENSUS ────────────────────────────────────────────────────────────

class Consensus:
    def aggregate(self, cells, n_classes=10):
        votes = {}
        for cell in cells:
            if cell.last_vote is not None:
                key = round(cell.last_vote)
                weight = cell.confidence * cell.energy
                votes[key] = votes.get(key, 0) + weight
        if not votes:
            return 0, 0.0
        best = max(votes, key=votes.get)
        return best, votes[best]


# ─── SIMULATOR ────────────────────────────────────────────────────────────

class CRFSimulator:
    def __init__(self, n_init=200, max_cells=1000, state_dim=64, k_neighbors=4):
        self.n_init = n_init
        self.max_cells = max_cells
        self.state_dim = state_dim
        self.graph = SparseGraph(k_neighbors)
        self.scheduler = Scheduler(max_cells)
        self.consensus = Consensus()

    def seed(self, input_vec, n_cells=None):
        n = n_cells or self.n_init
        self.input_vec = input_vec
        d = len(input_vec) if hasattr(input_vec, '__len__') else self.state_dim
        self.cells = []
        for i in range(n):
            prog = [random.randint(0, len(OPS) - 1) for _ in range(random.randint(4, 16))]
            cell = Cell(i, prog, self.state_dim)
            if i < d:
                cell.registers[0] = input_vec[i] if hasattr(input_vec, '__getitem__') else input_vec
            self.cells.append(cell)

    def step(self):
        g = self.graph.build(self.cells)
        outboxes = {}
        split_reqs = {}
        for cell in self.cells:
            ob, sr = cell.step(g.get(cell.cid, []))
            outboxes[cell.cid] = ob
            split_reqs[cell.cid] = sr

        self.graph.route_messages(self.cells, g, outboxes)
        changes = self.scheduler.update(self.cells)
        return changes

    def run(self, steps=20):
        for s in range(steps):
            self.step()
        vote, conf = self.consensus.aggregate(self.cells)
        return vote, conf, len(self.cells)

    def state_matrix(self):
        return torch.tensor([c.state for c in self.cells])


# ─── BENCHMARK HELPERS ───────────────────────────────────────────────────

def encode_arc_grid(grid, max_dim=64):
    flat = []
    for row in grid:
        flat.extend([float(x) / 10.0 for x in row])
    while len(flat) < max_dim:
        flat.append(0.0)
    return flat[:max_dim]

def eval_arc(sim, grid, n_steps=20):
    inp = encode_arc_grid(grid)
    sim.seed(inp, n_cells=100)
    vote, conf, n = sim.run(n_steps)
    return vote, conf, n


# ─── DEMO ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    random.seed(42)
    sim = CRFSimulator(n_init=200, max_cells=1000, state_dim=64, k_neighbors=4)
    sim.seed([0.5, -0.3, 0.8, 0.1, -0.7], n_cells=200)

    for step in range(30):
        changes = sim.step()
        if step % 10 == 0:
            vote, conf = sim.consensus.aggregate(sim.cells)
            print(f'step {step:3d} | cells {len(sim.cells):4d} | vote {vote:4d} conf {conf:.3f}')

    vote, conf, n = sim.consensus.aggregate(sim.cells), 0, len(sim.cells)
    print(f'\nfinal | cells {n} | consensus vote {vote[0]} (conf {vote[1]:.3f})')

    # ARC-like test
    grid = [[1, 2], [3, 4]]
    out = eval_arc(sim, grid)
    print(f'ARC test | vote {out[0]} conf {out[1]:.3f} cells {out[2]}')
