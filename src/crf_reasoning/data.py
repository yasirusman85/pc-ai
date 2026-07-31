"""
data.py — Dataset loaders for CRF experiments
==============================================
Provides:
  1. SyntheticDataset   — character-level synthetic sequences (no downloads)
  2. TinyStoriesDataset — wraps HuggingFace TinyStories if available,
                          falls back to synthetic generation
  3. make_dataloader    — convenience wrapper
  4. CharTokenizer      — simple char-level tokenizer
"""

import os
import sys
import math
import random
import struct
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader

# ─── Char tokenizer ─────────────────────────────────────────────────────────


class CharTokenizer:
    """
    Character-level tokenizer.
    vocab = printable ASCII (32-126) + special tokens.
    """

    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3
    OFFSET = 4  # printable ASCII starts here

    def __init__(self):
        # printable ASCII: chr(32)..chr(126)  → 95 chars
        self.vocab_size = 95 + self.OFFSET  # 99
        self._char2id = {chr(i + 32): i + self.OFFSET for i in range(95)}
        self._id2char = {v: k for k, v in self._char2id.items()}

    def encode(self, text: str) -> List[int]:
        return [self._char2id.get(c, self.UNK_ID) for c in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(self._id2char.get(i, "?") for i in ids if i >= self.OFFSET)


# ─── Synthetic dataset ───────────────────────────────────────────────────────


def _make_story(seed: int, length: int = 200) -> str:
    """
    Generates a deterministic synthetic 'story' with:
      - simple noun/verb/adjective patterns → tests language modeling
      - repeating structure  → tests memory
      - number sequences     → tests arithmetic / pattern recognition
    """
    rng = random.Random(seed)
    nouns = [
        "cat",
        "dog",
        "bird",
        "child",
        "king",
        "woman",
        "man",
        "city",
        "tree",
        "sun",
    ]
    verbs = ["ran", "saw", "loved", "built", "found", "lost", "helped", "climbed"]
    adjs = ["big", "small", "red", "quiet", "brave", "old", "bright", "lost"]
    connectors = [" and ", ". Then ", ", so ", ". After that, ", " but "]

    sentences = []
    n = rng.randint(3, 8)
    for _ in range(n):
        kind = rng.random()
        if kind < 0.5:
            s = (
                f"The {rng.choice(adjs)} {rng.choice(nouns)} "
                f"{rng.choice(verbs)} the {rng.choice(nouns)}"
            )
        elif kind < 0.75:
            # count sequence
            start = rng.randint(1, 20)
            nums = " ".join(str(start + i) for i in range(rng.randint(3, 6)))
            s = f"The numbers are {nums}"
        else:
            s = (
                f"Once upon a time a {rng.choice(adjs)} {rng.choice(nouns)} "
                f"{rng.choice(verbs)}{rng.choice(connectors)}"
                f"the {rng.choice(nouns)} {rng.choice(verbs)}"
            )
        sentences.append(s)

    text = ". ".join(sentences) + "."
    return text[:length]


class SyntheticDataset(Dataset):
    """
    CPU-friendly synthetic text dataset.
    Generates n_stories deterministically, tokenizes at character level.
    """

    def __init__(
        self,
        n_stories: int = 2000,
        seq_len: int = 128,
        seed: int = 42,
        tokenizer: Optional[CharTokenizer] = None,
    ):
        self.seq_len = seq_len
        self.tokenizer = tokenizer or CharTokenizer()
        self.vocab_size = self.tokenizer.vocab_size

        rng = random.Random(seed)
        stories = []
        for i in range(n_stories):
            s = _make_story(rng.randint(0, 10**9), length=seq_len * 4)
            stories.append(s)

        # Tokenize and chunk into seq_len+1 windows
        self.chunks: List[torch.Tensor] = []
        for story in stories:
            ids = self.tokenizer.encode(story)
            for start in range(0, len(ids) - seq_len, seq_len // 2):
                chunk = ids[start : start + seq_len + 1]
                if len(chunk) == seq_len + 1:
                    self.chunks.append(torch.tensor(chunk, dtype=torch.long))

        # Shuffle deterministically
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(self.chunks), generator=g)
        self.chunks = [self.chunks[i] for i in perm.tolist()]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        x = chunk[:-1]  # seq_len
        y = chunk[1:]  # seq_len  (next-token targets)
        return x, y


# ─── TinyStories dataset (HuggingFace, with synthetic fallback) ──────────────


class TinyStoriesDataset(Dataset):
    """
    Wraps roneneldan/TinyStories from HuggingFace datasets.
    Falls back to SyntheticDataset if the package is unavailable or
    network is not accessible.
    """

    def __init__(
        self,
        split: str = "train",
        seq_len: int = 128,
        max_rows: int = 5000,
        tokenizer: Optional[CharTokenizer] = None,
        cache_dir: str = "/tmp/tiny_stories_cache",
        force_download: bool = False,
    ):
        self.seq_len = seq_len
        self.tokenizer = tokenizer or CharTokenizer()
        self.vocab_size = self.tokenizer.vocab_size
        self.force_download = force_download

        self._chunks = self._load(split, max_rows, cache_dir)

    def _load(self, split: str, max_rows: int, cache_dir: str):
        # Try HuggingFace
        try:
            from datasets import load_dataset

            load_kwargs = {
                "path": "roneneldan/TinyStories",
                "split": split,
                "streaming": not self.force_download,
                "trust_remote_code": False,
            }

            if self.force_download:
                load_kwargs["cache_dir"] = cache_dir
                load_kwargs["streaming"] = False

            ds = load_dataset(**load_kwargs)
            chunks = []

            for i, row in enumerate(ds):
                if i >= max_rows:
                    break
                text = row["text"]
                if not text or len(text) < 10:
                    continue

                ids = self.tokenizer.encode(text)
                for start in range(0, len(ids) - self.seq_len, self.seq_len // 2):
                    chunk = ids[start : start + self.seq_len + 1]
                    if len(chunk) == self.seq_len + 1:
                        chunks.append(torch.tensor(chunk, dtype=torch.long))
                if len(chunks) >= max_rows * 4:
                    break

            if chunks:
                print(f"[data] Loaded {len(chunks)} chunks from TinyStories/{split}")
                return chunks
        except Exception as e:
            print(f"[data] TinyStories unavailable ({e}), using synthetic data")

        # Fallback
        syn = SyntheticDataset(
            n_stories=max_rows,
            seq_len=self.seq_len,
            seed=0 if split == "train" else 99,
            tokenizer=self.tokenizer,
        )
        return syn.chunks

    def __len__(self):
        return len(self._chunks)

    def __getitem__(self, idx):
        chunk = self._chunks[idx]
        return chunk[:-1], chunk[1:]


# ─── Reasoning / arithmetic dataset ─────────────────────────────────────────


class ArithmeticDataset(Dataset):
    """
    Simple arithmetic word problems as a proxy for GSM8K.
    Format: "Q: What is <a> + <b>? A: <a+b>"
    Teaches multi-step reasoning at character level.
    """

    def __init__(
        self,
        n_samples: int = 2000,
        seq_len: int = 64,
        seed: int = 42,
        tokenizer: Optional[CharTokenizer] = None,
        difficulty: str = "mixed",  # 'easy', 'hard', 'mixed'
    ):
        self.tokenizer = tokenizer or CharTokenizer()
        self.vocab_size = self.tokenizer.vocab_size
        self.seq_len = seq_len

        rng = random.Random(seed)
        self.chunks = []

        for _ in range(n_samples):
            if difficulty == "easy" or (difficulty == "mixed" and rng.random() < 0.5):
                a, b = rng.randint(1, 99), rng.randint(1, 99)
                text = f"Q: What is {a} plus {b}? A: {a+b}."
            else:
                a, b, c = rng.randint(1, 50), rng.randint(1, 50), rng.randint(1, 20)
                text = (
                    f"Q: If you have {a} apples and get {b} more "
                    f"then give away {c}, how many? A: {a+b-c}."
                )

            ids = self.tokenizer.encode(text)
            if len(ids) < seq_len + 1:
                ids += [self.tokenizer.PAD_ID] * (seq_len + 1 - len(ids))
            ids = ids[: seq_len + 1]
            self.chunks.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return chunk[:-1], chunk[1:]


# ─── Multi-step chain-of-thought dataset (GSM8K proxy) ──────────────────────


class ChainOfThoughtDataset(Dataset):
    """
    Multi-step reasoning problems that require 2–6 intermediate steps.
    Harder than ArithmeticDataset: requires tracking running totals.

    Format:
      "Q: Alice has 12 apples. She gives 3 to Bob and buys 5 more.
       How many does she have? Step 1: 12-3=9. Step 2: 9+5=14. A: 14."

    Sequence length is stratified: short = 1-step, long = 4-step,
    allowing difficulty to be proxied by seq length for adaptive-compute
    validation (Fix 3/6).
    """

    def __init__(
        self,
        n_samples: int = 2000,
        seq_len: int = 128,
        seed: int = 42,
        tokenizer: Optional[CharTokenizer] = None,
        n_steps_range: tuple = (1, 5),
    ):
        self.tokenizer = tokenizer or CharTokenizer()
        self.vocab_size = self.tokenizer.vocab_size
        self.seq_len = seq_len
        self.chunks = []
        self.n_steps_per_example = []  # track difficulty

        rng = random.Random(seed)
        ops = ["+", "-"]

        for _ in range(n_samples):
            n_op = rng.randint(*n_steps_range)
            vals = [rng.randint(1, 30) for _ in range(n_op + 1)]
            oplist = [rng.choice(ops) for _ in range(n_op)]

            # Build question
            parts = [str(vals[0])]
            for op, v in zip(oplist, vals[1:]):
                parts.append(f"{'plus' if op=='+' else 'minus'} {v}")
            question = " ".join(parts) + "?"

            # Build step-by-step chain
            running = vals[0]
            steps = []
            for step_i, (op, v) in enumerate(zip(oplist, vals[1:])):
                new = running + v if op == "+" else running - v
                steps.append(f"Step {step_i+1}: {running}{op}{v}={new}")
                running = new

            text = f"Q: {question} {' '.join(steps)} A: {running}."
            ids = self.tokenizer.encode(text)

            if len(ids) < seq_len + 1:
                ids += [self.tokenizer.PAD_ID] * (seq_len + 1 - len(ids))
            ids = ids[: seq_len + 1]
            self.chunks.append(torch.tensor(ids, dtype=torch.long))
            self.n_steps_per_example.append(n_op)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return chunk[:-1], chunk[1:]

    def difficulty_split(self, easy_max_steps: int = 2):
        """Returns (easy_indices, hard_indices) based on step count."""
        easy = [
            i for i, n in enumerate(self.n_steps_per_example) if n <= easy_max_steps
        ]
        hard = [i for i, n in enumerate(self.n_steps_per_example) if n > easy_max_steps]
        return easy, hard


# ─── Code-completion dataset ─────────────────────────────────────────────────


class CodeCompletionDataset(Dataset):
    """
    Synthetic Python-like code snippets for code-completion benchmarking.
    Generates function stubs with arithmetic bodies that can be pattern-matched.
    No external downloads required.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        seq_len: int = 128,
        seed: int = 42,
        tokenizer: Optional[CharTokenizer] = None,
    ):
        self.tokenizer = tokenizer or CharTokenizer()
        self.vocab_size = self.tokenizer.vocab_size
        self.seq_len = seq_len
        self.chunks = []

        rng = random.Random(seed)
        fn_names = ["add", "sub", "mul", "total", "compute", "calc", "evaluate"]
        var_names = ["x", "y", "z", "a", "b", "n", "val"]

        for _ in range(n_samples):
            fname = rng.choice(fn_names)
            args = rng.sample(var_names, rng.randint(1, 3))
            ops = [rng.choice(["+", "-", "*"]) for _ in range(len(args) - 1)]

            # Build body
            if len(args) == 1:
                body = f"return {args[0]}"
            else:
                expr = args[0]
                for op, v in zip(ops, args[1:]):
                    expr = f"{expr} {op} {v}"
                body = f"return {expr}"

            # Add occasional docstring
            doc = ""
            if rng.random() < 0.4:
                doc = f'    """Compute result."""\n'

            sig = f"def {fname}({', '.join(args)}):\n"
            code = sig + doc + f"    {body}\n"

            ids = self.tokenizer.encode(code)
            if len(ids) < seq_len + 1:
                ids += [self.tokenizer.PAD_ID] * (seq_len + 1 - len(ids))
            ids = ids[: seq_len + 1]
            self.chunks.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return chunk[:-1], chunk[1:]


class ARCProxyDataset(Dataset):
    """
    Synthetic ARC-like pattern completion at character level.
    Each example: a 2D grid of digits, and the task is to predict the
    next row according to a simple rule.

    Rules (randomly chosen per example):
      - increment: each cell is previous + 1 mod 10
      - copy: next row = first row
      - flip: reverse each row
    """

    def __init__(
        self,
        n_samples: int = 1000,
        grid_size: int = 4,
        seq_len: int = 64,
        seed: int = 7,
        tokenizer: Optional[CharTokenizer] = None,
    ):
        self.tokenizer = tokenizer or CharTokenizer()
        self.vocab_size = self.tokenizer.vocab_size
        self.seq_len = seq_len

        rng = random.Random(seed)
        self.chunks = []

        for _ in range(n_samples):
            rule = rng.choice(["increment", "copy", "flip"])
            rows = [
                [rng.randint(0, 9) for _ in range(grid_size)] for _ in range(grid_size)
            ]

            if rule == "increment":
                target = [[(v + 1) % 10 for v in rows[-1]]]
            elif rule == "copy":
                target = [rows[0][:]]
            else:  # flip
                target = [rows[-1][::-1]]

            # Encode as text
            def row2str(r):
                return " ".join(str(v) for v in r)

            context = " | ".join(row2str(r) for r in rows)
            answer = row2str(target[0])
            text = f"Input: {context} Output: {answer}"

            ids = self.tokenizer.encode(text)
            if len(ids) < seq_len + 1:
                ids += [self.tokenizer.PAD_ID] * (seq_len + 1 - len(ids))
            ids = ids[: seq_len + 1]
            self.chunks.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return chunk[:-1], chunk[1:]


# ─── Convenience factory ─────────────────────────────────────────────────────


def make_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: Optional[int] = None,
    drop_last: bool = False,
) -> DataLoader:
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        generator=generator,
        drop_last=drop_last,
    )


def get_datasets(
    name: str = "synthetic",
    seq_len: int = 128,
    max_train: int = 3000,
    max_val: int = 500,
    tokenizer: Optional[CharTokenizer] = None,
    use_real: bool = False,
) -> Tuple[Dataset, Dataset]:
    """
    Returns (train_dataset, val_dataset).
    name: 'synthetic' | 'tinystories' | 'arithmetic' | 'arc' |
          'chain_of_thought' | 'code' | 'humaneval' | 'gsm8k' | 'shakespeare'
    use_real: If True, uses real datasets when available (requires downloads)
    """
    tok = tokenizer or CharTokenizer()

    # Use real datasets if requested and available
    if use_real:
        try:
            from real_datasets import RealDatasetFactory

            if name in ["humaneval", "gsm8k", "arc"]:
                train_ds = RealDatasetFactory.create_dataset(
                    name,
                    split="train",
                    seq_len=seq_len,
                    max_samples=max_train,
                    tokenizer=tok,
                )
                val_ds = RealDatasetFactory.create_dataset(
                    name,
                    split="test",
                    seq_len=seq_len,
                    max_samples=max_val,
                    tokenizer=tok,
                )
                return train_ds, val_ds
        except ImportError:
            print("[data] Real datasets module not available, using synthetic")

    if name == "tinystories":
        train_ds = TinyStoriesDataset("train", seq_len, max_train, tok)
        val_ds = TinyStoriesDataset("validation", seq_len, max_val, tok)
    elif name == "arithmetic":
        train_ds = ArithmeticDataset(max_train, seq_len, seed=0, tokenizer=tok)
        val_ds = ArithmeticDataset(max_val, seq_len, seed=99, tokenizer=tok)
    elif name == "arc":
        train_ds = ARCProxyDataset(max_train, seq_len=seq_len, seed=0, tokenizer=tok)
        val_ds = ARCProxyDataset(max_val, seq_len=seq_len, seed=99, tokenizer=tok)
    elif name == "chain_of_thought":
        train_ds = ChainOfThoughtDataset(max_train, seq_len, seed=0, tokenizer=tok)
        val_ds = ChainOfThoughtDataset(max_val, seq_len, seed=99, tokenizer=tok)
    elif name == "code":
        train_ds = CodeCompletionDataset(max_train, seq_len, seed=0, tokenizer=tok)
        val_ds = CodeCompletionDataset(max_val, seq_len, seed=99, tokenizer=tok)
    elif name == "humaneval":
        # Fallback to code completion
        train_ds = CodeCompletionDataset(max_train, seq_len, seed=0, tokenizer=tok)
        val_ds = CodeCompletionDataset(max_val, seq_len, seed=99, tokenizer=tok)
    elif name == "gsm8k":
        # Fallback to arithmetic with chain-of-thought
        train_ds = ChainOfThoughtDataset(max_train, seq_len, seed=0, tokenizer=tok)
        val_ds = ChainOfThoughtDataset(max_val, seq_len, seed=99, tokenizer=tok)
    elif name == "shakespeare":
        # Real language modeling data (Tiny Shakespeare, 1.1M chars)
        try:
            from shakespeare_dataset import get_shakespeare_datasets
        except ImportError:
            sys.path.insert(
                0,
                str(
                    Path(__file__).parent.parent.parent
                    / "experiments"
                    / "real_experiments"
                ),
            )
            from shakespeare_dataset import get_shakespeare_datasets
        train_ds, val_ds = get_shakespeare_datasets(
            seq_len=seq_len, train_ratio=0.9, tokenizer=tok
        )
    elif name == "synthetic":
        train_ds = SyntheticDataset(max_train, seq_len, seed=0, tokenizer=tok)
        val_ds = SyntheticDataset(max_val, seq_len, seed=99, tokenizer=tok)
    else:
        raise ValueError(
            f"Unknown dataset name {name!r}. Valid: synthetic, tinystories, "
            f"arithmetic, arc, chain_of_thought, code, humaneval, gsm8k, shakespeare"
        )

    return train_ds, val_ds


# ─── Quick self-test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    tok = CharTokenizer()
    print(f"Vocab size: {tok.vocab_size}")
    print(f"Encode 'hello': {tok.encode('hello')}")
    print(f"Decode back:    {tok.decode(tok.encode('hello'))}")

    for name in ["synthetic", "arithmetic", "arc"]:
        tr, va = get_datasets(name, seq_len=64, max_train=200, max_val=50)
        x, y = tr[0]
        print(f"[{name}] train={len(tr)} val={len(va)} x.shape={x.shape}")
        dl = make_dataloader(tr, batch_size=8)
        xb, yb = next(iter(dl))
        print(f"  batch: x={xb.shape} y={yb.shape}")
