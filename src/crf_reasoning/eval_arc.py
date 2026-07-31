"""
eval_arc.py — ARC-AGI grid evaluation (Phase 5)
===============================================
Parses generated grid text, computes cell-level accuracy, and tracks
adaptive compute (active cell count) by puzzle difficulty.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from data import CharTokenizer


# ─── Grid parsing ─────────────────────────────────────────────────────────────

def parse_grid_text(text: str) -> Optional[List[List[int]]]:
    """
    Parse a flattened grid from model output.
    Accepts: '0 1 2 0 1' or '0:2 1:1 2:1' (RLE) formats.
    """
    text = text.strip()
    if not text:
        return None

    # RLE format: "0:3 1:2"
    if ':' in text and re.search(r'\d:\d', text):
        cells = []
        for part in text.split():
            if ':' not in part:
                continue
            val, count = part.split(':')
            cells.extend([int(val)] * int(count))
        if cells:
            side = int(len(cells) ** 0.5)
            if side * side == len(cells):
                return [cells[i * side:(i + 1) * side] for i in range(side)]
            return [cells]

    # Flat format: space-separated integers
    nums = [int(x) for x in re.findall(r'\d+', text)]
    if not nums:
        return None
    side = int(len(nums) ** 0.5)
    if side * side == len(nums):
        return [nums[i * side:(i + 1) * side] for i in range(side)]
    return [nums]


def grid_cell_accuracy(
    pred: Optional[List[List[int]]],
    gold: Optional[List[List[int]]],
) -> float:
    """Fraction of matching cells (requires same shape)."""
    if pred is None or gold is None:
        return 0.0
    if len(pred) != len(gold) or (pred and len(pred[0]) != len(gold[0])):
        return 0.0
    total = sum(len(row) for row in gold)
    if total == 0:
        return 0.0
    correct = sum(
        int(pred[r][c] == gold[r][c])
        for r in range(len(gold))
        for c in range(len(gold[r]))
    )
    return correct / total


def grid_exact_match(
    pred: Optional[List[List[int]]],
    gold: Optional[List[List[int]]],
) -> bool:
    """True if predicted grid exactly matches gold."""
    if pred is None or gold is None:
        return False
    return pred == gold


# ─── Eval dataset ─────────────────────────────────────────────────────────────

class ARCEvalDataset(Dataset):
    """ARC puzzles with input/output grids and encoded prompts."""

    def __init__(
        self,
        split: str = 'train',
        max_samples: Optional[int] = None,
        tokenizer: Optional[CharTokenizer] = None,
        cache_dir: str = '/tmp/arc_cache',
        grid_encoding: str = 'flatten',
    ):
        self.tokenizer = tokenizer or CharTokenizer()
        self.grid_encoding = grid_encoding
        self.items: List[Dict] = self._load(split, cache_dir, max_samples)

    @staticmethod
    def _encode_grid(grid: List[List[int]], encoding: str) -> str:
        if encoding == 'rle':
            flat = [str(cell) for row in grid for cell in row]
            encoded = []
            i = 0
            while i < len(flat):
                current = flat[i]
                count = 1
                while i + count < len(flat) and flat[i + count] == current:
                    count += 1
                encoded.append(f"{current}:{count}")
                i += count
            return ' '.join(encoded)
        return ' '.join(str(cell) for row in grid for cell in row)

    def _load(
        self, split: str, cache_dir: str, max_samples: Optional[int],
    ) -> List[Dict]:
        items = []
        try:
            from datasets import load_dataset
            ds = load_dataset('fchollet/ARC-AGI', split=split, cache_dir=cache_dir)
            for i, row in enumerate(ds):
                if max_samples is not None and i >= max_samples:
                    break
                inp = row['input']
                out = row['output']
                inp_text = self._encode_grid(inp, self.grid_encoding)
                out_text = self._encode_grid(out, self.grid_encoding)
                prompt = f"Input: {inp_text} Output: "
                items.append({
                    'input_grid': inp,
                    'output_grid': out,
                    'prompt_text': prompt,
                    'prompt_ids': self.tokenizer.encode(prompt),
                    'difficulty': len(inp) * len(inp[0]) if inp else 0,
                    'gold_text': out_text,
                })
            if items:
                print(f"[arc-eval] Loaded {len(items)} puzzles from ARC-AGI/{split}")
                return items
        except Exception as e:
            print(f"[arc-eval] HF unavailable ({e}), using synthetic fallback")

        # Synthetic proxy puzzles
        for i in range(max_samples or 30):
            size = 3 + (i % 3)
            inp = [[(r + c + i) % 10 for c in range(size)] for r in range(size)]
            out = [[cell + 1 if cell < 9 else 0 for cell in row] for row in inp]
            inp_text = self._encode_grid(inp, self.grid_encoding)
            out_text = self._encode_grid(out, self.grid_encoding)
            prompt = f"Input: {inp_text} Output: "
            items.append({
                'input_grid': inp,
                'output_grid': out,
                'prompt_text': prompt,
                'prompt_ids': self.tokenizer.encode(prompt),
                'difficulty': size * size,
                'gold_text': out_text,
            })
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        return self.items[idx]


# ─── Evaluation ───────────────────────────────────────────────────────────────

@dataclass
class ARCEvalResult:
    n_total: int = 0
    n_exact: int = 0
    mean_cell_acc: float = 0.0
    exact_match_rate: float = 0.0
    by_difficulty: Dict[str, Dict] = field(default_factory=dict)
    adaptive_compute: Dict[str, float] = field(default_factory=dict)
    predictions: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'n_total': self.n_total,
            'n_exact': self.n_exact,
            'mean_cell_acc': self.mean_cell_acc,
            'exact_match_rate': self.exact_match_rate,
            'by_difficulty': self.by_difficulty,
            'adaptive_compute': self.adaptive_compute,
            'predictions': self.predictions,
        }


def _difficulty_bucket(size: int) -> str:
    if size <= 9:
        return 'easy'
    if size <= 25:
        return 'medium'
    return 'hard'


@torch.no_grad()
def evaluate_arc(
    model,
    dataset: Optional[ARCEvalDataset] = None,
    split: str = 'train',
    max_samples: Optional[int] = None,
    max_new_tokens: int = 256,
    device: torch.device = torch.device('cpu'),
    collect_adaptive: bool = True,
    verbose: bool = False,
) -> ARCEvalResult:
    """
    Evaluate ARC grid prediction with cell accuracy and exact-match rate.
    Optionally tracks mean active cell count per difficulty bucket.
    """
    ds = dataset or ARCEvalDataset(split=split, max_samples=max_samples)
    model = model.to(device)
    model.eval()
    tokenizer = CharTokenizer()
    result = ARCEvalResult()

    bucket_stats: Dict[str, Dict] = {}
    bucket_cells: Dict[str, List[float]] = {}

    for item in ds:
        x = torch.tensor([item['prompt_ids']], dtype=torch.long, device=device)
        collect_metrics = collect_adaptive and hasattr(model, 'forward')
        logits, _, metrics = model(x, collect_metrics=collect_metrics)
        # Greedy decode continuation
        gen_ids = item['prompt_ids'][:]
        for _ in range(max_new_tokens):
            xc = torch.tensor([gen_ids[-model.max_seq_len:]], dtype=torch.long, device=device)
            logits, _, m = model(xc, collect_metrics=collect_metrics)
            nxt = logits[0, -1].argmax().item()
            gen_ids.append(nxt)
            decoded = tokenizer.decode(gen_ids)
            if len(decoded) > len(item['prompt_text']) + len(item['gold_text']) + 10:
                break

        gen_text = tokenizer.decode(gen_ids)
        output_part = gen_text[len(item['prompt_text']):]
        pred_grid = parse_grid_text(output_part)
        gold = item['output_grid']
        cell_acc = grid_cell_accuracy(pred_grid, gold)
        exact = grid_exact_match(pred_grid, gold)

        result.n_total += 1
        result.n_exact += int(exact)
        bucket = _difficulty_bucket(item['difficulty'])
        if bucket not in bucket_stats:
            bucket_stats[bucket] = {'n': 0, 'cell_acc_sum': 0.0, 'exact': 0}
        bucket_stats[bucket]['n'] += 1
        bucket_stats[bucket]['cell_acc_sum'] += cell_acc
        bucket_stats[bucket]['exact'] += int(exact)

        if collect_adaptive and metrics is not None and metrics.n_cells_per_step:
            mean_n = sum(metrics.n_cells_per_step) / len(metrics.n_cells_per_step)
            bucket_cells.setdefault(bucket, []).append(mean_n)

        result.predictions.append({
            'difficulty': item['difficulty'],
            'bucket': bucket,
            'cell_acc': cell_acc,
            'exact': exact,
            'pred_grid': pred_grid,
        })
        if verbose:
            mark = '✓' if exact else f'{cell_acc:.0%}'
            print(f"  [{bucket}] {mark} diff={item['difficulty']}")

    if result.n_total:
        result.mean_cell_acc = sum(p['cell_acc'] for p in result.predictions) / result.n_total
        result.exact_match_rate = result.n_exact / result.n_total

    for bucket, stats in bucket_stats.items():
        n = stats['n']
        result.by_difficulty[bucket] = {
            'n': n,
            'mean_cell_acc': stats['cell_acc_sum'] / max(1, n),
            'exact_match_rate': stats['exact'] / max(1, n),
        }
        if bucket in bucket_cells:
            cells = bucket_cells[bucket]
            result.adaptive_compute[bucket] = sum(cells) / len(cells)

    return result
