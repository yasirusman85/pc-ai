"""
eval_gsm8k.py — GSM8K exact-match evaluation (Phase 4)
======================================================
Evaluates models on final numeric answer correctness, not token-level
next-char accuracy. Parses #### markers from GSM8K CoT format.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from data import CharTokenizer


# ─── Answer parsing ───────────────────────────────────────────────────────────

def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extract final numeric answer from GSM8K CoT text or model output."""
    if '####' in text:
        return text.split('####')[-1].strip()
    # Fallback: last number in text
    nums = re.findall(r'-?\d[\d,]*\.?\d*', text)
    return nums[-1].strip() if nums else None


def normalize_numeric_answer(ans: Optional[str]) -> Optional[str]:
    """Normalize numeric strings for comparison (strip commas, whitespace)."""
    if ans is None:
        return None
    ans = ans.strip().replace(',', '').replace('$', '').replace('%', '')
    # Remove trailing period
    ans = ans.rstrip('.')
    if not ans:
        return None
    try:
        val = float(ans)
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return ans.lower()


def answers_match(pred: Optional[str], gold: Optional[str]) -> bool:
    """Return True if normalized numeric answers match."""
    p = normalize_numeric_answer(pred)
    g = normalize_numeric_answer(gold)
    if p is None or g is None:
        return False
    return p == g


# ─── Eval dataset with metadata ─────────────────────────────────────────────

class GSM8KEvalDataset(Dataset):
    """
    GSM8K items preserving question/answer metadata for exact-match eval.
    Each item: (prompt_ids, gold_answer_str, question_str).
    """

    def __init__(
        self,
        split: str = 'test',
        max_samples: Optional[int] = None,
        tokenizer: Optional[CharTokenizer] = None,
        cache_dir: str = '/tmp/gsm8k_cache',
        prompt_prefix: str = 'Q: ',
        answer_prefix: str = ' A: ',
    ):
        self.tokenizer = tokenizer or CharTokenizer()
        self.prompt_prefix = prompt_prefix
        self.answer_prefix = answer_prefix
        self.items: List[Dict] = self._load(split, cache_dir, max_samples)

    def _load(
        self, split: str, cache_dir: str, max_samples: Optional[int],
    ) -> List[Dict]:
        items = []
        try:
            from datasets import load_dataset
            ds = load_dataset('openai/gsm8k', 'main', split=split, cache_dir=cache_dir)
            for i, row in enumerate(ds):
                if max_samples is not None and i >= max_samples:
                    break
                question = row['question']
                full_answer = row['answer']
                gold = extract_gsm8k_answer(full_answer)
                prompt_text = f"{self.prompt_prefix}{question}{self.answer_prefix}"
                items.append({
                    'question': question,
                    'full_answer': full_answer,
                    'gold_answer': gold,
                    'prompt_text': prompt_text,
                    'prompt_ids': self.tokenizer.encode(prompt_text),
                })
            if items:
                print(f"[gsm8k-eval] Loaded {len(items)} items from GSM8K/{split}")
                return items
        except Exception as e:
            print(f"[gsm8k-eval] HF unavailable ({e}), using synthetic fallback")

        # Synthetic fallback
        for i in range(max_samples or 50):
            a, b = (i + 3) % 17 + 1, (i + 7) % 13 + 1
            question = f"What is {a} plus {b}?"
            gold = str(a + b)
            prompt_text = f"{self.prompt_prefix}{question}{self.answer_prefix}"
            items.append({
                'question': question,
                'full_answer': f"#### {gold}",
                'gold_answer': gold,
                'prompt_text': prompt_text,
                'prompt_ids': self.tokenizer.encode(prompt_text),
            })
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        return self.items[idx]


# ─── Generation + scoring ───────────────────────────────────────────────────

@dataclass
class GSM8KEvalResult:
    n_total: int = 0
    n_correct: int = 0
    accuracy: float = 0.0
    predictions: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'n_total': self.n_total,
            'n_correct': self.n_correct,
            'accuracy': self.accuracy,
            'predictions': self.predictions,
        }


@torch.no_grad()
def generate_answer(
    model,
    prompt_ids: List[int],
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    device: torch.device = torch.device('cpu'),
) -> str:
    """Greedy (or sampled) continuation from prompt token ids."""
    model.eval()
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    tokenizer = CharTokenizer()

    for _ in range(max_new_tokens):
        xc = x[:, -model.max_seq_len:]
        logits, _, _ = model(xc)
        logits = logits[:, -1, :]
        if temperature <= 0:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1)
        x = torch.cat([x, nxt], dim=1)
        decoded = tokenizer.decode(x[0].tolist())
        if '####' in decoded[len(tokenizer.decode(prompt_ids)):]:
            break

    full = tokenizer.decode(x[0].tolist())
    # Return only generated portion
    prompt_str = tokenizer.decode(prompt_ids)
    return full[len(prompt_str):] if full.startswith(prompt_str) else full


@torch.no_grad()
def evaluate_gsm8k_exact_match(
    model,
    dataset: Optional[GSM8KEvalDataset] = None,
    split: str = 'test',
    max_samples: Optional[int] = None,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    device: torch.device = torch.device('cpu'),
    verbose: bool = False,
) -> GSM8KEvalResult:
    """
    Run exact-match evaluation on GSM8K test set.

    Metric: fraction of questions where extracted numeric answer matches gold.
    """
    ds = dataset or GSM8KEvalDataset(split=split, max_samples=max_samples)
    model = model.to(device)
    result = GSM8KEvalResult()

    for item in ds:
        gen_text = generate_answer(
            model, item['prompt_ids'],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
        )
        pred = extract_gsm8k_answer(gen_text)
        gold = item['gold_answer']
        correct = answers_match(pred, gold)
        result.n_total += 1
        result.n_correct += int(correct)
        entry = {
            'question': item['question'],
            'gold': gold,
            'pred': pred,
            'correct': correct,
            'generated': gen_text[:200],
        }
        result.predictions.append(entry)
        if verbose:
            mark = '✓' if correct else '✗'
            print(f"  {mark} gold={gold} pred={pred}")

    result.accuracy = result.n_correct / max(1, result.n_total)
    return result
