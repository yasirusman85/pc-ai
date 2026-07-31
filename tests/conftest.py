"""Pytest config: put src/crf_reasoning on sys.path (project uses flat imports)."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / 'src' / 'crf_reasoning'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
