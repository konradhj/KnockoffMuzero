from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any


class RunLogger:
    """Dual jsonl + stdout logger. Flushes after every record."""

    def __init__(self, log_dir: str | Path, to_jsonl: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.to_jsonl = to_jsonl
        self._path = self.log_dir / "run.jsonl"
        if to_jsonl and not self._path.exists():
            self._path.touch()
        self._t0 = time.time()

    def log(self, event: str, **fields: Any) -> None:
        record = {"t": round(time.time() - self._t0, 3), "event": event, **fields}
        line = json.dumps(record, default=_coerce)
        if self.to_jsonl:
            with open(self._path, "a") as f:
                f.write(line + "\n")
        print(line, file=sys.stdout, flush=True)


def _coerce(o: Any) -> Any:
    try:
        return float(o)
    except (TypeError, ValueError):
        return str(o)
