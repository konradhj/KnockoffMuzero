"""Quick audit: find float literals > 1.0 inside .py files outside configs/.

Catches the embarrassment of a hardcoded learning_rate=1e-3 slipping past the
config. Not authoritative -- literal-finding is heuristic -- but useful.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NEEDLE = re.compile(r"(?<![\w.])(?:\d+\.\d+|\d+\.\d*e[+-]?\d+|\d+e[+-]?\d+)")
SKIP_DIRS = {"tests", "scripts", ".venv", "checkpoints", "logs"}


def main() -> int:
    offenders = []
    for py in ROOT.rglob("*.py"):
        if any(part in SKIP_DIRS for part in py.parts):
            continue
        if "configs/" in py.as_posix():
            continue
        lines = py.read_text().splitlines()
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for m in NEEDLE.finditer(line):
                s = m.group(0)
                try:
                    v = float(s)
                except ValueError:
                    continue
                if v >= 1.0:
                    offenders.append(f"{py.relative_to(ROOT)}:{i}: {s}   // {line.strip()}")
    if offenders:
        print("Possible hardcoded scalars outside configs/ (review manually):")
        for o in offenders:
            print(" -", o)
        return 1
    print("OK: no suspicious literals found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
