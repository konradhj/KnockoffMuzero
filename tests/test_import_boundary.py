"""The critical-divide guarantee: nothing under muzero/ai/ may reference
muzero/simworlds/. Grepping beats any amount of documentation."""
from pathlib import Path


AI_ROOT = Path(__file__).resolve().parent.parent / "muzero" / "ai"


def test_ai_never_imports_simworlds():
    offenders = []
    for py in AI_ROOT.rglob("*.py"):
        text = py.read_text()
        if "muzero.simworlds" in text or "from muzero.simworlds" in text \
                or "from ..simworlds" in text or "from ...simworlds" in text:
            offenders.append(str(py))
    assert not offenders, (
        "AI core imports game-specific code (critical-divide violation):\n"
        + "\n".join(offenders)
    )
