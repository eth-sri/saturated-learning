import math
import re

BOXED = re.compile(r"\\boxed\{([^{}]+)\}")


def verify(generation: str, ground_truth) -> tuple[str | None, bool]:
    matches = list(BOXED.finditer(generation))
    if not matches:
        return None, False

    raw = matches[-1].group(1).strip()

    try:
        pred = int(raw)
    except ValueError:
        try:
            f = float(raw)
            if not math.isfinite(f) or f != int(f):
                return raw, False
            pred = int(f)
        except (ValueError, OverflowError):
            return raw, False

    try:
        return str(pred), pred == int(ground_truth)
    except (ValueError, TypeError):
        return raw, False
