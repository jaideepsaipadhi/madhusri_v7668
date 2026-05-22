#!/usr/bin/env python3
import json
import sys
from pathlib import Path

# This script is run from the challenge repo root by final_placer/placer.py.
REPO_ROOT = Path.cwd().resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch


def load_candidate(pt_path: Path):
    obj = torch.load(pt_path, map_location="cpu", weights_only=False)

    if isinstance(obj, dict):
        if "placement" in obj:
            return obj["placement"].float()
        if "macro_positions" in obj:
            return obj["macro_positions"].float()
        raise RuntimeError(f"No placement tensor in {pt_path}; keys={list(obj.keys())}")

    return obj.float()


class SavedCandidatePlacer:
    def __init__(self, placement):
        self.placement = placement

    def place(self, benchmark):
        return self.placement


def get_field(obj, *names, default=None):
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def scalar_costs(obj):
    if isinstance(obj, dict):
        items = obj.items()
    else:
        try:
            items = vars(obj).items()
        except TypeError:
            return {}

    out = {}
    for k, v in items:
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
    return out


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: score_saved_candidate.py <bench> <candidate.pt> <out.json>")

    bench = sys.argv[1]
    pt_path = Path(sys.argv[2])
    out_json = Path(sys.argv[3])

    from macro_place.evaluate import evaluate_benchmark

    testcase_root = REPO_ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04"

    placement = load_candidate(pt_path)
    result = evaluate_benchmark(SavedCandidatePlacer(placement), bench, str(testcase_root))

    proxy = get_field(result, "proxy", "proxy_cost", "cost", "total_cost")
    overlaps = get_field(result, "overlaps", "overlap_count", default=0)
    valid = get_field(result, "valid", default=None)

    if proxy is None:
        raise RuntimeError(f"Could not extract proxy from evaluate_benchmark result={result!r}")

    overlaps = int(overlaps or 0)
    if valid is None:
        valid = overlaps == 0

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "pt": str(pt_path),
        "proxy": float(proxy),
        "valid": bool(valid),
        "overlaps": overlaps,
        "costs": scalar_costs(result),
    }))


if __name__ == "__main__":
    main()
