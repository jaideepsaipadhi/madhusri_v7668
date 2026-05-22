#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost


def json_safe(x):
    try:
        import numpy as np
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
    except Exception:
        pass

    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return x.item()
        return x.detach().cpu().tolist()

    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}

    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]

    return x


def main():
    bench = sys.argv[1]
    pt = Path(sys.argv[2])
    out_json = Path(sys.argv[3])

    benchmark, plc = load_benchmark_from_dir(f"external/MacroPlacement/Testcases/ICCAD04/{bench}")

    obj = torch.load(pt, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        placement = obj["placement"].float()
    else:
        placement = obj.float()

    costs = compute_proxy_cost(placement, benchmark, plc)

    result = {
        "pt": str(pt),
        "proxy": float(costs["proxy_cost"]),
        "wl": float(costs["wirelength_cost"]),
        "den": float(costs["density_cost"]),
        "cong": float(costs["congestion_cost"]),
        "overlaps": int(costs.get("overlap_count", 0)),
        "valid": int(costs.get("overlap_count", 0)) == 0,
        "costs": json_safe(costs),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
