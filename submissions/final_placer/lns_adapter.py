from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path


def run_lns_safely(
    bench,
    benchmark,
    plc,
    placement,
    root: Path,
    timeout_deadline: float,
    compute_proxy_cost,
):
    """
    DREAMPlace-only adapter.

    Calls submissions/final_placer/lns_engine.py::run_lns.
    No XPlace dependency.
    """
    if os.environ.get("FINAL_RUN_LNS", "1") != "1":
        print("[lns_adapter] FINAL_RUN_LNS != 1; skipping LNS", flush=True)
        costs = compute_proxy_cost(placement, benchmark, plc)
        return placement, costs

    try:
        here = Path(__file__).resolve().parent
        if str(here) not in sys.path:
            sys.path.insert(0, str(here))

        import lns_engine

        print("[lns_adapter] calling DREAMPlace-only exact old-style lns_engine.run_lns", flush=True)

        new_placement, new_costs = lns_engine.run_lns(
            bench=bench,
            benchmark=benchmark,
            plc=plc,
            placement=placement,
            root=root,
            timeout_deadline=timeout_deadline,
        )

        print(f"[lns_adapter] LNS returned costs={new_costs}", flush=True)
        return new_placement, new_costs

    except Exception:
        print("[lns_adapter] LNS crashed; falling back to input placement", flush=True)
        traceback.print_exc()
        costs = compute_proxy_cost(placement, benchmark, plc)
        return placement, costs
