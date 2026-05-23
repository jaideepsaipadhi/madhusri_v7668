from __future__ import annotations
import subprocess
import concurrent.futures

import os

# =============================================================================
# FINAL SUBMISSION DEFAULTS
# =============================================================================
# These match the tested stable pipeline:
#   27-config DREAMPlace L1
#   DREAMPlace L2 off
#   quick GPU zero-overlap repair on
#   old DREAMPlace final legalizer on
#   LNS Layer 1 only
# =============================================================================
os.environ.setdefault("FINAL_L1_MAX_CONFIGS", "0")
os.environ.setdefault("FINAL_RUN_L2", "0")
os.environ.setdefault("FINAL_RUN_FULL_REPAIR_RERUN", "0")
os.environ.setdefault("FINAL_QUICK_GPU_ZERO_REPAIR", "1")
os.environ.setdefault("FINAL_QUICK_GPU_MAX_START_OVERLAPS", "64")
os.environ.setdefault("FINAL_OLD_DREAMPLACE_LEGALIZER", "1")
os.environ.setdefault("FINAL_OLD_LEGALIZER_PAIRWISE_ITERS", "10")
os.environ.setdefault("FINAL_RUN_LNS", "1")
os.environ.setdefault("FINAL_LNS_RUN_CONT", "0")
os.environ.setdefault("FINAL_TIMEOUT_SEC", "3550")
os.environ.setdefault("FINAL_LNS_MIN_TIME_SEC", "2000")
os.environ.setdefault("FINAL_LNS_MIN_TIME_REMAINING_SEC", "2000")
os.environ.setdefault("FINAL_RETURN_MARGIN_SEC", "30")

import re
import sys
import json
import time
import shutil
import signal
import importlib.util
from pathlib import Path

import torch
import multiprocessing as mp

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost


REPO = Path(os.environ.get("MACRO_PLACE_REPO", Path.cwd())).resolve()
EVAL_PY = Path(os.environ.get("EVAL_PY", sys.executable))
ICCAD_ROOT = REPO / "external/MacroPlacement/Testcases/ICCAD04"


class DPConfig:
    __slots__ = ("seed", "target_density", "density_weight")

    def __init__(self, seed, target_density, density_weight):
        self.seed = str(seed)
        self.target_density = str(target_density)
        self.density_weight = str(density_weight)

    def __repr__(self):
        return (
            f"DPConfig(seed={self.seed!r}, "
            f"target_density={self.target_density!r}, "
            f"density_weight={self.density_weight!r})"
        )

class FinalPlacerTimeout(RuntimeError):
    pass


def infer_bench_name(benchmark: Benchmark) -> str:
    for attr in ["name", "benchmark_name", "bench_name", "id", "path", "root"]:
        if hasattr(benchmark, attr):
            value = str(getattr(benchmark, attr))
            m = re.search(r"ibm\d{2}", value)
            if m:
                return m.group(0)

    joined = " ".join(sys.argv)
    m = re.search(r"ibm\d{2}", joined)
    if m:
        return m.group(0)

    env = os.environ.get("FINAL_PLACER_BENCH")
    if env:
        return env

    raise RuntimeError("Could not infer benchmark name.")


def json_safe(obj):
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]

    return obj


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_dreamplace_placer_module():
    return load_module(
        REPO / "submissions/dreamplace_only/placer.py",
        "dreamplace_only_placer_final",
    )


def load_lns_module():
    return load_module(
        REPO / "lns_work/portfolio_layer1_targeted_continuation_lns.py",
        "portfolio_layer1_targeted_continuation_lns_final",
    )


def set_dreamplace_env(cfg: DPConfig, fast: bool = False):
    os.environ["EXPLICIT_ENV_OVERRIDE"] = "1"
    os.environ["DREAMPLACE_FORCE"] = "1"
    os.environ["DREAMPLACE_RANDOM_SEED"] = str(cfg.seed)
    os.environ["DREAMPLACE_TARGET_DENSITY"] = str(cfg.target_density)
    os.environ["DREAMPLACE_DENSITY_WEIGHT"] = str(cfg.density_weight)
    os.environ["DREAMPLACE_OMIT_DENSITY_WEIGHT"] = "0"
    os.environ["DREAMPLACE_DETERMINISTIC"] = "1"
    os.environ["SOFT_MODE"] = "off"

    os.environ["DREAMPLACE_ROOT"] = os.environ.get("DREAMPLACE_ROOT", str(Path(os.environ.get("DREAMPLACE_ROOT", "/workspace/DREAMPlace/install"))))
    os.environ["DREAMPLACE_PYTHON"] = os.environ.get("DREAMPLACE_PYTHON", sys.executable)
    os.environ.setdefault("DREAMPLACE_TIMEOUT_SEC", "3300")

    if fast:
        os.environ["FAST_SWEEP_MODE"] = "1"
        os.environ.setdefault("FAST_SWEEP_GPU_REPULSE_ITERS", "2000")
    else:
        os.environ.pop("FAST_SWEEP_MODE", None)


def l1_configs():
    """
    Full DREAMPlace L1 grid:
      3 seeds × 3 target densities × 3 density weights = 27 configs.
    """
    seeds = [999, 1000, 1001]
    target_densities = [0.75, 0.80, 0.85]
    density_weights = [4e-4, 8e-4, 1.2e-3]

    rows = []
    for seed in seeds:
        for td in target_densities:
            for dw in density_weights:
                rows.append(DPConfig(str(seed), str(td), str(dw)))
    return rows


def l2_configs_from_l1(rows, topk: int = 3):
    """
    DREAMPlace L2 config refinement around top scored L1 basins.

    Important:
      This is config-space refinement, not placement warm-start refinement.
      Current run_dreamplace_config does not inject a saved .pt as the initial placement.

    Selection:
      top-K scored L1 rows, preferring:
        valid first,
        then fewer overlaps,
        then lower proxy.

    Refinement:
      parent td plus small offsets
      parent density weight times local multipliers
      same parent seed
    """
    scored = [
        r for r in rows
        if r.get("proxy") is not None and r.get("cfg") is not None and r.get("pt")
    ]

    scored.sort(
        key=lambda r: (
            int(not bool(r.get("valid", False))),
            int(r.get("overlaps", 10**9) if r.get("overlaps") is not None else 10**9),
            float(r.get("proxy", 1e9)),
        )
    )

    parents = scored[: int(topk)]

    td_offsets = [-0.025, -0.0125, 0.0, 0.0125, 0.025]
    dw_mults = [0.75, 1.00, 1.25]

    out = []
    seen = set()

    for parent in parents:
        cfg = parent["cfg"]

        seed = cfg.seed if hasattr(cfg, "seed") else str(cfg["seed"])
        td0 = float(cfg.target_density if hasattr(cfg, "target_density") else cfg["target_density"])
        dw0 = float(cfg.density_weight if hasattr(cfg, "density_weight") else cfg["density_weight"])

        print(
            f"[dreamplace-l2-configs] parent={parent.get('name')} "
            f"proxy={parent.get('proxy')} overlaps={parent.get('overlaps')} valid={parent.get('valid')} "
            f"seed={seed} td0={td0} dw0={dw0}",
            flush=True,
        )

        for off in td_offsets:
            td = min(0.95, max(0.55, td0 + off))
            for mult in dw_mults:
                dw = max(1e-6, dw0 * mult)

                c = DPConfig(
                    str(seed),
                    f"{td:.6g}",
                    f"{dw:.6g}",
                )

                key = (c.seed, c.target_density, c.density_weight)
                if key in seen:
                    continue

                seen.add(key)
                out.append(c)

    print(f"[dreamplace-l2-configs] parents={len(parents)} configs={len(out)}", flush=True)
    return out


def run_dreamplace_config(bench: str, benchmark, plc, placer_mod, cfg: DPConfig, root: Path, idx: int, layer: str):
    """
    Fast DREAMPlace config run.

    Important:
      - uses fast=True
      - saves placement only
      - does NOT compute_proxy_cost here
      - layer scoring happens later in parallel
    """
    set_dreamplace_env(cfg, fast=True)

    os.environ["DREAMPLACE_OUT_ROOT"] = str(root / "dreamplace_runs" / layer / f"cfg_{idx:04d}")

    placer = placer_mod.MyPlacer()

    t0 = time.time()
    placement = placer.place(benchmark).detach().cpu().float()
    runtime = time.time() - t0

    cand_dir = root / "candidates" / layer
    cand_dir.mkdir(parents=True, exist_ok=True)

    name = (
        f"{layer}_s{cfg.seed}_td{cfg.target_density}_dw{cfg.density_weight}"
        .replace(".", "p")
        .replace("-", "m")
        .replace("+", "")
    )
    pt = cand_dir / f"{name}.pt"

    torch.save(
        {
            "placement": placement,
            "cfg": {
                "seed": cfg.seed,
                "target_density": cfg.target_density,
                "density_weight": cfg.density_weight,
            },
            "bench": bench,
        },
        pt,
    )

    row = {
        "name": name,
        "cfg": cfg,
        "pt": str(pt),
        "runtime": runtime,
        "proxy": None,
        "valid": False,
        "overlaps": None,
        "costs": None,
    }

    print(
        f"[dreamplace-{layer}] idx={idx} seed={cfg.seed} td={cfg.target_density} "
        f"dw={cfg.density_weight} saved={pt} runtime={runtime:.2f}s",
        flush=True,
    )

    return row


def select_best_valid(rows):
    valid = [r for r in rows if r.get("valid") and r.get("proxy") is not None and r.get("pt")]
    valid.sort(key=lambda r: r["proxy"])
    return valid[0] if valid else None


def select_best_invalid(rows):
    """
    If no VALID candidates exist, choose the INVALID placement with the best
    official proxy cost. This intentionally uses compute_proxy_cost on invalids
    as a quality signal before full repair.
    """
    invalid = [
        r for r in rows
        if (not r.get("valid"))
        and r.get("pt")
        and r.get("proxy") is not None
    ]

    if not invalid:
        return None

    invalid.sort(
        key=lambda r: (
            float(r.get("proxy", 1e99)),
            int(r.get("overlaps", 10**9)),
            float((r.get("costs") or {}).get("total_overlap_area", 1e99)),
        )
    )
    return invalid[0]


def load_candidate(pt: str | Path):
    obj = torch.load(pt, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        if "placement" in obj:
            return obj["placement"].float()
        if "macro_positions" in obj:
            return obj["macro_positions"].float()
        raise RuntimeError(f"No placement tensor in {pt}; keys={list(obj.keys())}")
    return obj.float()



def score_candidate_subprocess(bench: str, pt: str, name: str, root: Path):
    """
    Score one saved placement in a separate Python process.
    This avoids pickling problems inside dynamically loaded submissions.
    """
    out_dir = root / "score_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{name}.json"

    cmd = [
        str(EVAL_PY),
        str(REPO / "experiments/score_saved_candidate.py"),
        bench,
        str(pt),
        str(out_json),
    ]

    p = subprocess.run(
        cmd,
        cwd=str(REPO),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    if p.returncode != 0 or not out_json.exists():
        return {
            "name": name,
            "pt": pt,
            "error": p.stdout[-3000:],
            "proxy": None,
            "valid": False,
            "overlaps": None,
        }

    data = json.loads(out_json.read_text())
    data["name"] = name
    return data


def score_layer_parallel(bench: str, rows: list, root: Path, workers: int):
    jobs = [r for r in rows if r.get("pt")]

    print(f"[score-parallel] scoring {len(jobs)} placements workers={workers}", flush=True)

    if not jobs:
        return rows

    name_to_row = {r["name"]: r for r in rows}

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(workers)) as ex:
        futs = [
            ex.submit(score_candidate_subprocess, bench, r["pt"], r["name"], root)
            for r in jobs
        ]

        for fut in concurrent.futures.as_completed(futs):
            res = fut.result()
            done += 1
            row = name_to_row[res["name"]]

            if res.get("proxy") is not None:
                row["proxy"] = float(res["proxy"])
                row["valid"] = bool(res["valid"])
                row["overlaps"] = int(res["overlaps"])
                row["costs"] = res["costs"]

                print(
                    f"[score-parallel] {done}/{len(jobs)} {res['name']} "
                    f"proxy={res['proxy']:.6f} overlaps={res['overlaps']} valid={res['valid']}",
                    flush=True,
                )
            else:
                row["proxy"] = None
                row["valid"] = False
                row["overlaps"] = None
                row["error"] = res.get("error", "")
                print(f"[score-parallel] {done}/{len(jobs)} {res['name']} FAILED", flush=True)

    return rows


def run_layer(bench, benchmark, plc, placer_mod, configs, root: Path, layer: str, timeout_deadline: float):
    rows = []

    max_configs_env = os.environ.get(f"FINAL_{layer.upper()}_MAX_CONFIGS")
    if max_configs_env is not None and str(max_configs_env).strip() != "":
        max_configs = int(max_configs_env)
        # Convention:
        #   unset or 0 => run all configs
        #   positive N  => run first N configs
        if max_configs > 0:
            configs = configs[:max_configs]

    print(f"[dreamplace-{layer}] configs={len(configs)}", flush=True)

    for idx, cfg in enumerate(configs, 1):
        if time.time() > timeout_deadline:
            print(f"[dreamplace-{layer}] timeout before config {idx}", flush=True)
            break

        try:
            row = run_dreamplace_config(bench, benchmark, plc, placer_mod, cfg, root, idx, layer)
        except Exception as e:
            print(f"[dreamplace-{layer}] FAILED idx={idx} cfg={cfg} err={repr(e)}", flush=True)
            row = {
                "name": f"{layer}_failed_{idx}",
                "cfg": cfg,
                "pt": None,
                "runtime": 0.0,
                "proxy": None,
                "valid": False,
                "overlaps": None,
                "error": repr(e),
            }

        rows.append(row)

        # Save frequently.
        serializable = []
        for r in rows:
            rr = dict(r)
            if isinstance(rr.get("cfg"), DPConfig):
                rr["cfg"] = {
                    "seed": rr["cfg"].seed,
                    "target_density": rr["cfg"].target_density,
                    "density_weight": rr["cfg"].density_weight,
                }
            serializable.append(json_safe(rr))
        (root / f"{layer}_results.json").write_text(json.dumps(serializable, indent=2))

    rows = score_layer_parallel(
        bench=bench,
        rows=rows,
        root=root,
        workers=int(os.environ.get("FINAL_SCORE_WORKERS", "12")),
    )

    # Save scored rows.
    serializable = []
    for r in rows:
        rr = dict(r)
        if isinstance(rr.get("cfg"), DPConfig):
            rr["cfg"] = {
                "seed": rr["cfg"].seed,
                "target_density": rr["cfg"].target_density,
                "density_weight": rr["cfg"].density_weight,
            }
        serializable.append(json_safe(rr))
    (root / f"{layer}_results.json").write_text(json.dumps(serializable, indent=2))

    return rows



def repair_best_invalid_candidate(placement, benchmark, placer_mod):
    """
    Full repair for exactly one selected invalid candidate.

    Used only when L1/L2 produces no valid placement.
    """
    print("[final_placer] no valid placement found; repairing best invalid candidate with full legalization", flush=True)

    out = placement.clone()

    if hasattr(placer_mod, "legalize_hard_macros_post_dreamplace"):
        out = placer_mod.legalize_hard_macros_post_dreamplace(out, benchmark)
    else:
        print("[final_placer] WARNING: placer_mod has no legalize_hard_macros_post_dreamplace", flush=True)

    if hasattr(placer_mod, "final_bounds_repair"):
        out = placer_mod.final_bounds_repair(out, benchmark)

    if hasattr(placer_mod, "final_legality_margin_repair"):
        out = placer_mod.final_legality_margin_repair(out, benchmark)

    return out




# ======================================================================
# MULTIPROCESS PAIRWISE REPAIR
# ======================================================================

def detect_hard_overlap_pairs(placement, benchmark):
    """
    Return all overlapping hard-macro pairs.
    """
    pairs = []

    widths = benchmark.node_widths.cpu()
    heights = benchmark.node_heights.cpu()

    n = int(placement.shape[0])

    pos = placement.detach().cpu()

    for i in range(n):
        xi = float(pos[i, 0])
        yi = float(pos[i, 1])
        wi = float(widths[i])
        hi = float(heights[i])

        for j in range(i + 1, n):

            xj = float(pos[j, 0])
            yj = float(pos[j, 1])
            wj = float(widths[j])
            hj = float(heights[j])

            ox = min(xi + wi, xj + wj) - max(xi, xj)
            oy = min(yi + hi, yj + hj) - max(yi, yj)

            if ox > 0 and oy > 0:
                pairs.append((i, j))

    return pairs


def _pairwise_chunk_worker(args):
    import torch

    (
        placement_cpu,
        pairs,
        widths,
        heights,
        canvas_w,
        canvas_h,
        margin,
    ) = args

    placement = placement_cpu.clone()

    best = None

    for (i, j) in pairs:

        xi = float(placement[i, 0])
        yi = float(placement[i, 1])

        xj = float(placement[j, 0])
        yj = float(placement[j, 1])

        wi = float(widths[i])
        hi = float(heights[i])

        wj = float(widths[j])
        hj = float(heights[j])

        ox = min(xi + wi, xj + wj) - max(xi, xj)
        oy = min(yi + hi, yj + hj) - max(yi, yj)

        if ox <= 0 or oy <= 0:
            continue

        area = ox * oy

        move_j = (wj * hj) < (wi * hi)

        if move_j:
            idx = j
            sign = 1.0 if xj >= xi else -1.0
            dx = sign * (ox + margin)
        else:
            idx = i
            sign = -1.0 if xi >= xj else 1.0
            dx = sign * (ox + margin)

        dy = 0.0

        cand_x = float(placement[idx, 0]) + dx
        cand_y = float(placement[idx, 1]) + dy

        cand_x = max(0.0, min(cand_x, canvas_w - float(widths[idx])))
        cand_y = max(0.0, min(cand_y, canvas_h - float(heights[idx])))

        move_mag = abs(dx) + abs(dy)

        score = (
            area,
            -move_mag,
        )

        if best is None or score > best["score"]:
            best = {
                "idx": idx,
                "dx": cand_x - float(placement[idx, 0]),
                "dy": cand_y - float(placement[idx, 1]),
                "score": score,
            }

    return best




def ensure_benchmark_geometry_aliases(benchmark):
    """
    Compatibility shim for repair code that expects:
      benchmark.node_widths
      benchmark.node_heights

    Some challenge Benchmark objects only expose:
      benchmark.node_sizes
      benchmark.macro_sizes

    This function adds aliases when possible.
    """
    if hasattr(benchmark, "node_widths") and hasattr(benchmark, "node_heights"):
        return benchmark

    sizes = None
    for attr in ("node_sizes", "macro_sizes", "sizes"):
        if hasattr(benchmark, attr):
            sizes = getattr(benchmark, attr)
            break

    if sizes is None:
        print("[geometry-alias] WARNING: no node_sizes/macro_sizes found", flush=True)
        return benchmark

    try:
        if hasattr(sizes, "detach"):
            # torch tensor shape [N, 2]
            widths = sizes[:, 0]
            heights = sizes[:, 1]
        else:
            import torch
            ts = torch.as_tensor(sizes)
            widths = ts[:, 0]
            heights = ts[:, 1]

        setattr(benchmark, "node_widths", widths)
        setattr(benchmark, "node_heights", heights)
        print("[geometry-alias] added benchmark.node_widths/node_heights", flush=True)

    except Exception as e:
        print(f"[geometry-alias] failed to create aliases: {e!r}", flush=True)

    return benchmark



def count_hard_overlaps(*call_args, **kwargs):
    """
    Compatibility helper for finalist repair.

    Counts rectangle overlaps among macro-like nodes. This is intentionally
    conservative and robust across possible call signatures:
      count_hard_overlaps(benchmark, placement)
      count_hard_overlaps(placement, benchmark)
      count_hard_overlaps(benchmark, plc, placement)

    Returns an integer overlap count.
    """
    benchmark = None
    placement = None

    for obj in call_args:
        if hasattr(obj, "clone") and hasattr(obj, "shape"):
            placement = obj
        elif benchmark is None and (
            hasattr(obj, "macro_sizes")
            or hasattr(obj, "node_sizes")
            or hasattr(obj, "hard_macro_indices")
            or hasattr(obj, "macro_indices")
        ):
            benchmark = obj

    if benchmark is None or placement is None:
        print("[count_hard_overlaps] WARNING: could not infer benchmark/placement; returning 0", flush=True)
        return 0

    # Get sizes.
    sizes = None
    for attr in ("macro_sizes", "node_sizes", "sizes"):
        if hasattr(benchmark, attr):
            sizes = getattr(benchmark, attr)
            break

    if sizes is None:
        print("[count_hard_overlaps] WARNING: no sizes found; returning 0", flush=True)
        return 0

    # Convert to Python lists cheaply.
    try:
        import torch
        if hasattr(placement, "detach"):
            pos = placement.detach().cpu()
        else:
            pos = torch.as_tensor(placement).cpu()

        if hasattr(sizes, "detach"):
            sz = sizes.detach().cpu()
        else:
            sz = torch.as_tensor(sizes).cpu()
    except Exception:
        print("[count_hard_overlaps] WARNING: tensor conversion failed; returning 0", flush=True)
        return 0

    n = min(int(pos.shape[0]), int(sz.shape[0]))

    # Prefer explicit hard macro indices if benchmark provides them.
    ids = None
    for attr in (
        "hard_macro_indices",
        "hard_macro_ids",
        "movable_hard_macro_indices",
        "movable_hard_macro_ids",
        "macro_indices",
        "macro_ids",
    ):
        if hasattr(benchmark, attr):
            try:
                raw = getattr(benchmark, attr)
                ids = [int(x) for x in list(raw) if 0 <= int(x) < n]
                if ids:
                    break
            except Exception:
                ids = None

    # Fallback: use nodes with positive width/height, and for large designs
    # keep the larger half by area as a hard-macro approximation.
    if not ids:
        areas = []
        for i in range(n):
            try:
                w = float(sz[i][0])
                h = float(sz[i][1])
            except Exception:
                continue
            if w > 0 and h > 0:
                areas.append((w * h, i))

        if len(areas) > 1200:
            areas.sort(reverse=True)
            ids = [i for _, i in areas[: max(1, len(areas) // 2)]]
        else:
            ids = [i for _, i in areas]

    rects = []
    for i in ids:
        try:
            x = float(pos[i][0])
            y = float(pos[i][1])
            w = float(sz[i][0])
            h = float(sz[i][1])
        except Exception:
            continue

        if w <= 0 or h <= 0:
            continue

        # Treat placement as center coordinates, which matches MacroPlacement convention.
        rects.append((x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0))

    overlaps = 0
    m = len(rects)
    eps = 1e-9

    for a in range(m):
        ax1, ay1, ax2, ay2 = rects[a]
        for b in range(a + 1, m):
            bx1, by1, bx2, by2 = rects[b]
            if ax1 < bx2 - eps and ax2 > bx1 + eps and ay1 < by2 - eps and ay2 > by1 + eps:
                overlaps += 1

    return int(overlaps)



def targeted_pairwise_repair(
    placement,
    benchmark,
    max_iters=100,
    margin=1e-4,
):
    """
    Multiprocess pairwise overlap repair.
    """

    import concurrent.futures

    out = placement.clone()

    widths = benchmark.node_widths.cpu()
    heights = benchmark.node_heights.cpu()

    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)

    workers = int(os.environ.get("FINAL_FALLBACK_PAIRWISE_WORKERS", "12"))

    for it in range(max_iters):

        overlaps = detect_hard_overlap_pairs(out, benchmark)

        if not overlaps:
            print(f"[pairwise] converged iter={it}", flush=True)
            break

        chunks = [[] for _ in range(workers)]

        for k, pair in enumerate(overlaps):
            chunks[k % workers].append(pair)

        args = []

        out_cpu = out.detach().cpu()

        for chunk in chunks:
            if chunk:
                args.append(
                    (
                        out_cpu,
                        chunk,
                        widths,
                        heights,
                        canvas_w,
                        canvas_h,
                        margin,
                    )
                )

        proposals = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:

            futs = [ex.submit(_pairwise_chunk_worker, a) for a in args]

            for fut in concurrent.futures.as_completed(futs):
                try:
                    r = fut.result()
                    if r is not None:
                        proposals.append(r)
                except Exception as e:
                    print(f"[pairwise] worker failed: {repr(e)}", flush=True)

        if not proposals:
            print(f"[pairwise] no proposals iter={it}", flush=True)
            break

        proposals.sort(key=lambda x: x["score"], reverse=True)

        touched = set()
        applied = 0

        for p in proposals:

            idx = int(p["idx"])

            if idx in touched:
                continue

            touched.add(idx)

            out[idx, 0] += float(p["dx"])
            out[idx, 1] += float(p["dy"])

            out[idx, 0] = max(
                0.0,
                min(float(out[idx, 0]), canvas_w - float(widths[idx])),
            )

            out[idx, 1] = max(
                0.0,
                min(float(out[idx, 1]), canvas_h - float(heights[idx])),
            )

            applied += 1

        remaining = count_hard_overlaps(out, benchmark)

        print(
            f"[pairwise] iter={it+1}/{max_iters} "
            f"pairs={len(overlaps)} "
            f"applied={applied} "
            f"remaining={remaining}",
            flush=True,
        )

        if remaining == 0:
            break

    return out

# ======================================================================


def select_invalid_finalists(rows, k=3):
    """
    Pick up to three invalid finalists:
      1. lowest overlap count / overlap area
      2. lowest proxy cost, distinct
      3. second-lowest overlap count / overlap area, distinct
    """
    invalid = []
    for r in rows:
        if r.get("valid"):
            continue
        if not r.get("pt"):
            continue
        if r.get("proxy") is None:
            continue

        costs = r.get("costs") or {}
        overlaps = int(r.get("overlaps", costs.get("overlap_count", 10**9)))
        area = float(costs.get("total_overlap_area", 1e99))
        max_area = float(costs.get("max_overlap_area", 1e99))
        proxy = float(r.get("proxy", 1e99))

        invalid.append({
            "row": r,
            "overlaps": overlaps,
            "area": area,
            "max_area": max_area,
            "proxy": proxy,
            "pt": str(r["pt"]),
        })

    if not invalid:
        return []

    picks = []
    used = set()

    def add_candidate(cands, reason):
        for c in cands:
            key = c["pt"]
            if key in used:
                continue
            used.add(key)
            rr = dict(c["row"])
            rr["_fallback_reason"] = reason
            rr["_fallback_rank_info"] = {
                "overlaps": c["overlaps"],
                "area": c["area"],
                "max_area": c["max_area"],
                "proxy": c["proxy"],
            }
            picks.append(rr)
            return

    by_overlap = sorted(invalid, key=lambda c: (c["overlaps"], c["area"], c["max_area"], c["proxy"]))
    by_proxy = sorted(invalid, key=lambda c: (c["proxy"], c["overlaps"], c["area"], c["max_area"]))

    add_candidate(by_overlap, "lowest_overlap")
    add_candidate(by_proxy, "lowest_proxy")
    add_candidate(by_overlap, "second_lowest_overlap")

    return picks[:k]


def repair_single_invalid_finalist(row, benchmark, plc, label):
    """
    Repair one invalid finalist:
      targeted pairwise polish first
      GPU shelf fallback if still overlapping
      score with compute_proxy_cost
    """
    placement = load_candidate(row["pt"])
    start_overlaps = count_hard_overlaps(placement, benchmark)
    print(
        f"[final_placer] repair finalist {label}: name={row.get('name')} "
        f"reason={row.get('_fallback_reason')} proxy={row.get('proxy')} "
        f"overlaps={row.get('overlaps')} strict_overlaps={start_overlaps}",
        flush=True,
    )

    t0 = time.time()

    # 1. Targeted pairwise only.
    placement = targeted_pairwise_repair(
        placement,
        benchmark,
        max_iters=int(os.environ.get("FINAL_FALLBACK_PAIRWISE_ITERS", "25")),
        margin=float(os.environ.get("FINAL_FALLBACK_PAIRWISE_MARGIN", "1e-4")),
    )
    pair_overlaps = count_hard_overlaps(placement, benchmark)
    print(
        f"[final_placer] repair finalist {label}: after pairwise overlaps={pair_overlaps} "
        f"time={time.time() - t0:.2f}s",
        flush=True,
    )

    # 2. GPU shelf fallback until clean or max passes.
    shelf_passes = int(os.environ.get("FINAL_FALLBACK_SHELF_PASSES", "3"))
    for pass_id in range(shelf_passes):
        if pair_overlaps == 0:
            break
        placement = final_shelf_fallback(
            placement,
            benchmark,
            margin=float(os.environ.get("FINAL_FALLBACK_SHELF_MARGIN", "1e-4")),
        )
        pair_overlaps = count_hard_overlaps(placement, benchmark)
        print(
            f"[final_placer] repair finalist {label}: after shelf pass {pass_id+1}/{shelf_passes} "
            f"overlaps={pair_overlaps}",
            flush=True,
        )

    # 3. Final score.
    costs = compute_proxy_cost(placement, benchmark, plc)
    proxy = float(costs["proxy_cost"])
    overlaps = int(costs.get("overlap_count", 0))
    valid = overlaps == 0

    print(
        f"[final_placer] repair finalist {label}: repaired proxy={proxy:.6f} "
        f"official_overlaps={overlaps} valid={valid} costs={json_safe(costs)}",
        flush=True,
    )

    return {
        "source": row,
        "placement": placement,
        "costs": costs,
        "proxy": proxy,
        "overlaps": overlaps,
        "valid": valid,
        "runtime": time.time() - t0,
    }


def repair_invalid_finalists_and_choose(rows, benchmark, plc, root):
    """
    Try up to three invalid finalists and choose best repaired valid.
    If none repair valid, choose lowest-overlap repaired placement.
    """
    finalists = select_invalid_finalists(rows, k=3)

    if not finalists:
        print("[final_placer] no invalid finalists available", flush=True)
        return None, None

    repaired = []
    for idx, row in enumerate(finalists, 1):
        try:
            rr = repair_single_invalid_finalist(row, benchmark, plc, label=f"{idx}/{len(finalists)}")
            repaired.append(rr)

            out = root / f"repaired_invalid_finalist_{idx}.pt"
            torch.save(
                {
                    "placement": rr["placement"].detach().cpu(),
                    "costs": json_safe(rr["costs"]),
                    "source": json_safe(row),
                },
                out,
            )
            print(f"[final_placer] saved repaired finalist {idx}: {out}", flush=True)

        except Exception as e:
            print(f"[final_placer] repair finalist {idx} failed: {repr(e)}", flush=True)

    if not repaired:
        return None, None

    valid = [r for r in repaired if r["valid"]]
    if valid:
        valid.sort(key=lambda r: r["proxy"])
        best = valid[0]
    else:
        repaired.sort(key=lambda r: (r["overlaps"], r["proxy"]))
        best = repaired[0]

    print(
        f"[final_placer] selected repaired invalid finalist proxy={best['proxy']:.6f} "
        f"valid={best['valid']} overlaps={best['overlaps']}",
        flush=True,
    )

    return best["placement"], best["costs"]




def select_rows_for_full_repair_rerun(rows, topk: int = 3):
    usable = [
        r for r in rows
        if r.get("proxy") is not None and r.get("cfg") is not None and r.get("pt")
    ]

    invalid = [r for r in usable if not r.get("valid")]
    valid = [r for r in usable if r.get("valid")]

    selected = []
    used = set()

    def add_many(cands, reason):
        for r in cands:
            if len(selected) >= topk:
                return
            key = str(r.get("pt"))
            if key in used:
                continue
            used.add(key)
            rr = dict(r)
            rr["_full_repair_reason"] = reason
            selected.append(rr)

    add_many(
        sorted(invalid, key=lambda r: (
            int(r.get("overlaps", 10**9) if r.get("overlaps") is not None else 10**9),
            float(r.get("proxy", 1e9)),
        )),
        "lowest_invalid_overlap",
    )

    add_many(
        sorted(invalid, key=lambda r: (
            float(r.get("proxy", 1e9)),
            int(r.get("overlaps", 10**9) if r.get("overlaps") is not None else 10**9),
        )),
        "lowest_invalid_proxy",
    )

    add_many(
        sorted(valid, key=lambda r: float(r.get("proxy", 1e9))),
        "best_valid",
    )

    print(
        "[full-repair-rerun] selected="
        + str([(r.get("name"), r.get("_full_repair_reason"), r.get("proxy"), r.get("overlaps"), r.get("valid")) for r in selected]),
        flush=True,
    )
    return selected


def run_full_repair_reruns(bench, benchmark, plc, placer_mod, rows, root: Path, timeout_deadline: float):
    if os.environ.get("FINAL_RUN_FULL_REPAIR_RERUN", "1") != "1":
        print("[full-repair-rerun] disabled", flush=True)
        return []

    topk = int(os.environ.get("FINAL_FULL_REPAIR_TOPK", "3"))
    selected = select_rows_for_full_repair_rerun(rows, topk=topk)

    out = []

    for idx, parent in enumerate(selected, 1):
        if time.time() > timeout_deadline:
            print(f"[full-repair-rerun] timeout before idx={idx}", flush=True)
            break

        cfg = parent.get("cfg")
        if cfg is None:
            continue

        try:
            print(
                f"[full-repair-rerun] idx={idx}/{len(selected)} "
                f"parent={parent.get('name')} proxy={parent.get('proxy')} "
                f"overlaps={parent.get('overlaps')} reason={parent.get('_full_repair_reason')} cfg={cfg}",
                flush=True,
            )

            row = run_dreamplace_config(
                bench=bench,
                benchmark=benchmark,
                plc=plc,
                placer_mod=placer_mod,
                cfg=cfg,
                root=root,
                idx=idx,
                layer="repair",
                fast=False,
                name_suffix=f"_full_from_{idx}",
            )

            row["parent_name"] = parent.get("name")
            row["parent_proxy"] = parent.get("proxy")
            row["parent_overlaps"] = parent.get("overlaps")
            row["_repair_reason"] = parent.get("_full_repair_reason")
            out.append(row)

        except Exception as e:
            print(f"[full-repair-rerun] FAILED idx={idx} err={e!r}", flush=True)

    out = score_layer_parallel(
        bench=bench,
        rows=out,
        root=root,
        workers=int(os.environ.get("FINAL_SCORE_WORKERS", "12")),
    )

    print(
        "[full-repair-rerun] scored="
        + str([(r.get("name"), r.get("proxy"), r.get("overlaps"), r.get("valid"), r.get("parent_name")) for r in out]),
        flush=True,
    )

    return out


def select_three_finalists(rows):
    """
    Finalists:
      1. best VALID by proxy
      2. INVALID with lowest overlaps
      3. INVALID with lowest proxy
    """
    valid = []
    invalid = []

    for r in rows:
        if r.get("proxy") is None or not r.get("pt"):
            continue

        overlaps = int(
            r.get(
                "overlaps",
                (r.get("costs") or {}).get("overlap_count", 10**9),
            )
        )

        proxy = float(r.get("proxy", 1e99))

        item = {
            "row": r,
            "proxy": proxy,
            "overlaps": overlaps,
            "pt": str(r["pt"]),
        }

        if r.get("valid"):
            valid.append(item)
        else:
            invalid.append(item)

    finalists = []
    used = set()

    def add(items, reason):
        for x in items:
            if x["pt"] in used:
                continue

            used.add(x["pt"])

            rr = dict(x["row"])
            rr["_finalist_reason"] = reason

            finalists.append(rr)
            return

    valid.sort(key=lambda x: x["proxy"])
    invalid_overlap = sorted(invalid, key=lambda x: (x["overlaps"], x["proxy"]))
    invalid_proxy = sorted(invalid, key=lambda x: (x["proxy"], x["overlaps"]))

    add(valid, "best_valid")
    add(invalid_overlap, "lowest_invalid_overlap")
    add(invalid_proxy, "lowest_invalid_proxy")

    print(
        "[final_placer] finalists="
        + str([
            (
                f.get("name"),
                f.get("_finalist_reason"),
                f.get("proxy"),
                f.get("overlaps"),
            )
            for f in finalists
        ]),
        flush=True,
    )

    return finalists




def quick_gpu_zero_overlap_repair(placement, benchmark, plc):
    """
    Fast finalist-stage GPU-only overlap repair.

    Important behavior:
      - Try the known-good DREAMPlace GPU repulsion call once.
      - If it reaches zero overlaps, accept it immediately.
      - If it improves but does not reach zero, do NOT accept it here; let the
        old DREAMPlace final legalizer continue with GPU repulsion -> pairwise
        polish -> shelf fallback.
      - Avoid older fallback attempts that expect benchmark.device and throw:
        AttributeError("'Benchmark' object has no attribute 'device'")
    """
    from macro_place.objective import compute_proxy_cost

    def score(place):
        costs = compute_proxy_cost(place, benchmark, plc)
        proxy = float(costs.get("proxy_cost", costs.get("proxy", 1e99)))
        overlaps = int(costs.get("overlap_count", costs.get("overlaps", 10**9)))
        return costs, proxy, overlaps

    try:
        start_costs, start_proxy, start_overlaps = score(placement)
    except Exception as e:
        print(f"[quick-gpu-zero] could not score start placement: {e!r}", flush=True)
        return placement, None, False

    max_start = int(os.environ.get("FINAL_QUICK_GPU_ZERO_MAX_START_OVERLAPS", "64"))
    print(
        f"[quick-gpu-zero] start proxy={start_proxy:.6f} "
        f"overlaps={start_overlaps} max_start_overlaps={max_start}",
        flush=True,
    )

    if start_overlaps == 0:
        print("[quick-gpu-zero] already zero overlaps", flush=True)
        return placement, start_costs, True

    if start_overlaps > max_start:
        print("[quick-gpu-zero] skipped; too many starting overlaps", flush=True)
        return placement, start_costs, False

    try:
        mod = load_dreamplace_placer_module()
    except Exception as e:
        print(f"[quick-gpu-zero] could not load DREAMPlace placer module: {e!r}", flush=True)
        return placement, start_costs, False

    fn = getattr(mod, "legalize_hard_macros_gpu_repulsion", None)
    if fn is None:
        print("[quick-gpu-zero] legalize_hard_macros_gpu_repulsion missing", flush=True)
        return placement, start_costs, False

    print("[quick-gpu-zero] trying legalize_hard_macros_gpu_repulsion", flush=True)

    try:
        out = fn(placement, benchmark, max_iters=int(os.environ.get("FINAL_QUICK_GPU_MAX_ITERS", "2000")))

        # legalize_hard_macros_gpu_repulsion has had multiple return shapes
        # across our restored DREAMPlace code. Pick the tensor-like placement,
        # not a PlacementCost/object.
        if isinstance(out, tuple):
            q_place = None
            for item in out:
                if hasattr(item, "shape") and hasattr(item, "float"):
                    q_place = item
                    break
            if q_place is None:
                raise TypeError(f"no tensor-like placement in return tuple types={[type(x).__name__ for x in out]}")
        else:
            q_place = out

        q_costs, q_proxy, q_overlaps = score(q_place)
        print(
            f"[quick-gpu-zero] legalize_hard_macros_gpu_repulsion result "
            f"proxy={q_proxy:.6f} overlaps={q_overlaps}",
            flush=True,
        )

        if q_overlaps == 0:
            print("[quick-gpu-zero] GPU-only zero-overlap success", flush=True)
            return q_place, q_costs, True

        print("[quick-gpu-zero] GPU-only did not reach zero; continuing to old legalizer", flush=True)
        return placement, start_costs, False

    except Exception as e:
        print(f"[quick-gpu-zero] legalize_hard_macros_gpu_repulsion attempt failed: {e!r}", flush=True)
        return placement, start_costs, False


def run_old_dreamplace_final_legalizer(placement, benchmark):
    """
    Final fallback repair using the old DREAMPlace hard-macro legalizer:

      [DREAMPlace legalizer] hard overlaps before: ...
      [DREAMPlace legalizer] after GPU repulsion: ...
      [DREAMPlace legalizer] after pairwise polish: ...
      [DREAMPlace legalizer] after shelf fallback: 0

    This is only used in finalist repair, not the initial DREAMPlace fast sweep.
    """
    import os

    print("[final_placer] running OLD DREAMPlace final legalizer", flush=True)

    mod = load_dreamplace_placer_module()

    if not hasattr(mod, "legalize_hard_macros_post_dreamplace"):
        print("[final_placer] old DREAMPlace legalizer missing; returning unchanged", flush=True)
        return placement

    # Preserve env.
    old_env = {
        "SKIP_PAIRWISE_POLISH": os.environ.get("SKIP_PAIRWISE_POLISH"),
        "SKIP_SHELF_FALLBACK": os.environ.get("SKIP_SHELF_FALLBACK"),
        "PAIRWISE_POLISH_ITERS": os.environ.get("PAIRWISE_POLISH_ITERS"),
        "GPU_REPULSION_ITERS": os.environ.get("GPU_REPULSION_ITERS"),
        "GPU_REPULSION_MARGIN": os.environ.get("GPU_REPULSION_MARGIN"),
    }

    # Force old full legalizer stages on.
    os.environ["SKIP_PAIRWISE_POLISH"] = "0"
    os.environ["SKIP_SHELF_FALLBACK"] = "0"
    os.environ["PAIRWISE_POLISH_ITERS"] = os.environ.get("FINAL_OLD_LEGALIZER_PAIRWISE_ITERS", "10")
    os.environ.setdefault("GPU_REPULSION_ITERS", "2000")
    os.environ.setdefault("GPU_REPULSION_MARGIN", "0.0001")

    try:
        out = mod.legalize_hard_macros_post_dreamplace(placement, benchmark)
        if hasattr(out, "detach"):
            out = out.detach().cpu().float()
        return out

    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v



def run_finalist_pipeline(row, benchmark, plc, root, idx):
    benchmark = ensure_benchmark_geometry_aliases(benchmark)

    placement = load_candidate(row["pt"])

    print(
        f"[final_placer] finalist {idx} start "
        f"name={row.get('name')} "
        f"reason={row.get('_finalist_reason')} "
        f"proxy={row.get('proxy')} "
        f"overlaps={row.get('overlaps')}",
        flush=True,
    )

    # ------------------------------------------------------------
    # QUICK GPU ZERO-OVERLAP REPAIR
    # ------------------------------------------------------------
    # This restores the old fast behavior where GPU repulsion alone could
    # turn a near-valid invalid into a valid candidate.
    if os.environ.get("FINAL_QUICK_GPU_ZERO_REPAIR", "1") == "1":
        q_place, q_costs, q_ok = quick_gpu_zero_overlap_repair(placement, benchmark, plc)
        if q_ok:
            q_proxy = float(q_costs["proxy_cost"])
            q_overlaps = int(q_costs.get("overlap_count", 0))
            print(
                f"[final_placer] finalist {idx} QUICK_GPU_ZERO success "
                f"proxy={q_proxy:.6f} overlaps={q_overlaps}",
                flush=True,
            )
            return {
                "placement": q_place,
                "costs": q_costs,
                "proxy": q_proxy,
                "overlaps": q_overlaps,
                "valid": q_overlaps == 0,
            }
        else:
            placement = q_place

    # ------------------------------------------------------------
    # FINAL OLD DREAMPLACE LEGALIZER
    # ------------------------------------------------------------
    # If quick GPU zero-overlap did not succeed, run the old full legalizer:
    # GPU repulsion -> pairwise polish -> shelf fallback.
    if os.environ.get("FINAL_OLD_DREAMPLACE_LEGALIZER", "1") == "1":
        placement = run_old_dreamplace_final_legalizer(placement, benchmark)

    old_leg = count_hard_overlaps(placement, benchmark)

    print(
        f"[final_placer] finalist {idx} after OLD DreamPlace legalizer overlaps={old_leg}",
        flush=True,
    )

    costs = compute_proxy_cost(placement, benchmark, plc)

    proxy = float(costs["proxy_cost"])
    overlaps = int(costs.get("overlap_count", 0))
    valid = overlaps == 0

    print(
        f"[final_placer] finalist {idx} final "
        f"proxy={proxy:.6f} overlaps={overlaps} valid={valid}",
        flush=True,
    )

    return {
        "placement": placement,
        "costs": costs,
        "proxy": proxy,
        "overlaps": overlaps,
        "valid": valid,
    }






def high_intensity_gpu_micro_polish(*call_args, **kwargs):
    """
    Real finalist-stage GPU micro-polish.

    Important:
      This does NOT change the initial DREAMPlace/fast_sweep GPU repulsion.
      It only runs later inside run_finalist_pipeline during legalization.

    Goal:
      Take a near-valid candidate with a small number of hard overlaps and
      gently separate overlapping macro rectangles on GPU.

    Expected call shape:
      high_intensity_gpu_micro_polish(placement, benchmark, max_iters=...)
    """
    import os
    import math
    import torch

    print("[micro-polish] real GPU micro-polish active", flush=True)

    placement = None
    benchmark = None

    for obj in call_args:
        if hasattr(obj, "clone") and hasattr(obj, "shape"):
            placement = obj
        elif benchmark is None and (
            hasattr(obj, "node_sizes")
            or hasattr(obj, "macro_sizes")
            or hasattr(obj, "node_widths")
            or hasattr(obj, "node_heights")
        ):
            benchmark = obj

    if placement is None or benchmark is None:
        print("[micro-polish] could not infer placement/benchmark; returning unchanged", flush=True)
        return call_args[0] if call_args else placement

    max_iters = int(kwargs.get("max_iters", os.environ.get("FINAL_FALLBACK_MICRO_ITERS", "1000")))
    if max_iters <= 0:
        print("[micro-polish] max_iters <= 0; returning unchanged", flush=True)
        return placement

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pos0 = placement.detach().float().cpu()
    pos = pos0.clone().to(device)

    # Get sizes.
    sizes = None
    for attr in ("node_sizes", "macro_sizes", "sizes"):
        if hasattr(benchmark, attr):
            obj = getattr(benchmark, attr)
            sizes = obj.detach().float().cpu() if hasattr(obj, "detach") else torch.as_tensor(obj, dtype=torch.float32)
            break

    if sizes is None and hasattr(benchmark, "node_widths") and hasattr(benchmark, "node_heights"):
        w = getattr(benchmark, "node_widths")
        h = getattr(benchmark, "node_heights")
        w = w.detach().float().cpu() if hasattr(w, "detach") else torch.as_tensor(w, dtype=torch.float32)
        h = h.detach().float().cpu() if hasattr(h, "detach") else torch.as_tensor(h, dtype=torch.float32)
        sizes = torch.stack([w, h], dim=1)

    if sizes is None:
        print("[micro-polish] no sizes found; returning unchanged", flush=True)
        return placement

    n = min(pos.shape[0], sizes.shape[0])
    pos = pos[:n]
    sizes = sizes[:n].to(device)

    # Prefer explicit hard macro ids if benchmark exposes them.
    ids = None
    for attr in (
        "hard_macro_indices",
        "hard_macro_ids",
        "movable_hard_macro_indices",
        "movable_hard_macro_ids",
        "macro_indices",
        "macro_ids",
    ):
        if hasattr(benchmark, attr):
            try:
                raw = getattr(benchmark, attr)
                ids = torch.tensor([int(x) for x in list(raw) if 0 <= int(x) < n], dtype=torch.long)
                if ids.numel() > 0:
                    break
            except Exception:
                ids = None

    # Fallback: use largest-area nodes as macro candidates.
    if ids is None or ids.numel() == 0:
        area_cpu = (sizes[:, 0].detach().cpu().clamp_min(0) * sizes[:, 1].detach().cpu().clamp_min(0))
        positive = torch.nonzero(area_cpu > 0, as_tuple=False).flatten()
        if positive.numel() == 0:
            print("[micro-polish] no positive-size nodes; returning unchanged", flush=True)
            return placement

        # Keep a manageable macro-like subset. For ibm10 this should cover the hard macro population.
        k = min(int(os.environ.get("FINAL_MICRO_POLISH_MAX_MACROS", "900")), int(positive.numel()))
        top = torch.topk(area_cpu[positive], k=k, largest=True).indices
        ids = positive[top].long()

    ids = ids.to(device)
    m = int(ids.numel())

    if m <= 1:
        print("[micro-polish] not enough macro ids; returning unchanged", flush=True)
        return placement

    psel = pos[ids].clone()
    ssel = sizes[ids].clone()

    # Canvas / clamp bounds. Use the current placement extent with size padding.
    half = ssel / 2.0
    min_x = torch.min(psel[:, 0] - half[:, 0])
    max_x = torch.max(psel[:, 0] + half[:, 0])
    min_y = torch.min(psel[:, 1] - half[:, 1])
    max_y = torch.max(psel[:, 1] + half[:, 1])

    x_lo = min_x + half[:, 0]
    x_hi = max_x - half[:, 0]
    y_lo = min_y + half[:, 1]
    y_hi = max_y - half[:, 1]

    # Conservative controls.
    margin = float(os.environ.get("FINAL_MICRO_POLISH_MARGIN", "1e-4"))
    lr = float(os.environ.get("FINAL_MICRO_POLISH_LR", "0.08"))
    max_step = float(os.environ.get("FINAL_MICRO_POLISH_MAX_STEP", "0.02"))
    log_every = max(1, int(os.environ.get("FINAL_MICRO_POLISH_LOG_EVERY", "100")))

    # Cap iterations for this O(m^2) GPU pass.
    max_iters = min(max_iters, int(os.environ.get("FINAL_MICRO_POLISH_ITER_CAP", "2000")))

    def overlap_count_and_push(cur):
        x = cur[:, 0]
        y = cur[:, 1]
        w = ssel[:, 0]
        h = ssel[:, 1]

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]

        ox = (w[:, None] + w[None, :]) * 0.5 + margin - torch.abs(dx)
        oy = (h[:, None] + h[None, :]) * 0.5 + margin - torch.abs(dy)

        mask = (ox > 0) & (oy > 0)
        mask.fill_diagonal_(False)

        # Count each overlap once.
        upper = torch.triu(mask, diagonal=1)
        count = int(upper.sum().detach().cpu().item())

        if count == 0:
            return count, torch.zeros_like(cur)

        # Push along smaller penetration axis.
        use_x = ox < oy

        sx = torch.sign(dx)
        sy = torch.sign(dy)
        sx = torch.where(sx == 0, torch.ones_like(sx), sx)
        sy = torch.where(sy == 0, torch.ones_like(sy), sy)

        push_x = torch.where(use_x & mask, sx * ox, torch.zeros_like(ox))
        push_y = torch.where((~use_x) & mask, sy * oy, torch.zeros_like(oy))

        # Normalize by number of conflicts per macro to avoid huge jumps.
        deg = mask.float().sum(dim=1).clamp_min(1.0)

        force_x = push_x.sum(dim=1) / deg
        force_y = push_y.sum(dim=1) / deg

        force = torch.stack([force_x, force_y], dim=1)
        return count, force

    start_count, _ = overlap_count_and_push(psel)
    print(f"[micro-polish] start macros={m} overlaps={start_count} device={device} max_iters={max_iters}", flush=True)

    best = psel.clone()
    best_count = start_count

    for it in range(1, max_iters + 1):
        cnt, force = overlap_count_and_push(psel)

        if cnt < best_count:
            best_count = cnt
            best = psel.clone()

        if cnt == 0:
            best_count = 0
            best = psel.clone()
            print(f"[micro-polish] converged iter={it}", flush=True)
            break

        step = torch.clamp(lr * force, min=-max_step, max=max_step)
        psel = psel + step

        # Clamp.
        psel[:, 0] = torch.maximum(torch.minimum(psel[:, 0], x_hi), x_lo)
        psel[:, 1] = torch.maximum(torch.minimum(psel[:, 1], y_hi), y_lo)

        if it == 1 or it % log_every == 0:
            print(f"[micro-polish] iter={it} overlaps={cnt} best={best_count}", flush=True)

        # If no progress for long enough, bail.
        if it > 300 and it % 200 == 0 and best_count >= cnt:
            pass

    out = pos.clone()
    out[ids] = best

    final_count, _ = overlap_count_and_push(best)
    print(f"[micro-polish] done start={start_count} best={best_count} final={final_count}", flush=True)

    result = placement.detach().float().cpu().clone()
    result[:n] = out.detach().cpu()

    return result



def final_shelf_fallback(*call_args, **kwargs):
    """
    Safe compatibility shelf fallback.

    Older finalist pipeline expects final_shelf_fallback(...). If the real
    function is missing, this no-op keeps the finalist pipeline alive so it can
    score the repaired finalist and reach the LNS adapter.
    """
    print("[shelf-fallback-shim] final_shelf_fallback shim active; returning unchanged placement", flush=True)

    placement = None
    for obj in call_args:
        if hasattr(obj, "clone") and hasattr(obj, "shape"):
            placement = obj
            break

    if placement is None:
        for obj in call_args:
            if isinstance(obj, (list, tuple)) and len(obj) > 0:
                placement = obj
                break

    if placement is None:
        return call_args[0] if call_args else None

    return placement.clone() if hasattr(placement, "clone") else placement



def run_lns(bench, benchmark, plc, placement, root: Path, timeout_deadline: float):
    print("[final_placer] dispatching to isolated LNS adapter", flush=True)
    import sys
    from pathlib import Path as _Path

    _here = _Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))

    from lns_adapter import run_lns_safely
    return run_lns_safely(
        bench=bench,
        benchmark=benchmark,
        plc=plc,
        placement=placement,
        root=root,
        timeout_deadline=timeout_deadline,
        compute_proxy_cost=compute_proxy_cost,
    )


class FinalPlacer:
    def __init__(self):
        print("[final_placer] DREAMPlace L1/L2 + bilayer LNS initialized", flush=True)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        total_t0 = time.time()
        bench = infer_bench_name(benchmark)

        timeout_sec = float(os.environ.get("FINAL_TIMEOUT_SEC", "3550"))
        deadline = time.time() + timeout_sec

        root_base = Path(os.environ.get("FINAL_PLACER_ROOT", "/dev/shm/dreamplace_final_placer_runs"))
        root = root_base / f"{bench}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{time.time_ns() % 1000000}"
        root.mkdir(parents=True, exist_ok=True)

        benchmark_root = ICCAD_ROOT / bench
        _, plc = __import__("macro_place.loader").loader.load_benchmark_from_dir(str(benchmark_root))

        placer_mod = load_dreamplace_placer_module()

        print(f"[final_placer] bench={bench}", flush=True)
        # Defensive --all isolation: remove stale DREAMPlace export for this benchmark.
        try:
            import shutil
            shutil.rmtree(Path("dreamplace_ibm") / str(bench), ignore_errors=True)
        except Exception as e:
            print(f"[final_placer] warning: could not clean dreamplace_ibm/{bench}: {e!r}", flush=True)
        print(f"[final_placer] root={root}", flush=True)
        print(f"[final_placer] timeout_sec={timeout_sec}", flush=True)

        try:
            # L1
            l1 = run_layer(
                bench,
                benchmark,
                plc,
                placer_mod,
                l1_configs(),
                root,
                "l1",
                deadline - 600.0,  # reserve time for LNS/fallback
            )

            l1_best = select_best_valid(l1)
            if l1_best:
                print(f"[final_placer] L1 best {l1_best['name']} proxy={l1_best['proxy']:.6f}", flush=True)
            else:
                print("[final_placer] no valid L1 candidate", flush=True)

            # DREAMPlace L2 removed for now.
            # Use the full 27-config L1 portfolio, then rerun the strongest
            # near-valid / low-proxy L1 basins with full DREAMPlace legalization.
            print("[dreamplace-l2] disabled/removed; using L1 + full-repair reruns", flush=True)

            repair_rows = []
            if time.time() < deadline - 360.0:
                repair_rows = run_full_repair_reruns(
                    bench=bench,
                    benchmark=benchmark,
                    plc=plc,
                    placer_mod=placer_mod,
                    rows=l1,
                    root=root,
                    timeout_deadline=deadline - 300.0,
                )
            else:
                print(f"[full-repair-rerun] skipped time_left={deadline - time.time():.1f}s", flush=True)

            all_rows = list(l1) + list(repair_rows)

            print(
                f"[final_placer] merged candidates l1={len(l1)} repair={len(repair_rows)} total={len(all_rows)}",
                flush=True,
            )

            best = select_best_valid(all_rows)

            finalists = select_three_finalists(all_rows)

            finalist_results = []

            for idx, row in enumerate(finalists, 1):
                try:
                    rr = run_finalist_pipeline(
                        row,
                        benchmark,
                        plc,
                        root,
                        idx,
                    )

                    finalist_results.append(rr)

                except Exception as e:
                    print(
                        f"[final_placer] finalist {idx} failed: {repr(e)}",
                        flush=True,
                    )

            if not finalist_results:
                print("[final_placer] no finalist results; using initial placement", flush=True)

                placement = benchmark.macro_positions.clone()
                base_proxy = None
                base_costs = None

            else:
                finalist_results.sort(
                    key=lambda r: (
                        not r["valid"],
                        r["proxy"],
                        r["overlaps"],
                    )
                )

                chosen = finalist_results[0]

                placement = chosen["placement"]
                base_proxy = chosen["proxy"]
                base_costs = chosen["costs"]

                print(
                    f"[final_placer] selected finalist "
                    f"proxy={base_proxy:.6f} "
                    f"valid={chosen['valid']} "
                    f"overlaps={chosen['overlaps']}",
                    flush=True,
                )

                repaired_path = root / "selected_finalist_lns_base.pt"

                torch.save(
                    {
                        "placement": placement.detach().cpu(),
                        "bench": bench,
                        "costs": json_safe(base_costs),
                    },
                    repaired_path,
                )

                print(
                    f"[final_placer] saved finalist LNS base {repaired_path}",
                    flush=True,
                )

            # Optional LNS
            lns_min_time = float(os.environ.get("FINAL_LNS_MIN_TIME_REMAINING_SEC", "2000"))
            return_margin = float(os.environ.get("FINAL_RETURN_MARGIN_SEC", "30"))
            time_left = deadline - time.time()

            if os.environ.get("FINAL_RUN_LNS", "1") == "1" and time_left > lns_min_time:
                try:
                    print(
                        f"[final_placer] running LNS time_left={time_left:.1f}s "
                        f"lns_min_time={lns_min_time:.1f}s return_margin={return_margin:.1f}s",
                        flush=True,
                    )

                    placement, costs = run_lns(
                        bench,
                        benchmark,
                        plc,
                        placement,
                        root,
                        deadline - return_margin,
                    )

                    proxy = float(costs["proxy_cost"])
                    print(f"[final_placer] after LNS proxy={proxy:.6f} costs={json_safe(costs)}", flush=True)

                except Exception as e:
                    print(f"[final_placer] LNS failed, returning pre-LNS best: {repr(e)}", flush=True)

            else:
                print(
                    f"[final_placer] skipping LNS: FINAL_RUN_LNS={os.environ.get('FINAL_RUN_LNS', '1')} "
                    f"time_left={time_left:.1f}s lns_min_time={lns_min_time:.1f}s",
                    flush=True,
                )

            final_path = root / "final_placement.pt"
            torch.save(placement.detach().cpu(), final_path)
            print(f"[final_placer] saved {final_path}", flush=True)
            print(f"[final_placer] total_time={time.time() - total_t0:.2f}s", flush=True)

            return placement

        finally:
            persist_root = Path(os.environ.get("FINAL_PERSIST_ROOT", "/workspace/final_placer_runs"))
            persist_root.mkdir(parents=True, exist_ok=True)
            dst = persist_root / root.name

            try:
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(root, dst, ignore=shutil.ignore_patterns("dreamplace_runs"))
                print(f"[final_placer] persisted lightweight results to {dst}", flush=True)
            except Exception as e:
                print(f"[final_placer] persist warning: {repr(e)}", flush=True)


Placer = FinalPlacer
