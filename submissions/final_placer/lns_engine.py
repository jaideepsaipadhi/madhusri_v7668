import os

# =============================================================================
# FINAL LNS DEFAULTS
# =============================================================================
os.environ.setdefault("FINAL_LNS_ROWS", "3")
os.environ.setdefault("FINAL_LNS_COLS", "4")
os.environ.setdefault("FINAL_LNS_WORKERS", "12")
os.environ.setdefault("FINAL_LNS_CHUNK_SIZE", "8")
os.environ.setdefault("FINAL_LNS_TOP_REGION_KEEP", "2")
os.environ.setdefault("FINAL_LNS_TOP_REGION_BY", "macros")
os.environ.setdefault("FINAL_LNS_GPU_PREFILTER_PERCENT", "0.10")
os.environ.setdefault("FINAL_LNS_GPU_PREFILTER_TOPK", "0")
os.environ.setdefault("FINAL_LNS_GPU_PREFILTER_BINS", "32")
os.environ.setdefault("FINAL_LNS_LAYER1_TOP_PERCENT", "0.10")
os.environ.setdefault("FINAL_LNS_LAYER1_TOPK", "0")
os.environ.setdefault("FINAL_LNS_LAYER1_MAX_REJECTS", "10")
os.environ.setdefault("FINAL_LNS_LAYER1_MAX_ACCEPTS", "20")
os.environ.setdefault("FINAL_LNS_RUN_CONT", "0")

import math
import time
import shutil
import random
import argparse
import itertools
import importlib.util
from pathlib import Path
from dataclasses import dataclass
from multiprocessing import get_context
from collections import Counter, defaultdict

import torch

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost


@dataclass(frozen=True)
class Move:
    region: int
    macro_id: int
    new_x: float
    new_y: float
    tag: str
    spec_score: float = 1e99


@dataclass(frozen=True)
class DPConfig:
    seed: str
    target_density: str
    density_weight: str


_GLOBAL = {}


def elapsed(t0):
    return time.time() - t0


def parse_csv(raw: str):
    return [x.strip() for x in raw.split(",") if x.strip()]


def load_myplacer():
    p = Path("submissions/dreamplace_only/placer.py").resolve()
    spec = importlib.util.spec_from_file_location("dreamplace_only_placer", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MyPlacer


def exact_costs(placement, benchmark, plc):
    return compute_proxy_cost(placement, benchmark, plc)


def get_sizes(benchmark):
    if hasattr(benchmark, "macro_sizes"):
        return benchmark.macro_sizes
    if hasattr(benchmark, "node_sizes"):
        return benchmark.node_sizes
    return None


def soft_macro_ids(benchmark):
    hard = int(benchmark.num_hard_macros)
    total = int(benchmark.num_macros)
    return list(range(hard, total))


def legal_move(benchmark, sizes, i, nx, ny):
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)

    if sizes is None:
        return 0.0 <= nx <= canvas_w and 0.0 <= ny <= canvas_h

    w = float(sizes[i, 0])
    h = float(sizes[i, 1])

    return (w / 2.0) <= nx <= (canvas_w - w / 2.0) and (h / 2.0) <= ny <= (canvas_h - h / 2.0)


def assign_region(benchmark, placement, i, rows, cols):
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)

    x = float(placement[i, 0])
    y = float(placement[i, 1])

    col = int((x / max(canvas_w, 1e-9)) * cols)
    row = int((y / max(canvas_h, 1e-9)) * rows)

    col = max(0, min(cols - 1, col))
    row = max(0, min(rows - 1, row))

    return row * cols + col


def set_dreamplace_env(cfg: DPConfig):
    os.environ["EXPLICIT_ENV_OVERRIDE"] = "1"
    os.environ["DREAMPLACE_FORCE"] = "1"
    os.environ["DREAMPLACE_RANDOM_SEED"] = str(cfg.seed)
    os.environ["DREAMPLACE_TARGET_DENSITY"] = str(cfg.target_density)
    os.environ["DREAMPLACE_DENSITY_WEIGHT"] = str(cfg.density_weight)
    os.environ["DREAMPLACE_OMIT_DENSITY_WEIGHT"] = "0"
    os.environ["DREAMPLACE_DETERMINISTIC"] = "1"
    os.environ["SOFT_MODE"] = "off"
    os.environ.setdefault("DREAMPLACE_ROOT", "/workspace/DREAMPlace/install")
    os.environ.setdefault("DREAMPLACE_PYTHON", sys.executable)
    os.environ.setdefault("DREAMPLACE_TIMEOUT_SEC", "3300")


def run_dreamplace_config(bench: str, benchmark, plc, MyPlacer, cfg: DPConfig):
    set_dreamplace_env(cfg)

    out_dir = Path("dreamplace_ibm") / bench
    if out_dir.exists():
        shutil.rmtree(out_dir)

    placer = MyPlacer()

    t0 = time.time()
    placement = placer.place(benchmark)
    dt = time.time() - t0

    costs = exact_costs(placement, benchmark, plc)
    proxy = float(costs["proxy_cost"])
    overlap_count = int(costs.get("overlap_count", 0))

    return placement.clone(), costs, proxy, overlap_count, dt


def init_worker(bench_name: str):
    root = Path("external/MacroPlacement/Testcases/ICCAD04")
    benchmark, plc = load_benchmark_from_dir(root / bench_name)
    _GLOBAL["benchmark"] = benchmark
    _GLOBAL["plc"] = plc


def score_chunk_worker(args):
    placement, chunk_id, region, chunk_moves = args
    benchmark = _GLOBAL["benchmark"]
    plc = _GLOBAL["plc"]

    scored = []

    for base_move in chunk_moves:
        trial = placement.clone()
        trial[base_move.macro_id, 0] = base_move.new_x
        trial[base_move.macro_id, 1] = base_move.new_y

        costs = compute_proxy_cost(trial, benchmark, plc)
        proxy = float(costs["proxy_cost"])

        scored.append(
            Move(
                region=base_move.region,
                macro_id=base_move.macro_id,
                new_x=base_move.new_x,
                new_y=base_move.new_y,
                tag=base_move.tag,
                spec_score=proxy,
            )
        )

    scored.sort(key=lambda m: m.spec_score)

    return {
        "chunk_id": chunk_id,
        "region": region,
        "tested": len(scored),
        "best": scored[0].spec_score if scored else None,
        "moves": scored,
    }


def make_dynamic_chunks(region_moves, chunk_size):
    """
    Normal bilayer LNS chunking.

    No adaptive pruning.
    No hot-region scheduler.
    Just split each region into chunks and score all chunks.
    """
    tasks = []
    chunk_id = 0

    for region, moves in region_moves.items():
        for start in range(0, len(moves), chunk_size):
            chunk = moves[start:start + chunk_size]
            if not chunk:
                continue
            tasks.append((chunk_id, region, chunk))
            chunk_id += 1

    # Smallest chunks first, as requested.
    tasks.sort(key=lambda x: (len(x[2]), x[0]))
    return tasks


def direction_vectors(include_diagonal=False):
    d = {
        "R": (1.0, 0.0),
        "L": (-1.0, 0.0),
        "U": (0.0, 1.0),
        "D": (0.0, -1.0),
    }

    if include_diagonal:
        inv = 1.0 / math.sqrt(2.0)
        d.update({
            "UR": (inv, inv),
            "UL": (-inv, inv),
            "DR": (inv, -inv),
            "DL": (-inv, -inv),
        })

    return d


def generate_moves_for_macro(benchmark, placement, macro_id, region, step_fracs, dirs):
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    base = max(canvas_w, canvas_h)
    sizes = get_sizes(benchmark)

    x = float(placement[macro_id, 0])
    y = float(placement[macro_id, 1])

    moves = []
    seen = set()

    for step_frac in step_fracs:
        step_abs = step_frac * base
        for tag, (dx, dy) in dirs.items():
            nx = x + dx * step_abs
            ny = y + dy * step_abs

            key = (macro_id, round(nx, 9), round(ny, 9), tag)
            if key in seen:
                continue
            seen.add(key)

            if legal_move(benchmark, sizes, macro_id, nx, ny):
                moves.append(Move(region=region, macro_id=macro_id, new_x=nx, new_y=ny, tag=f"{tag}@{step_frac}"))

    return moves


def generate_layer1_moves_by_region(benchmark, placement, rows, cols):
    dirs = direction_vectors(include_diagonal=False)
    region_moves = {r: [] for r in range(rows * cols)}
    region_macro_counts = {r: set() for r in range(rows * cols)}

    for i in soft_macro_ids(benchmark):
        r = assign_region(benchmark, placement, i, rows, cols)
        region_macro_counts[r].add(i)
        region_moves[r].extend(generate_moves_for_macro(benchmark, placement, i, r, [0.25], dirs))

    return region_moves, {r: len(v) for r, v in region_macro_counts.items()}


def run_exact_replay(
    label,
    benchmark,
    plc,
    placement,
    candidates,
    top_percent,
    topk_cap,
    max_consecutive_rejects,
    max_accepts,
    t0,
):
    current_costs = exact_costs(placement, benchmark, plc)
    current_proxy = float(current_costs["proxy_cost"])

    best_placement = placement.clone()
    best_costs = current_costs
    best_proxy = current_proxy

    candidates.sort(key=lambda m: m.spec_score)

    full_pool = len(candidates)
    replay_k = int(math.ceil(full_pool * top_percent)) if top_percent > 0 else full_pool
    if topk_cap > 0:
        replay_k = min(replay_k, topk_cap)
    replay_k = max(0, min(replay_k, full_pool))

    candidates = candidates[:replay_k]

    print(
        f"[{label}-replay] elapsed={elapsed(t0):.2f}s start_proxy={current_proxy:.6f} "
        f"candidate_pool_full={full_pool} candidate_pool_replay={len(candidates)} "
        f"top_percent={top_percent} topk_cap={topk_cap} best_spec={candidates[0].spec_score if candidates else None}",
        flush=True,
    )

    accepted_records = []
    exact_rescored = 0
    accepted = 0
    consecutive_rejects = 0

    for cand in candidates:
        if accepted >= max_accepts:
            print(f"[{label}-stop] hit max_accepts={max_accepts}", flush=True)
            break

        old_x = float(placement[cand.macro_id, 0])
        old_y = float(placement[cand.macro_id, 1])

        placement[cand.macro_id, 0] = cand.new_x
        placement[cand.macro_id, 1] = cand.new_y

        costs = exact_costs(placement, benchmark, plc)
        proxy = float(costs["proxy_cost"])
        delta = proxy - current_proxy
        exact_rescored += 1

        if proxy < current_proxy:
            old_proxy = current_proxy
            current_proxy = proxy
            current_costs = costs
            consecutive_rejects = 0
            accepted += 1

            accepted_records.append({
                "macro_id": cand.macro_id,
                "region": cand.region,
                "tag": cand.tag,
                "old_proxy": old_proxy,
                "new_proxy": current_proxy,
                "delta": delta,
            })

            if current_proxy < best_proxy:
                best_proxy = current_proxy
                best_placement = placement.clone()
                best_costs = costs

            print(
                f"[{label}-feed] elapsed={elapsed(t0):.2f}s macro={cand.macro_id} "
                f"region={cand.region} tag={cand.tag} spec={cand.spec_score:.6f} "
                f"exact={proxy:.6f} delta={delta:.6f} decision=ACCEPT "
                f"{old_proxy:.6f}->{current_proxy:.6f} accepted={accepted}",
                flush=True,
            )
        else:
            placement[cand.macro_id, 0] = old_x
            placement[cand.macro_id, 1] = old_y
            consecutive_rejects += 1

            if max_consecutive_rejects > 0 and consecutive_rejects >= max_consecutive_rejects:
                print(
                    f"[{label}-prune] elapsed={elapsed(t0):.2f}s "
                    f"max_consecutive_rejects={max_consecutive_rejects} current={current_proxy:.6f}",
                    flush=True,
                )
                break

    final_costs = exact_costs(best_placement, benchmark, plc)
    final_proxy = float(final_costs["proxy_cost"])

    print(
        f"[{label}-done] elapsed={elapsed(t0):.2f}s proxy={final_proxy:.6f} "
        f"accepted={accepted} exact_rescored={exact_rescored} costs={final_costs}",
        flush=True,
    )

    return best_placement, final_costs, accepted_records


def parallel_score_candidates(bench, placement, region_moves, workers, chunk_size, t0, label):
    """
    Normal bilayer LNS parallel scoring.

    Scores all chunks produced by make_dynamic_chunks.
    No adaptive hot-region pruning.
    """
    chunk_specs = make_dynamic_chunks(region_moves, chunk_size)
    tasks = [(placement.clone(), chunk_id, region, chunk) for chunk_id, region, chunk in chunk_specs]

    candidates = []
    spec_scored = 0

    total_moves = sum(len(chunk) for _, _, chunk in chunk_specs)

    print(
        f"[{label}-score-start] elapsed={elapsed(t0):.2f}s "
        f"workers={workers} chunks={len(chunk_specs)} total_moves={total_moves}",
        flush=True,
    )

    ctx = get_context("spawn")
    with ctx.Pool(processes=workers, initializer=init_worker, initargs=(bench,)) as pool:
        for done, result in enumerate(pool.imap_unordered(score_chunk_worker, tasks), start=1):
            spec_scored += result["tested"]
            candidates.extend(result["moves"])
            print(
                f"[{label}-chunk] elapsed={elapsed(t0):.2f}s "
                f"done={done}/{len(chunk_specs)} "
                f"done_chunk={result['chunk_id']} "
                f"region={result['region']} "
                f"tested={result['tested']} "
                f"best_spec={result['best']}",
                flush=True,
            )

    print(
        f"[{label}-score-done] elapsed={elapsed(t0):.2f}s "
        f"spec_scored={spec_scored} chunks={len(chunk_specs)}",
        flush=True,
    )

    return candidates, spec_scored, len(chunk_specs)



# =============================================================================
# GPU PRE-SCORE MOVE PREFILTER
# =============================================================================
#
# Purpose:
#   Generate all old-exact LNS moves, but only send the top fraction to
#   expensive chunk scoring.
#
# Example:
#   7152 generated moves
#   FINAL_LNS_GPU_PREFILTER_PERCENT=0.25
#   => only 1788 moves are chunk-scored
# =============================================================================


def _move_get_macro_id(m):
    for name in ("macro_id", "node_id", "idx", "i"):
        if hasattr(m, name):
            try:
                return int(getattr(m, name))
            except Exception:
                pass
        if isinstance(m, dict) and name in m:
            try:
                return int(m[name])
            except Exception:
                pass
    if isinstance(m, (tuple, list)) and len(m) > 0:
        try:
            return int(m[0])
        except Exception:
            pass
    return None


def _move_get_region(m, default_region=0):
    for name in ("region", "region_id", "r"):
        if hasattr(m, name):
            try:
                return int(getattr(m, name))
            except Exception:
                pass
        if isinstance(m, dict) and name in m:
            try:
                return int(m[name])
            except Exception:
                pass
    return int(default_region)


def _move_get_new_xy(m):
    # Common dataclass/dict names.
    for xname, yname in (
        ("new_x", "new_y"),
        ("nx", "ny"),
        ("x_new", "y_new"),
        ("to_x", "to_y"),
        ("x", "y"),
    ):
        if hasattr(m, xname) and hasattr(m, yname):
            try:
                return float(getattr(m, xname)), float(getattr(m, yname))
            except Exception:
                pass
        if isinstance(m, dict) and xname in m and yname in m:
            try:
                return float(m[xname]), float(m[yname])
            except Exception:
                pass

    # Common tuple fallback:
    #   (macro_id, old_x, old_y, new_x, new_y, tag, region)
    if isinstance(m, (tuple, list)) and len(m) >= 5:
        try:
            return float(m[3]), float(m[4])
        except Exception:
            pass

    return None, None


def _benchmark_sizes_tensor(benchmark, device):
    import torch

    for attr in ("node_sizes", "macro_sizes", "sizes"):
        if hasattr(benchmark, attr):
            obj = getattr(benchmark, attr)
            if hasattr(obj, "detach"):
                return obj.detach().to(device=device, dtype=torch.float32)
            return torch.as_tensor(obj, device=device, dtype=torch.float32)

    return None


def _placement_tensor(placement, device):
    import torch

    if hasattr(placement, "detach"):
        return placement.detach().to(device=device, dtype=torch.float32)
    return torch.as_tensor(placement, device=device, dtype=torch.float32)


def gpu_prefilter_region_moves(benchmark, placement, region_moves, label):
    """
    GPU cheap prefilter before expensive chunk scoring.

    Keeps top FINAL_LNS_GPU_PREFILTER_PERCENT of generated moves by a cheap
    density-relief score.

    Score intuition:
      prefer moving macro area from a more crowded grid bin to a less crowded bin
      penalize huge jumps slightly

    This does not decide acceptance. It only decides which moves are worth
    expensive chunk scoring / exact replay.
    """
    import os
    import math
    import torch

    percent = float(os.environ.get("FINAL_LNS_GPU_PREFILTER_PERCENT", "0.25"))
    topk = int(os.environ.get("FINAL_LNS_GPU_PREFILTER_TOPK", "0"))
    bins = int(os.environ.get("FINAL_LNS_GPU_PREFILTER_BINS", "32"))

    if percent <= 0 and topk <= 0:
        return region_moves

    flat = []
    for region, moves in region_moves.items():
        for m in moves:
            flat.append((region, m))

    n = len(flat)
    if n == 0:
        print(f"[gpu-prefilter] {label}: no moves", flush=True)
        return region_moves

    keep_n = n
    if percent > 0:
        keep_n = max(1, int(math.ceil(n * percent)))
    if topk > 0:
        keep_n = min(keep_n, topk)
    keep_n = min(keep_n, n)

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pos = _placement_tensor(placement, device)
        sizes = _benchmark_sizes_tensor(benchmark, device)

        if sizes is None:
            raise RuntimeError("benchmark has no node_sizes/macro_sizes/sizes")

        # Clamp shape.
        n_nodes = min(int(pos.shape[0]), int(sizes.shape[0]))
        pos = pos[:n_nodes]
        sizes = sizes[:n_nodes]

        mids = []
        nx = []
        ny = []
        regions = []
        kept_moves = []

        for region, m in flat:
            mid = _move_get_macro_id(m)
            x, y = _move_get_new_xy(m)
            if mid is None or x is None or y is None:
                continue
            if mid < 0 or mid >= n_nodes:
                continue

            mids.append(mid)
            nx.append(x)
            ny.append(y)
            regions.append(region)
            kept_moves.append(m)

        if not mids:
            raise RuntimeError("could not parse any moves for GPU prefilter")

        mids_t = torch.tensor(mids, device=device, dtype=torch.long)
        nx_t = torch.tensor(nx, device=device, dtype=torch.float32)
        ny_t = torch.tensor(ny, device=device, dtype=torch.float32)

        x_old = pos[mids_t, 0]
        y_old = pos[mids_t, 1]

        w = sizes[mids_t, 0].clamp_min(1e-6)
        h = sizes[mids_t, 1].clamp_min(1e-6)
        area = w * h

        # Infer canvas from current placement extent.
        all_x = pos[:, 0]
        all_y = pos[:, 1]
        min_x = torch.min(all_x)
        max_x = torch.max(all_x)
        min_y = torch.min(all_y)
        max_y = torch.max(all_y)

        span_x = (max_x - min_x).clamp_min(1e-6)
        span_y = (max_y - min_y).clamp_min(1e-6)

        def bin_ids(x, y):
            bx = torch.clamp(((x - min_x) / span_x * bins).long(), 0, bins - 1)
            by = torch.clamp(((y - min_y) / span_y * bins).long(), 0, bins - 1)
            return by * bins + bx

        # Build current coarse density grid.
        base_bin = bin_ids(pos[:, 0], pos[:, 1])
        base_area = (sizes[:, 0].clamp_min(0) * sizes[:, 1].clamp_min(0)).float()
        grid = torch.zeros(bins * bins, device=device, dtype=torch.float32)
        grid.scatter_add_(0, base_bin, base_area)

        old_bin = bin_ids(x_old, y_old)
        new_bin = bin_ids(nx_t, ny_t)

        old_occ = grid[old_bin]
        new_occ = grid[new_bin]

        # Reward moving area out of crowded bin into less crowded bin.
        relief = old_occ - new_occ

        # Mild displacement penalty so massive jumps do not dominate.
        disp = torch.sqrt((nx_t - x_old) ** 2 + (ny_t - y_old) ** 2)
        disp_norm = disp / (span_x + span_y)

        score = relief + 0.05 * area - 0.01 * disp_norm

        k = min(keep_n, int(score.numel()))
        top_idx = torch.topk(score, k=k, largest=True).indices.detach().cpu().tolist()

        out = {}
        for j in top_idx:
            r = regions[j]
            m = kept_moves[j]
            out.setdefault(r, []).append(m)

        print(
            f"[gpu-prefilter] {label}: generated={n} parsed={len(kept_moves)} "
            f"kept={sum(len(v) for v in out.values())} percent={percent} topk={topk} "
            f"device={device} bins={bins}",
            flush=True,
        )

        return out

    except Exception as e:
        print(
            f"[gpu-prefilter] {label}: failed ({e!r}); falling back to first {keep_n}/{n} moves",
            flush=True,
        )

        out = {}
        remaining = keep_n
        for region, moves in region_moves.items():
            if remaining <= 0:
                break
            take = min(len(moves), remaining)
            if take > 0:
                out[region] = moves[:take]
                remaining -= take

        print(
            f"[gpu-prefilter] {label}: fallback kept={sum(len(v) for v in out.values())}/{n}",
            flush=True,
        )
        return out




def keep_top_largest_regions_for_lns(region_moves, region_macro_counts=None, label="layer1"):
    """
    Keep only the top-N largest regions before expensive chunk scoring.

    Intended rule:
      - divide canvas into regions
      - identify largest macro regions
      - keep only top 2 regions
      - then GPU-prefilter top 50% of moves from those regions

    Env:
      FINAL_LNS_TOP_REGION_KEEP=2
      FINAL_LNS_TOP_REGION_BY=macros | moves
    """
    import os

    keep_n = int(os.environ.get("FINAL_LNS_TOP_REGION_KEEP", "0"))
    if keep_n <= 0:
        return region_moves

    mode = os.environ.get("FINAL_LNS_TOP_REGION_BY", "macros").strip().lower()

    region_moves = dict(region_moves)
    region_macro_counts = dict(region_macro_counts or {})

    stats = {}
    for r, moves in region_moves.items():
        stats[r] = {
            "moves": len(moves),
            "macros": int(region_macro_counts.get(r, 0)),
        }

    if not stats:
        print(f"[top-region-filter] {label}: no regions", flush=True)
        return region_moves

    if mode == "moves":
        ranked = sorted(stats.items(), key=lambda kv: kv[1]["moves"], reverse=True)
    else:
        # Default: "largest macro regions" = most soft macros assigned to region.
        # Tie-break by generated moves.
        ranked = sorted(
            stats.items(),
            key=lambda kv: (kv[1]["macros"], kv[1]["moves"]),
            reverse=True,
        )

    kept_regions = [r for r, _ in ranked[:keep_n]]
    kept = {r: region_moves[r] for r in kept_regions if r in region_moves}

    before_moves = sum(v["moves"] for v in stats.values())
    after_moves = sum(len(v) for v in kept.values())

    print(
        f"[top-region-filter] {label}: mode={mode} keep_n={keep_n} "
        f"kept_regions={kept_regions} before_moves={before_moves} after_moves={after_moves} "
        f"stats={stats}",
        flush=True,
    )

    return kept



def run_layer1(bench, benchmark, plc, placement, rows, cols, workers, chunk_size, top_percent, topk_cap, max_rejects, max_accepts, t0):
    region_moves, region_macro_counts = generate_layer1_moves_by_region(benchmark, placement, rows, cols)
    region_moves = keep_top_largest_regions_for_lns(region_moves, region_macro_counts, label="layer1")
    region_moves = gpu_prefilter_region_moves(benchmark, placement, region_moves, "layer1")
    region_counts = {r: len(moves) for r, moves in region_moves.items()}
    total_moves = sum(region_counts.values())

    print(
        f"[layer1-scan] elapsed={elapsed(t0):.2f}s total_moves={total_moves} "
        f"region_moves={region_counts} region_macros={region_macro_counts}",
        flush=True,
    )

    candidates, spec_scored, chunks = parallel_score_candidates(
        bench=bench,
        placement=placement,
        region_moves=region_moves,
        workers=workers,
        chunk_size=chunk_size,
        t0=t0,
        label="layer1",
    )

    print(
        f"[layer1-score-done] elapsed={elapsed(t0):.2f}s spec_scored={spec_scored} chunks={chunks}",
        flush=True,
    )

    return run_exact_replay(
        label="layer1",
        benchmark=benchmark,
        plc=plc,
        placement=placement,
        candidates=candidates,
        top_percent=top_percent,
        topk_cap=topk_cap,
        max_consecutive_rejects=max_rejects,
        max_accepts=max_accepts,
        t0=t0,
    )


def continuation_dirs_for_tag(tag):
    base = tag.split("@")[0].upper()

    around = {
        "U": ["U", "UL", "UR", "D"],
        "D": ["D", "DL", "DR", "U"],
        "L": ["L", "UL", "DL", "R"],
        "R": ["R", "UR", "DR", "L"],
        "UL": ["UL", "U", "L", "DR"],
        "UR": ["UR", "U", "R", "DL"],
        "DL": ["DL", "D", "L", "UR"],
        "DR": ["DR", "D", "R", "UL"],
    }

    return around.get(base, ["R", "L", "U", "D", "UR", "UL", "DR", "DL"])


def generate_targeted_continuation_moves(benchmark, placement, rows, cols, accepted_records, hot_region_count):
    all_dirs = direction_vectors(include_diagonal=True)
    region_counter = Counter(r["region"] for r in accepted_records)
    hot_regions = [r for r, _ in region_counter.most_common(hot_region_count)]

    if not hot_regions:
        hot_regions = list(range(rows * cols))

    accepted_macros = []
    seen = set()
    macro_to_tags = defaultdict(list)

    for r in accepted_records:
        m = r["macro_id"]
        if m not in seen:
            seen.add(m)
            accepted_macros.append(m)
        macro_to_tags[m].append(r["tag"])

    hot_region_macros = []
    for i in soft_macro_ids(benchmark):
        reg = assign_region(benchmark, placement, i, rows, cols)
        if reg in hot_regions:
            hot_region_macros.append(i)

    target_macros = []
    seen = set()

    for m in accepted_macros + hot_region_macros:
        if m not in seen:
            seen.add(m)
            target_macros.append(m)

    region_moves = {r: [] for r in range(rows * cols)}

    # Accepted macros: continuation, opposite correction, and diagonals around prior direction.
    for m in accepted_macros:
        reg = assign_region(benchmark, placement, m, rows, cols)
        direction_names = set()
        for tag in macro_to_tags.get(m, []):
            direction_names.update(continuation_dirs_for_tag(tag))

        dirs = {k: all_dirs[k] for k in direction_names if k in all_dirs}
        region_moves[reg].extend(
            generate_moves_for_macro(
                benchmark=benchmark,
                placement=placement,
                macro_id=m,
                region=reg,
                step_fracs=[0.25, 0.125],
                dirs=dirs,
            )
        )

    # Hot-region macros: diagonal cleanup plus small cardinal correction.
    hot_dirs = {k: all_dirs[k] for k in ["UR", "UL", "DR", "DL", "R", "L", "U", "D"]}
    for m in hot_region_macros:
        reg = assign_region(benchmark, placement, m, rows, cols)
        region_moves[reg].extend(
            generate_moves_for_macro(
                benchmark=benchmark,
                placement=placement,
                macro_id=m,
                region=reg,
                step_fracs=[0.25, 0.125],
                dirs=hot_dirs,
            )
        )

    # Deduplicate moves.
    for r, moves in region_moves.items():
        dedup = {}
        for mv in moves:
            key = (mv.macro_id, round(mv.new_x, 9), round(mv.new_y, 9), mv.tag)
            dedup[key] = mv
        region_moves[r] = list(dedup.values())

    return region_moves, hot_regions, target_macros



def cheap_continuation_score(move, accepted_macro_set, accepted_region_set):
    """
    Lower is better. This is only a prefilter before official compute_proxy_cost.
    It is intentionally biased toward moves most likely to help after Layer 1.
    """
    score = 0.0

    # Keep accepted Layer 1 macros highly prioritized.
    if move.macro_id in accepted_macro_set:
        score -= 10.0

    # Keep hot regions prioritized.
    if move.region in accepted_region_set:
        score -= 3.0

    tag = move.tag.split("@")[0].upper()

    # Diagonal moves are often useful in continuation.
    if tag in {"UR", "UL", "DR", "DL"}:
        score -= 1.0

    # Smaller corrective moves often survive better late.
    if "@0.125" in move.tag:
        score -= 0.5

    # Deterministic tie-breaker so ordering is stable.
    score += 1e-6 * move.macro_id

    return score


def prefilter_region_moves(region_moves, accepted_records, hot_regions, topk, percent):
    accepted_macro_set = {r["macro_id"] for r in accepted_records}
    accepted_region_set = set(hot_regions)

    all_moves = []
    for moves in region_moves.values():
        all_moves.extend(moves)

    full_count = len(all_moves)

    if full_count == 0:
        return region_moves, full_count, 0

    if percent and percent > 0:
        keep_n = int(math.ceil(full_count * percent))
    else:
        keep_n = full_count

    if topk and topk > 0:
        keep_n = min(keep_n, topk)

    keep_n = max(0, min(keep_n, full_count))

    all_moves.sort(key=lambda mv: cheap_continuation_score(mv, accepted_macro_set, accepted_region_set))
    kept = all_moves[:keep_n]

    new_region_moves = {r: [] for r in region_moves.keys()}
    for mv in kept:
        new_region_moves[mv.region].append(mv)

    return new_region_moves, full_count, len(kept)



def run_continuation_layer(bench, benchmark, plc, placement, rows, cols, workers, chunk_size, top_percent, topk_cap, max_rejects, max_accepts, hot_region_count, accepted_records, t0):
    region_moves, hot_regions, target_macros = generate_targeted_continuation_moves(
        benchmark=benchmark,
        placement=placement,
        rows=rows,
        cols=cols,
        accepted_records=accepted_records,
        hot_region_count=hot_region_count,
    )

    pre_region_counts = {r: len(moves) for r, moves in region_moves.items()}
    pre_total_moves = sum(pre_region_counts.values())

    region_moves, prefilter_full, prefilter_kept = prefilter_region_moves(
        region_moves=region_moves,
        accepted_records=accepted_records,
        hot_regions=hot_regions,
        topk=args.cont_prefilter_topk if "args" in globals() else 1500,
        percent=args.cont_prefilter_percent if "args" in globals() else 0.25,
    )

    region_moves = gpu_prefilter_region_moves(benchmark, placement, region_moves, "cont")
    region_counts = {r: len(moves) for r, moves in region_moves.items()}
    total_moves = sum(region_counts.values())

    print(
        f"[cont-start] elapsed={elapsed(t0):.2f}s hot_regions={hot_regions} "
        f"target_macros={len(target_macros)} "
        f"prefilter_full={prefilter_full} prefilter_kept={prefilter_kept} "
        f"pre_region_moves={pre_region_counts} post_region_moves={region_counts}",
        flush=True,
    )

    candidates, spec_scored, chunks = parallel_score_candidates(
        bench=bench,
        placement=placement,
        region_moves=region_moves,
        workers=workers,
        chunk_size=chunk_size,
        t0=t0,
        label="cont",
    )

    print(
        f"[cont-score-done] elapsed={elapsed(t0):.2f}s spec_scored={spec_scored} chunks={chunks}",
        flush=True,
    )

    return run_exact_replay(
        label="cont",
        benchmark=benchmark,
        plc=plc,
        placement=placement,
        candidates=candidates,
        top_percent=top_percent,
        topk_cap=topk_cap,
        max_consecutive_rejects=max_rejects,
        max_accepts=max_accepts,
        t0=t0,
    )



def run_lns(bench, benchmark, plc, placement, root: Path, timeout_deadline: float):
    """
    Reusable LNS entrypoint for submissions/final_placer/placer.py.

    This intentionally does NOT run the DREAMPlace portfolio in this file.
    The outer final_placer already selected/repaired a finalist placement.
    Here we only run:
      finalist placement -> layer1 soft-macro exact replay -> targeted continuation replay

    Returns:
      (best_placement, best_costs)
    """
    import os
    import time
    import types

    t0 = time.time()

    def env_int(name, default):
        try:
            return int(os.environ.get(name, str(default)))
        except Exception:
            return int(default)

    def env_float(name, default):
        try:
            return float(os.environ.get(name, str(default)))
        except Exception:
            return float(default)

    # Conservative defaults for smoke safety. Increase after it works.
    rows = env_int("FINAL_LNS_ROWS", 3)
    cols = env_int("FINAL_LNS_COLS", 4)
    workers = env_int("FINAL_LNS_WORKERS", 12)
    chunk_size = env_int("FINAL_LNS_CHUNK_SIZE", 32)

    layer1_top_percent = env_float("FINAL_LNS_LAYER1_TOP_PERCENT", 0.10)
    layer1_topk = env_int("FINAL_LNS_LAYER1_TOPK", 0)
    layer1_max_rejects = env_int("FINAL_LNS_LAYER1_MAX_REJECTS", 10)
    layer1_max_accepts = env_int("FINAL_LNS_LAYER1_MAX_ACCEPTS", 20)

    cont_top_percent = env_float("FINAL_LNS_CONT_TOP_PERCENT", 0.10)
    cont_topk = env_int("FINAL_LNS_CONT_TOPK", 0)
    cont_max_rejects = env_int("FINAL_LNS_CONT_MAX_REJECTS", 25)
    cont_max_accepts = env_int("FINAL_LNS_CONT_MAX_ACCEPTS", 1000)
    cont_hot_regions = env_int("FINAL_LNS_CONT_HOT_REGIONS", 0)
    cont_hot_region_percent = env_float("FINAL_LNS_CONT_HOT_REGION_PERCENT", 0.33)

    cont_prefilter_topk = env_int("FINAL_LNS_CONT_PREFILTER_TOPK", 1500)
    cont_prefilter_percent = env_float("FINAL_LNS_CONT_PREFILTER_PERCENT", 0.25)

    # run_continuation_layer currently reads global args for prefilter settings.
    # Provide exactly the fields it expects, without calling argparse/main().
    global args
    args = types.SimpleNamespace(
        cont_prefilter_topk=cont_prefilter_topk,
        cont_prefilter_percent=cont_prefilter_percent,
    )

    start_costs = exact_costs(placement, benchmark, plc)
    start_proxy = float(start_costs["proxy_cost"])
    print(
        f"[lns_engine] start bench={bench} proxy={start_proxy:.6f} "
        f"rows={rows} cols={cols} workers={workers} chunk_size={chunk_size}",
        flush=True,
    )

    # Leave some time for final scoring/return.
    now = time.time()
    if now > timeout_deadline - 30:
        print("[lns_engine] not enough time for LNS; returning input placement", flush=True)
        return placement, start_costs

    try:
        layer1_placement, layer1_costs, layer1_accepts = run_layer1(
            bench=bench,
            benchmark=benchmark,
            plc=plc,
            placement=placement.clone(),
            rows=rows,
            cols=cols,
            workers=workers,
            chunk_size=chunk_size,
            top_percent=layer1_top_percent,
            topk_cap=layer1_topk,
            max_rejects=layer1_max_rejects,
            max_accepts=layer1_max_accepts,
            t0=t0,
        )

        layer1_proxy = float(layer1_costs["proxy_cost"])
        print(
            f"[lns_engine] after layer1 proxy={layer1_proxy:.6f} "
            f"accepted={len(layer1_accepts) if layer1_accepts is not None else 0}",
            flush=True,
        )

        # ------------------------------------------------------------
        # Optional LNS Layer 2 / continuation.
        # Disabled by default because we want to spend runtime on
        # DREAMPlace L2 instead of continuation LNS.
        # ------------------------------------------------------------
        if os.environ.get("FINAL_LNS_RUN_CONT", "0") != "1":
            best_placement, best_costs = placement, start_costs
            best_proxy = start_proxy

            if layer1_proxy < best_proxy:
                best_placement, best_costs = layer1_placement, layer1_costs
                best_proxy = layer1_proxy

            print(
                f"[lns_engine] continuation disabled; selected proxy={best_proxy:.6f}",
                flush=True,
            )
            return best_placement, best_costs

        if time.time() > timeout_deadline - 30:
            print("[lns_engine] timeout margin after layer1; returning layer1 placement", flush=True)
            return layer1_placement, layer1_costs

        hot_region_count = (
            cont_hot_regions
            if cont_hot_regions and cont_hot_regions > 0
            else max(1, int((rows * cols * cont_hot_region_percent) + 0.5))
        )

        cont_placement, cont_costs, cont_accepts = run_continuation_layer(
            bench=bench,
            benchmark=benchmark,
            plc=plc,
            placement=layer1_placement.clone(),
            rows=rows,
            cols=cols,
            workers=workers,
            chunk_size=chunk_size,
            top_percent=cont_top_percent,
            topk_cap=cont_topk,
            max_rejects=cont_max_rejects,
            max_accepts=cont_max_accepts,
            hot_region_count=hot_region_count,
            accepted_records=layer1_accepts,
            t0=t0,
        )

        cont_proxy = float(cont_costs["proxy_cost"])
        print(
            f"[lns_engine] after continuation proxy={cont_proxy:.6f} "
            f"accepted={len(cont_accepts) if cont_accepts is not None else 0}",
            flush=True,
        )

        # Safety: never return a worse placement from LNS.
        best_placement, best_costs = placement, start_costs
        best_proxy = start_proxy

        if layer1_proxy < best_proxy:
            best_placement, best_costs = layer1_placement, layer1_costs
            best_proxy = layer1_proxy

        if cont_proxy < best_proxy:
            best_placement, best_costs = cont_placement, cont_costs
            best_proxy = cont_proxy

        print(f"[lns_engine] selected proxy={best_proxy:.6f}", flush=True)
        return best_placement, best_costs

    except Exception:
        import traceback
        print("[lns_engine] LNS failed; returning input placement", flush=True)
        traceback.print_exc()
        return placement, start_costs



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", default="ibm01")
    parser.add_argument("--seeds", default="999,1000,1001")
    parser.add_argument("--target-densities", default="0.70,0.80,0.85,0.90")
    parser.add_argument("--density-weights", default="8e-5,1.2e-4,2e-4,3e-4,4e-4,6e-4,8e-4,1.2e-3")
    parser.add_argument("--max-configs", type=int, default=0)

    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=32)

    parser.add_argument("--layer1-top-percent", type=float, default=0.10)
    parser.add_argument("--layer1-topk", type=int, default=0)
    parser.add_argument("--layer1-max-rejects", type=int, default=25)
    parser.add_argument("--layer1-max-accepts", type=int, default=1000)

    parser.add_argument("--cont-top-percent", type=float, default=0.10)
    parser.add_argument("--cont-topk", type=int, default=0)
    parser.add_argument("--cont-max-rejects", type=int, default=25)
    parser.add_argument("--cont-max-accepts", type=int, default=1000)
    parser.add_argument("--cont-hot-regions", type=int, default=0)
    parser.add_argument("--cont-hot-region-percent", type=float, default=0.33)
    parser.add_argument("--cont-prefilter-topk", type=int, default=1500)
    parser.add_argument("--cont-prefilter-percent", type=float, default=0.25)

    parser.add_argument("--timeout-sec", type=float, default=3300)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    t0 = time.time()

    root = Path("external/MacroPlacement/Testcases/ICCAD04")
    benchmark, plc = load_benchmark_from_dir(root / args.bench)

    configs = [
        DPConfig(seed=s, target_density=td, density_weight=dw)
        for s, td, dw in itertools.product(
            parse_csv(args.seeds),
            parse_csv(args.target_densities),
            parse_csv(args.density_weights),
        )
    ]

    if args.max_configs and args.max_configs > 0:
        configs = configs[: args.max_configs]

    print(f"[portfolio-start] bench={args.bench} configs={len(configs)}", flush=True)

    MyPlacer = load_myplacer()
    best_key = None
    best_cfg = None
    best_placement = None
    best_costs = None

    for idx, cfg in enumerate(configs, start=1):
        if elapsed(t0) > args.timeout_sec:
            print(f"[portfolio-stop] timeout before config {idx}", flush=True)
            break

        placement, costs, proxy, overlap_count, dt = run_dreamplace_config(args.bench, benchmark, plc, MyPlacer, cfg)
        key = (overlap_count > 0, proxy)

        print(
            f"[portfolio] elapsed={elapsed(t0):.2f}s idx={idx}/{len(configs)} "
            f"seed={cfg.seed} target_density={cfg.target_density} density_weight={cfg.density_weight} "
            f"proxy={proxy:.6f} overlaps={overlap_count} runtime={dt:.2f}s",
            flush=True,
        )

        if best_key is None or key < best_key:
            best_key = key
            best_cfg = cfg
            best_placement = placement.clone()
            best_costs = costs
            print(
                f"[portfolio-best] idx={idx} seed={cfg.seed} target_density={cfg.target_density} "
                f"density_weight={cfg.density_weight} proxy={proxy:.6f}",
                flush=True,
            )

    if best_placement is None:
        raise RuntimeError("No portfolio placement produced.")

    print(
        f"[portfolio-done] elapsed={elapsed(t0):.2f}s best_seed={best_cfg.seed} "
        f"best_target_density={best_cfg.target_density} best_density_weight={best_cfg.density_weight} "
        f"best_costs={best_costs}",
        flush=True,
    )

    layer1_placement, layer1_costs, layer1_accepts = run_layer1(
        bench=args.bench,
        benchmark=benchmark,
        plc=plc,
        placement=best_placement.clone(),
        rows=args.rows,
        cols=args.cols,
        workers=args.workers,
        chunk_size=args.chunk_size,
        top_percent=args.layer1_top_percent,
        topk_cap=args.layer1_topk,
        max_rejects=args.layer1_max_rejects,
        max_accepts=args.layer1_max_accepts,
        t0=t0,
    )

    cont_placement, cont_costs, cont_accepts = run_continuation_layer(
        bench=args.bench,
        benchmark=benchmark,
        plc=plc,
        placement=layer1_placement.clone(),
        rows=args.rows,
        cols=args.cols,
        workers=args.workers,
        chunk_size=args.chunk_size,
        top_percent=args.cont_top_percent,
        topk_cap=args.cont_topk,
        max_rejects=args.cont_max_rejects,
        max_accepts=args.cont_max_accepts,
        hot_region_count=(
            args.cont_hot_regions
            if args.cont_hot_regions and args.cont_hot_regions > 0
            else max(1, int((args.rows * args.cols * args.cont_hot_region_percent) + 0.5))
        ),
        accepted_records=layer1_accepts,
        t0=t0,
    )

    final_proxy = float(cont_costs["proxy_cost"])

    print(f"[final] elapsed={elapsed(t0):.2f}s proxy={final_proxy:.6f} costs={cont_costs}", flush=True)

    if args.out:
        torch.save(cont_placement, args.out)
        print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()


# =============================================================================
# EXACT OLD XPLACE-STYLE LNS LOGIC, REBUILT FOR DREAMPLACE-ONLY FINAL_PLACER
# =============================================================================
#
# This intentionally overrides the broader LNS move generators above.
#
# Approved old structure:
#   Layer 1:
#     soft macros only
#     rows x cols spatial regions, normally 3 x 4
#     step = 0.25
#     dirs = R, L, U, D
#     total moves ~= soft_macro_count * 4
#
#   Continuation:
#     driven by accepted Layer 1 records
#     hot regions only
#     step = 0.125
#     continuation/correction dirs around accepted move direction
#     cheap prefilter before exact scoring/replay
# =============================================================================


def _old_exact_cardinal_dirs():
    return {
        "R": (1.0, 0.0),
        "L": (-1.0, 0.0),
        "U": (0.0, 1.0),
        "D": (0.0, -1.0),
    }


def _old_exact_cont_dirs(tag):
    base = str(tag).split("@")[0].upper()

    vecs = {
        "R": (1.0, 0.0),
        "L": (-1.0, 0.0),
        "U": (0.0, 1.0),
        "D": (0.0, -1.0),
        "UR": (1.0, 1.0),
        "UL": (-1.0, 1.0),
        "DR": (1.0, -1.0),
        "DL": (-1.0, -1.0),
    }

    table = {
        "R": ["R", "UR", "DR", "L"],
        "L": ["L", "UL", "DL", "R"],
        "U": ["U", "UL", "UR", "D"],
        "D": ["D", "DL", "DR", "U"],
        "UR": ["U", "R", "UR", "DL"],
        "UL": ["U", "L", "UL", "DR"],
        "DR": ["D", "R", "DR", "UL"],
        "DL": ["D", "L", "DL", "UR"],
    }

    tags = table.get(base, ["R", "L", "U", "D"])
    return {t: vecs[t] for t in tags if t in vecs}


def generate_layer1_moves_by_region(benchmark, placement, rows, cols):
    """
    Exact old Layer 1:
      one move radius: 0.25
      four cardinal dirs
      soft macros only
      grouped into rows*cols regions

    This is the key fix. It prevents the new broad LNS from generating
    a huge uncontrolled move pool.
    """
    from collections import defaultdict

    region_moves = defaultdict(list)
    region_macro_counts = defaultdict(int)

    mids = soft_macro_ids(benchmark)
    dirs = _old_exact_cardinal_dirs()
    step_fracs = [0.25]

    for macro_id in mids:
        region = assign_region(benchmark, placement, macro_id, rows, cols)
        region_macro_counts[region] += 1

        moves = generate_moves_for_macro(
            benchmark=benchmark,
            placement=placement,
            macro_id=macro_id,
            region=region,
            step_fracs=step_fracs,
            dirs=dirs,
        )

        if moves:
            region_moves[region].extend(moves)

    total_moves = sum(len(v) for v in region_moves.values())
    print(
        f"[old-exact-layer1-generate] soft_macros={len(mids)} "
        f"dirs={list(dirs.keys())} step_fracs={step_fracs} total_moves={total_moves} "
        f"expected_max={len(mids) * 4}",
        flush=True,
    )

    return dict(region_moves), dict(region_macro_counts)


def _old_exact_get_record_macro_id(rec):
    for name in ("macro_id", "node_id", "idx", "i"):
        if hasattr(rec, name):
            try:
                return int(getattr(rec, name))
            except Exception:
                pass
        if isinstance(rec, dict) and name in rec:
            try:
                return int(rec[name])
            except Exception:
                pass
    if isinstance(rec, (list, tuple)) and len(rec) > 0:
        try:
            return int(rec[0])
        except Exception:
            pass
    return None


def _old_exact_get_record_region(rec, benchmark=None, placement=None, rows=None, cols=None):
    for name in ("region", "region_id", "r"):
        if hasattr(rec, name):
            try:
                return int(getattr(rec, name))
            except Exception:
                pass
        if isinstance(rec, dict) and name in rec:
            try:
                return int(rec[name])
            except Exception:
                pass

    mid = _old_exact_get_record_macro_id(rec)
    if mid is not None and benchmark is not None and placement is not None:
        try:
            return assign_region(benchmark, placement, mid, rows, cols)
        except Exception:
            pass

    return None


def _old_exact_get_record_tag(rec):
    for name in ("tag", "direction", "dir"):
        if hasattr(rec, name):
            return str(getattr(rec, name))
        if isinstance(rec, dict) and name in rec:
            return str(rec[name])

    if isinstance(rec, (list, tuple)) and len(rec) >= 6:
        return str(rec[5])

    return "R"


def generate_targeted_continuation_moves(
    benchmark,
    placement,
    rows,
    cols,
    accepted_records,
    hot_region_count,
):
    """
    Exact old continuation:
      use accepted Layer 1 records
      choose hot regions
      target macros from accepted records in those hot regions
      step = 0.125
      continuation/correction dirs from accepted move direction
    """
    from collections import Counter, defaultdict

    if not accepted_records:
        print("[old-exact-cont-generate] no accepted layer1 records", flush=True)
        return {}, [], set()

    region_counter = Counter()
    macro_to_tags = defaultdict(list)
    macro_to_region = {}

    for rec in accepted_records:
        mid = _old_exact_get_record_macro_id(rec)
        if mid is None:
            continue

        region = _old_exact_get_record_region(
            rec,
            benchmark=benchmark,
            placement=placement,
            rows=rows,
            cols=cols,
        )

        if region is None:
            continue

        tag = _old_exact_get_record_tag(rec)

        region_counter[region] += 1
        macro_to_region[mid] = region
        macro_to_tags[mid].append(tag)

    if not region_counter:
        print("[old-exact-cont-generate] no hot regions found", flush=True)
        return {}, [], set()

    hot_regions = [
        r for r, _ in region_counter.most_common(
            max(1, min(int(hot_region_count), len(region_counter)))
        )
    ]

    hot_set = set(hot_regions)
    target_macros = {
        mid for mid, r in macro_to_region.items()
        if r in hot_set
    }

    region_moves = defaultdict(list)
    step_fracs = [0.125]

    for mid in sorted(target_macros):
        region = macro_to_region.get(mid)
        tags = macro_to_tags.get(mid) or ["R"]

        # Keep unique continuation dirs but preserve order.
        dirs = []
        seen = set()
        for tag in tags:
            for d in _old_exact_cont_dirs(tag):
                if d not in seen:
                    seen.add(d)
                    dirs.append(d)

        moves = generate_moves_for_macro(
            benchmark=benchmark,
            placement=placement,
            macro_id=mid,
            region=region,
            step_fracs=step_fracs,
            dirs=dirs,
        )

        if moves:
            region_moves[region].extend(moves)

    full_count = sum(len(v) for v in region_moves.values())

    print(
        f"[old-exact-cont-generate] hot_regions={hot_regions} "
        f"target_macros={len(target_macros)} step_fracs={step_fracs} "
        f"full_moves={full_count}",
        flush=True,
    )

    return dict(region_moves), hot_regions, target_macros


def prefilter_region_moves(region_moves, accepted_records, hot_regions, topk, percent):
    """
    Exact old continuation prefilter:
      keep top percent and cap at topk.
    This runs before expensive candidate scoring.
    """
    all_moves = []
    for region, moves in region_moves.items():
        for m in moves:
            all_moves.append((region, m))

    full_count = len(all_moves)

    if full_count == 0:
        return {}, 0, 0

    # Use cheap_continuation_score if available. If anything fails,
    # preserve original order.
    try:
        accepted_macro_set = set()
        accepted_region_set = set(hot_regions or [])

        for rec in accepted_records or []:
            mid = _old_exact_get_record_macro_id(rec)
            if mid is not None:
                accepted_macro_set.add(mid)

        all_moves.sort(
            key=lambda rm: cheap_continuation_score(
                rm[1],
                accepted_macro_set,
                accepted_region_set,
            ),
            reverse=True,
        )
    except Exception as e:
        print(f"[old-exact-cont-prefilter] cheap sort failed: {e!r}; preserving order", flush=True)

    keep_n = full_count

    if percent and percent > 0:
        import math
        keep_n = min(keep_n, max(1, int(math.ceil(full_count * float(percent)))))

    if topk and topk > 0:
        keep_n = min(keep_n, int(topk))

    kept_pairs = all_moves[:keep_n]

    kept = {}
    for region, move in kept_pairs:
        kept.setdefault(region, []).append(move)

    print(
        f"[old-exact-cont-prefilter] full={full_count} kept={keep_n} "
        f"topk={topk} percent={percent}",
        flush=True,
    )

    return kept, full_count, keep_n


print("[old-exact-lns] DREAMPlace-only exact old XPlace-style LNS overrides loaded", flush=True)
