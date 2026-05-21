import json
import shutil
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import torch


SCALE = 1000


def _default_challenge_root() -> Path:
    env = os.environ.get("CHALLENGE_ROOT")
    if env:
        return Path(env).resolve()

    # placer.py is expected at submissions/dreamplace_only/placer.py
    return Path(__file__).resolve().parents[2]


def _default_dreamplace_root() -> Path:
    env = os.environ.get("DREAMPLACE_ROOT")
    if env:
        return Path(env).resolve()

    # Local RunPod development path.
    local = Path(os.environ.get("DREAMPLACE_ROOT", "/workspace/DREAMPlace/install"))
    if local.exists():
        return local

    # Docker submission path.
    return Path("/opt/DREAMPlace/install")


def _default_dreamplace_python() -> Path:
    env = os.environ.get("DREAMPLACE_PYTHON")
    if env:
        return Path(env).resolve()

    # Local RunPod development path.
    local = Path(os.environ.get("DREAMPLACE_PYTHON", "/workspace/dreamplace_env/bin/python"))
    if local.exists():
        return local

    # Docker submission path.
    return Path("/opt/dreamplace_env/bin/python")


CHALLENGE_ROOT = _default_challenge_root()
DREAMPLACE_ROOT = _default_dreamplace_root()
DREAMPLACE_PYTHON = Path(os.environ.get("DREAMPLACE_PYTHON", "/workspace/dreamplace_env/bin/python"))
ICCAD_ROOT = Path(os.environ.get(
    "ICCAD_ROOT",
    str(CHALLENGE_ROOT / "external/MacroPlacement/Testcases/ICCAD04"),
)).resolve()
OUT_ROOT = Path(os.environ.get(
    "DREAMPLACE_OUT_ROOT",
    str(CHALLENGE_ROOT / "dreamplace_ibm"),
)).resolve()



GLOBAL_DREAMPLACE_CONFIG = {
    # Global reproducibility/runtime defaults.
    "DREAMPLACE_RANDOM_SEED": "1000",
    "DREAMPLACE_DETERMINISTIC": "1",

    # Global soft-macro default.
    "SOFT_MODE": "off",

    # Make sure density_weight is active in generated DREAMPlace JSON.
    "DREAMPLACE_OMIT_DENSITY_WEIGHT": "0",
}


FEATURE_DENSITY_PROFILES = {
    "small_macro_count": {
        # Applies to designs with fewer hard macros.
        # Learned from public benchmarks as a broad feature bucket.
        "max_hard_macros_exclusive": 365,
        "DREAMPLACE_TARGET_DENSITY": "0.85",
        "DREAMPLACE_DENSITY_WEIGHT": "1.2e-4",
    },
    "large_macro_count": {
        # Applies to larger hard-macro designs.
        "DREAMPLACE_TARGET_DENSITY": "0.80",
        "DREAMPLACE_DENSITY_WEIGHT": "2e-4",
    },
}


def apply_benchmark_env_config(bench: str, benchmark=None):
    """
    Feature-based DREAMPlace config selector.

    This intentionally avoids benchmark-name lookup. It uses a broad
    hard-macro-count threshold to select one of two density profiles.
    """
    allow_env_override = os.environ.get("EXPLICIT_ENV_OVERRIDE", "0") == "1"

    for key, value in GLOBAL_DREAMPLACE_CONFIG.items():
        if allow_env_override and key in os.environ:
            continue
        os.environ[key] = str(value)

    if benchmark is None:
        profile_name = "large_macro_count"
        profile = FEATURE_DENSITY_PROFILES[profile_name]
        hard_count = -1
    else:
        hard_count = int(torch.sum(benchmark.get_hard_macro_mask()).item())
        small_limit = int(FEATURE_DENSITY_PROFILES["small_macro_count"]["max_hard_macros_exclusive"])

        if hard_count < small_limit:
            profile_name = "small_macro_count"
            profile = FEATURE_DENSITY_PROFILES["small_macro_count"]
        else:
            profile_name = "large_macro_count"
            profile = FEATURE_DENSITY_PROFILES["large_macro_count"]

    for key, value in profile.items():
        if key == "max_hard_macros_exclusive":
            continue
        if allow_env_override and key in os.environ:
            continue
        os.environ[key] = str(value)

    print(
        f"[config] feature profile={profile_name}, hard_macros={hard_count}, "
        f"target_density={os.environ.get('DREAMPLACE_TARGET_DENSITY')}, "
        f"density_weight={os.environ.get('DREAMPLACE_DENSITY_WEIGHT')}, "
        f"soft_mode={os.environ.get('SOFT_MODE')}",
        flush=True,
    )

def safe_name(name: str) -> str:
    return name.replace("/", "__")


def infer_benchmark_name(benchmark=None) -> str:
    # Try benchmark object attributes first.
    if benchmark is not None:
        for attr in ["name", "benchmark_name", "bench_name", "id", "path", "root"]:
            if hasattr(benchmark, attr):
                value = str(getattr(benchmark, attr))
                m = re.search(r"ibm\d{2}", value)
                if m:
                    return m.group(0)

    # Fallback: scan evaluator command-line args.
    joined = " ".join(sys.argv)
    m = re.search(r"ibm\d{2}", joined)
    if m:
        return m.group(0)

    # Optional override.
    env_name = os.environ.get("DREAMPLACE_BENCH")
    if env_name:
        return env_name

    raise RuntimeError(
        "Could not infer benchmark name. Try running with DREAMPLACE_BENCH=ibm01."
    )


def split_pb_nodes(text: str):
    blocks = []
    lines = text.splitlines()
    in_node = False
    depth = 0
    cur = []

    for line in lines:
        stripped = line.strip()

        if not in_node and stripped == "node {":
            in_node = True
            depth = 1
            cur = [line]
            continue

        if in_node:
            cur.append(line)
            depth += line.count("{")
            depth -= line.count("}")
            if depth == 0:
                blocks.append("\n".join(cur))
                in_node = False
                cur = []

    return blocks


def parse_attrs(block: str):
    attrs = {}
    for am in re.finditer(
        r'attr\s*\{\s*key:\s*"([^"]+)"\s*value\s*\{\s*(?:f:\s*([0-9eE+\-.]+)|placeholder:\s*"([^"]+)")',
        block,
        flags=re.S,
    ):
        key = am.group(1)
        val = am.group(2) if am.group(2) is not None else am.group(3)
        attrs[key] = val
    return attrs


def parse_netlist(pb_path: Path):
    text = pb_path.read_text(errors="ignore")
    blocks = split_pb_nodes(text)

    nodes = []
    for idx, block in enumerate(blocks):
        name_m = re.search(r'name:\s*"([^"]+)"', block)
        if not name_m:
            continue

        name = name_m.group(1)
        inputs = re.findall(r'input:\s*"([^"]+)"', block)
        attrs = parse_attrs(block)

        nodes.append(
            {
                "idx": idx,
                "name": name,
                "inputs": inputs,
                "attrs": attrs,
            }
        )

    return nodes


def parse_plc(plc_path: Path):
    width = height = None
    positions = {}

    for line in plc_path.read_text(errors="ignore").splitlines():
        if line.startswith("# Width"):
            m = re.search(r"Width\s*:\s*([0-9.]+)\s+Height\s*:\s*([0-9.]+)", line)
            if m:
                width = float(m.group(1))
                height = float(m.group(2))

        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) >= 5 and parts[0].isdigit():
            idx = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            orient = parts[3]
            fixed = int(parts[4])
            positions[idx] = (x, y, orient, fixed)

    if width is None or height is None:
        raise RuntimeError(f"Could not parse canvas size from {plc_path}")

    return width, height, positions


def endpoint_for(name, macros, ports, macro_pins):
    if name in macro_pins:
        pin = macro_pins[name]
        parent = pin["macro_name"]
        if parent in macros:
            return parent, float(pin.get("x_offset", 0.0)), float(pin.get("y_offset", 0.0))

    if name in macros:
        return name, 0.0, 0.0

    if name in ports:
        return name, 0.0, 0.0

    return None


def export_bookshelf(bench: str):
    src = ICCAD_ROOT / bench
    pb = src / "netlist.pb.txt"
    plc = src / "initial.plc"

    if not pb.exists():
        raise FileNotFoundError(pb)
    if not plc.exists():
        raise FileNotFoundError(plc)

    out_dir = OUT_ROOT / bench
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes = parse_netlist(pb)
    plc_w, plc_h, plc_positions = parse_plc(plc)

    macros = {}
    ports = {}
    macro_pins = {}

    for n in nodes:
        attrs = n["attrs"]
        typ = attrs.get("type")

        if typ == "MACRO":
            name = n["name"]
            macros[name] = {
                "idx": n["idx"],
                "width": float(attrs.get("width", 0.01)),
                "height": float(attrs.get("height", 0.01)),
                "x": float(attrs.get("x", 0.0)),
                "y": float(attrs.get("y", 0.0)),
                "orientation": attrs.get("orientation", "N"),
            }

        elif typ == "PORT":
            name = n["name"]
            ports[name] = {
                "idx": n["idx"],
                "x": float(attrs.get("x", 0.0)),
                "y": float(attrs.get("y", 0.0)),
                "side": attrs.get("side", ""),
            }

        elif typ == "MACRO_PIN":
            name = n["name"]
            macro_pins[name] = {
                "idx": n["idx"],
                "macro_name": attrs.get("macro_name", ""),
                "x": float(attrs.get("x", 0.0)),
                "y": float(attrs.get("y", 0.0)),
                "x_offset": float(attrs.get("x_offset", 0.0)),
                "y_offset": float(attrs.get("y_offset", 0.0)),
            }

    nets = []
    for n in nodes:
        if not n["inputs"]:
            continue

        endpoints = []

        cur = endpoint_for(n["name"], macros, ports, macro_pins)
        if cur is not None:
            endpoints.append(cur)

        for inp in n["inputs"]:
            ep = endpoint_for(inp, macros, ports, macro_pins)
            if ep is not None:
                endpoints.append(ep)

        dedup = []
        seen = set()
        for ep in endpoints:
            key = (ep[0], round(ep[1], 6), round(ep[2], 6))
            if key not in seen:
                seen.add(key)
                dedup.append(ep)

        if len(dedup) >= 2:
            nets.append(dedup)

    macro_names = list(macros.keys())

    # .aux
    (out_dir / f"{bench}.aux").write_text(
        f"RowBasedPlacement : {bench}.nodes {bench}.nets {bench}.wts {bench}.pl {bench}.scl\n"
    )

    # .nodes
    with (out_dir / f"{bench}.nodes").open("w") as f:
        f.write("UCLA nodes 1.0\n\n")
        f.write(f"NumNodes : {len(macros) + len(ports)}\n")
        f.write(f"NumTerminals : {len(ports)}\n\n")

        for name, m in macros.items():
            f.write(
                f"{safe_name(name)} {int(round(m['width'] * SCALE))} "
                f"{int(round(m['height'] * SCALE))}\n"
            )

        for name in ports:
            f.write(f"{safe_name(name)} 1 1 terminal\n")

    # .pl
    with (out_dir / f"{bench}.pl").open("w") as f:
        f.write("UCLA pl 1.0\n\n")

        for name, m in macros.items():
            cx = m["x"]
            cy = m["y"]
            if m["idx"] in plc_positions:
                cx, cy, orient, fixed = plc_positions[m["idx"]]

            x_ll = cx - m["width"] / 2.0
            y_ll = cy - m["height"] / 2.0
            f.write(f"{safe_name(name)} {int(round(x_ll * SCALE))} {int(round(y_ll * SCALE))} : N\n")

        for name, p in ports.items():
            cx = p["x"]
            cy = p["y"]
            if p["idx"] in plc_positions:
                cx, cy, orient, fixed = plc_positions[p["idx"]]
            f.write(f"{safe_name(name)} {int(round(cx * SCALE))} {int(round(cy * SCALE))} : N /FIXED\n")

    # .scl
    num_rows = 41
    num_sites = 45
    scaled_w = int(round(plc_w * SCALE))
    scaled_h = int(round(plc_h * SCALE))
    row_height = max(1, scaled_h // num_rows)
    site_width = max(1, scaled_w // num_sites)

    with (out_dir / f"{bench}.scl").open("w") as f:
        f.write("UCLA scl 1.0\n\n")
        f.write(f"NumRows : {num_rows}\n\n")
        for r in range(num_rows):
            y = r * row_height
            f.write("CoreRow Horizontal\n")
            f.write(f"  Coordinate    : {y}\n")
            f.write(f"  Height        : {row_height}\n")
            f.write(f"  Sitewidth     : {site_width}\n")
            f.write(f"  Sitespacing   : {site_width}\n")
            f.write("  Siteorient    : N\n")
            f.write("  Sitesymmetry  : Y\n")
            f.write(f"  SubrowOrigin  : 0 NumSites : {num_sites}\n")
            f.write("End\n\n")

    # .wts
    (out_dir / f"{bench}.wts").write_text("UCLA wts 1.0\n\n")

    # .nets
    with (out_dir / f"{bench}.nets").open("w") as f:
        pins = sum(len(n) for n in nets)
        f.write("UCLA nets 1.0\n\n")
        f.write(f"NumNets : {len(nets)}\n")
        f.write(f"NumPins : {pins}\n\n")

        for i, net in enumerate(nets):
            f.write(f"NetDegree : {len(net)} n{i}\n")
            for obj, xoff, yoff in net:
                f.write(
                    f"  {safe_name(obj)} I : "
                    f"{int(round(xoff * SCALE))} {int(round(yoff * SCALE))}\n"
                )

    cfg = {
        "aux_input": str((out_dir / f"{bench}.aux").resolve()),
        "gpu": 1,
        "num_bins_x": 512,
        "num_bins_y": 512,
        "global_place_stages": [
            {
                "num_bins_x": 512,
                "num_bins_y": 512,
                "iteration": 1000,
                "learning_rate": 0.01,
                "wirelength": "weighted_average",
                "optimizer": "nesterov",
                "Llambda_density_weight_iteration": 1,
                "Lsub_iteration": 1,
            }
        ],
        "target_density": 1.0,
        "density_weight": 8e-5,
        "random_seed": 1000,
        "result_dir": str((out_dir / "results").resolve()),
        "scale_factor": 1.0,
        "ignore_net_degree": 100,
        "enable_fillers": 1,
        "global_place_flag": 1,
        "legalize_flag": int(os.environ.get("DREAMPLACE_LEGALIZE_FLAG", "0")),
        "abacus_legalize_flag": int(os.environ.get("DREAMPLACE_ABACUS_LEGALIZE_FLAG", "0")),
        "detailed_place_flag": int(os.environ.get("DREAMPLACE_DETAILED_PLACE_FLAG", "0")),
        "plot_flag": 0,
        "num_threads": 8,
    }

    (out_dir / f"{bench}.json").write_text(json.dumps(cfg, indent=2))

    meta = {
        "bench": bench,
        "scale": SCALE,
        "macro_names": macro_names,
        "macros": macros,
        "plc_width": plc_w,
        "plc_height": plc_h,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    return out_dir, meta


def run_dreamplace(bench: str, out_dir: Path):
    gp_pl = out_dir / "results" / bench / f"{bench}.gp.pl"
    force = os.environ.get("DREAMPLACE_FORCE", "0") == "1"

    if gp_pl.exists() and not force:
        return gp_pl

    cmd = [
        str(DREAMPLACE_PYTHON),
        str(DREAMPLACE_ROOT / "dreamplace/Placer.py"),
        str(out_dir / f"{bench}.json"),
    ]

    log_path = out_dir / "run_from_evaluator.log"
    timeout_sec = float(os.environ.get("DREAMPLACE_TIMEOUT_SEC", "3300"))

    t0 = time.time()
    with log_path.open("w") as log:
        try:
            subprocess.run(
                cmd,
                cwd=str(DREAMPLACE_ROOT),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            dt = time.time() - t0
            print(
                f"[timeout] DREAMPlace timed out after {dt:.1f}s for {bench}; "
                f"timeout={timeout_sec}s",
                flush=True,
            )

            if gp_pl.exists():
                print(
                    f"[timeout] using partially written DREAMPlace placement: {gp_pl}",
                    flush=True,
                )
                return gp_pl

            print(
                f"[timeout] no gp.pl produced for {bench}; using fallback placement",
                flush=True,
            )
            return None

        except subprocess.CalledProcessError:
            dt = time.time() - t0
            print(
                f"[error] DREAMPlace failed after {dt:.1f}s for {bench}. "
                f"See log: {log_path}",
                flush=True,
            )

            if gp_pl.exists():
                print(
                    f"[error] using existing DREAMPlace placement despite failure: {gp_pl}",
                    flush=True,
                )
                return gp_pl

            print(
                f"[error] no gp.pl produced for {bench}; using fallback placement",
                flush=True,
            )
            return None

    if not gp_pl.exists():
        raise RuntimeError(f"DREAMPlace finished but did not create {gp_pl}")

    return gp_pl


def read_gp_pl(path: Path):
    positions = {}

    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("UCLA"):
            continue

        parts = line.split()
        if len(parts) < 3:
            continue

        name = parts[0]
        try:
            x = float(parts[1])
            y = float(parts[2])
        except ValueError:
            continue

        positions[name] = (x, y)

    return positions


def export_bookshelf_via_tool(bench: str):
    out_dir = OUT_ROOT / bench

    subprocess.run(
        [
            sys.executable,
            str(CHALLENGE_ROOT / "tools/pb_plc_to_bookshelf.py"),
            "--bench",
            bench,
        ],
        cwd=str(CHALLENGE_ROOT),
        check=True,
    )

    meta_path = out_dir / "meta.json"
    if not meta_path.exists():
        raise RuntimeError(f"Converter did not create {meta_path}")

    # Optional experiments: remove fields from DREAMPlace JSON
    # after conversion/config selection has already written them.
    json_path = out_dir / f"{bench}.json"
    if json_path.exists():
        cfg = json.loads(json_path.read_text())
        changed = False

        # Optional expanded DREAMPlace knob overrides for Optuna basin search.
        try:
            if "DREAMPLACE_NUM_BINS_X" in os.environ:
                cfg["num_bins_x"] = int(os.environ["DREAMPLACE_NUM_BINS_X"])
                cfg["global_place_stages"][0]["num_bins_x"] = int(os.environ["DREAMPLACE_NUM_BINS_X"])

            if "DREAMPLACE_NUM_BINS_Y" in os.environ:
                cfg["num_bins_y"] = int(os.environ["DREAMPLACE_NUM_BINS_Y"])
                cfg["global_place_stages"][0]["num_bins_y"] = int(os.environ["DREAMPLACE_NUM_BINS_Y"])

            if "DREAMPLACE_GP_ITERATIONS" in os.environ:
                cfg["global_place_stages"][0]["iteration"] = int(os.environ["DREAMPLACE_GP_ITERATIONS"])

            if "DREAMPLACE_LEARNING_RATE" in os.environ:
                cfg["global_place_stages"][0]["learning_rate"] = float(os.environ["DREAMPLACE_LEARNING_RATE"])

            if "DREAMPLACE_LEARNING_RATE_DECAY" in os.environ:
                cfg["global_place_stages"][0]["learning_rate_decay"] = float(os.environ["DREAMPLACE_LEARNING_RATE_DECAY"])

            if "DREAMPLACE_WIRELENGTH" in os.environ:
                cfg["global_place_stages"][0]["wirelength"] = str(os.environ["DREAMPLACE_WIRELENGTH"])

            if "DREAMPLACE_OPTIMIZER" in os.environ:
                cfg["global_place_stages"][0]["optimizer"] = str(os.environ["DREAMPLACE_OPTIMIZER"])

            if "DREAMPLACE_GAMMA" in os.environ:
                cfg["gamma"] = float(os.environ["DREAMPLACE_GAMMA"])

            if "DREAMPLACE_STOP_OVERFLOW" in os.environ:
                cfg["stop_overflow"] = float(os.environ["DREAMPLACE_STOP_OVERFLOW"])

            if "DREAMPLACE_LLAMBDA_ITER" in os.environ:
                cfg["global_place_stages"][0]["Llambda_density_weight_iteration"] = int(os.environ["DREAMPLACE_LLAMBDA_ITER"])

            if "DREAMPLACE_LSUB_ITER" in os.environ:
                cfg["global_place_stages"][0]["Lsub_iteration"] = int(os.environ["DREAMPLACE_LSUB_ITER"])

            print(
                "[config-extra] "
                f"bins=({cfg.get('num_bins_x')},{cfg.get('num_bins_y')}) "
                f"gp_iter={cfg['global_place_stages'][0].get('iteration')} "
                f"lr={cfg['global_place_stages'][0].get('learning_rate')} "
                f"lr_decay={cfg['global_place_stages'][0].get('learning_rate_decay')} "
                f"wirelength={cfg['global_place_stages'][0].get('wirelength')} "
                f"optimizer={cfg['global_place_stages'][0].get('optimizer')} "
                f"gamma={cfg.get('gamma')} "
                f"stop_overflow={cfg.get('stop_overflow')} "
                f"Llambda={cfg['global_place_stages'][0].get('Llambda_density_weight_iteration')} "
                f"Lsub={cfg['global_place_stages'][0].get('Lsub_iteration')}",
                flush=True,
            )
        except Exception as e:
            print(f"[config-extra-error] failed to apply expanded DREAMPlace knobs: {e}", flush=True)

        if os.environ.get("DREAMPLACE_OMIT_TARGET_DENSITY", "0") == "1":
            cfg.pop("target_density", None)
            changed = True
            print(f"[config] omitted target_density from {json_path}", flush=True)

        if os.environ.get("DREAMPLACE_OMIT_DENSITY_WEIGHT", "0") == "1":
            cfg.pop("density_weight", None)
            changed = True
            print(f"[config] omitted density_weight from {json_path}", flush=True)

        if changed:
            json_path.write_text(json.dumps(cfg, indent=2))

    meta = json.loads(meta_path.read_text())
    return out_dir, meta



def apply_dreamplace_positions_to_all_macros(placement, benchmark, meta, gp_pos):
    """
    Apply DREAMPlace .gp.pl positions to all movable macro-like objects:
    hard macros + soft macros.

    DREAMPlace outputs lower-left DBU coordinates.
    Challenge evaluator expects center coordinates in micron-like units.
    """
    placement = placement.clone()

    macro_names = meta["macro_names"]
    macros = meta["macros"]

    if hasattr(benchmark, "get_movable_mask"):
        movable_ids = torch.where(benchmark.get_movable_mask())[0].tolist()
    else:
        movable_ids = list(range(placement.shape[0]))

    n = min(len(movable_ids), len(macro_names))

    for k in range(n):
        name = macro_names[k]
        safe = safe_name(name)

        if safe not in gp_pos:
            continue

        x_ll_dbu, y_ll_dbu = gp_pos[safe]
        m = macros[name]
        w = float(m["width"])
        h = float(m["height"])

        cx = x_ll_dbu / SCALE + w / 2.0
        cy = y_ll_dbu / SCALE + h / 2.0

        idx = movable_ids[k]
        placement[idx, 0] = float(cx)
        placement[idx, 1] = float(cy)

    return placement


def gpu_soft_macro_spread_post_dreamplace(placement, benchmark, iters: int = 120):
    """
    GPU soft-macro density cleanup.

    This moves soft macros only. Hard macros stay fixed after legalization.
    Soft macros are allowed to overlap, but spreading them away from hard macros
    and each other can reduce density/congestion proxy terms.
    """
    if iters <= 0:
        return placement

    if not torch.cuda.is_available():
        return placement

    device = torch.device("cuda")
    original_device = placement.device

    placement_gpu = placement.clone().to(device)
    sizes = benchmark.macro_sizes.to(device)

    hard_mask = benchmark.get_hard_macro_mask().to(device)
    if hasattr(benchmark, "get_movable_mask"):
        movable_mask = benchmark.get_movable_mask().to(device)
    else:
        movable_mask = torch.ones_like(hard_mask, dtype=torch.bool, device=device)

    soft_mask = movable_mask & (~hard_mask)

    soft_ids = torch.where(soft_mask)[0]
    hard_ids = torch.where(hard_mask)[0]

    if soft_ids.numel() == 0:
        return placement

    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    margin = 1e-4

    lr = float(os.environ.get("SOFT_SPREAD_LR", "0.035"))
    max_step_scale = float(os.environ.get("SOFT_SPREAD_MAX_STEP", "0.12"))

    for it in range(iters):
        soft_pos = placement_gpu[soft_ids]
        soft_wh = sizes[soft_ids]

        force = torch.zeros_like(soft_pos)

        # 1. Soft-vs-hard repulsion. Stronger, because hard macros are final obstacles.
        if hard_ids.numel() > 0:
            hard_pos = placement_gpu[hard_ids]
            hard_wh = sizes[hard_ids]

            # Chunk softs to avoid large memory spikes on big IBM cases.
            chunk = 512
            for start in range(0, soft_ids.numel(), chunk):
                end = min(start + chunk, soft_ids.numel())

                sp = soft_pos[start:end]
                sw = soft_wh[start:end]

                dx = sp[:, 0:1] - hard_pos[:, 0].unsqueeze(0)
                dy = sp[:, 1:2] - hard_pos[:, 1].unsqueeze(0)

                req_x = (sw[:, 0:1] + hard_wh[:, 0].unsqueeze(0)) / 2.0 + 0.02
                req_y = (sw[:, 1:2] + hard_wh[:, 1].unsqueeze(0)) / 2.0 + 0.02

                pen_x = req_x - torch.abs(dx)
                pen_y = req_y - torch.abs(dy)

                overlap = (pen_x > 0) & (pen_y > 0)

                sx = torch.sign(dx)
                sy = torch.sign(dy)
                sx = torch.where(sx == 0, torch.ones_like(sx), sx)
                sy = torch.where(sy == 0, torch.ones_like(sy), sy)

                push_x_axis = pen_x < pen_y

                fx = torch.where(push_x_axis, sx * pen_x, torch.zeros_like(pen_x))
                fy = torch.where(~push_x_axis, sy * pen_y, torch.zeros_like(pen_y))

                fx = torch.where(overlap, fx, torch.zeros_like(fx))
                fy = torch.where(overlap, fy, torch.zeros_like(fy))

                force[start:end, 0] += 1.5 * fx.sum(dim=1)
                force[start:end, 1] += 1.5 * fy.sum(dim=1)

        # 2. Soft-vs-soft mild repulsion. This reduces density spikes without fully legalizing softs.
        chunk = 512
        for start in range(0, soft_ids.numel(), chunk):
            end = min(start + chunk, soft_ids.numel())

            sp = soft_pos[start:end]
            sw = soft_wh[start:end]

            dx = sp[:, 0:1] - soft_pos[:, 0].unsqueeze(0)
            dy = sp[:, 1:2] - soft_pos[:, 1].unsqueeze(0)

            req_x = (sw[:, 0:1] + soft_wh[:, 0].unsqueeze(0)) / 2.0
            req_y = (sw[:, 1:2] + soft_wh[:, 1].unsqueeze(0)) / 2.0

            pen_x = req_x - torch.abs(dx)
            pen_y = req_y - torch.abs(dy)

            overlap = (pen_x > 0) & (pen_y > 0)

            # Remove self-overlap for chunk rows.
            global_rows = torch.arange(start, end, device=device)
            all_cols = torch.arange(soft_ids.numel(), device=device)
            self_mask = global_rows[:, None] == all_cols[None, :]
            overlap = overlap & (~self_mask)

            sx = torch.sign(dx)
            sy = torch.sign(dy)
            sx = torch.where(sx == 0, torch.ones_like(sx), sx)
            sy = torch.where(sy == 0, torch.ones_like(sy), sy)

            push_x_axis = pen_x < pen_y

            fx = torch.where(push_x_axis, sx * pen_x, torch.zeros_like(pen_x))
            fy = torch.where(~push_x_axis, sy * pen_y, torch.zeros_like(pen_y))

            fx = torch.where(overlap, fx, torch.zeros_like(fx))
            fy = torch.where(overlap, fy, torch.zeros_like(fy))

            force[start:end, 0] += 0.25 * fx.sum(dim=1)
            force[start:end, 1] += 0.25 * fy.sum(dim=1)

        norm = torch.norm(force, dim=1, keepdim=True).clamp_min(1e-6)
        max_step = torch.maximum(soft_wh[:, 0], soft_wh[:, 1]).unsqueeze(1) * max_step_scale

        step = force / norm * torch.minimum(norm * lr, max_step)
        placement_gpu[soft_ids] += step

        # Clamp soft macros inside canvas.
        half = sizes[soft_ids] / 2.0
        placement_gpu[soft_ids, 0] = torch.clamp(
            placement_gpu[soft_ids, 0],
            half[:, 0] + margin,
            canvas_w - half[:, 0] - margin,
        )
        placement_gpu[soft_ids, 1] = torch.clamp(
            placement_gpu[soft_ids, 1],
            half[:, 1] + margin,
            canvas_h - half[:, 1] - margin,
        )

        if it % 40 == 39:
            lr *= 0.75

    return placement_gpu.to(original_device)


def normalize_orientation(orient: str) -> str:
    """
    Normalize DREAMPlace/Bookshelf orientations to allowed non-rotated forms.
    The evaluator currently consumes positions only, but keeping orientation
    normalized prevents rotation-width/height mismatch in our conversion path.
    """
    if orient is None:
        return "N"

    orient = str(orient).strip().upper()
    allowed = {"N", "FN", "FS", "S"}
    if orient in allowed:
        return orient
    return "N"


def read_gp_pl(path: Path):
    """
    Read DREAMPlace .gp.pl.

    Returns:
      name -> (x_ll_dbu, y_ll_dbu, orientation)

    DREAMPlace emits lower-left Bookshelf coordinates in DBU.
    """
    positions = {}

    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("UCLA"):
            continue

        parts = line.split()
        if len(parts) < 3:
            continue

        name = parts[0]
        try:
            x = float(parts[1])
            y = float(parts[2])
        except ValueError:
            continue

        orient = "N"
        if ":" in parts:
            colon = parts.index(":")
            if colon + 1 < len(parts):
                orient = normalize_orientation(parts[colon + 1])

        positions[name] = (x, y, orient)

    return positions


def final_bounds_repair(placement, benchmark):
    """
    Final safety pass:
    - clamp every macro center inside the canvas using benchmark.macro_sizes
    - detect and report out-of-bounds count before repair
    """
    placement = placement.clone()
    sizes = benchmark.macro_sizes
    margin = 1e-4

    before = 0
    for i in range(placement.shape[0]):
        w, h = sizes[i]
        x = placement[i, 0]
        y = placement[i, 1]

        if (
            x < w / 2.0 + margin
            or x > benchmark.canvas_width - w / 2.0 - margin
            or y < h / 2.0 + margin
            or y > benchmark.canvas_height - h / 2.0 - margin
        ):
            before += 1

        placement[i, 0] = torch.clamp(
            placement[i, 0],
            w / 2.0 + margin,
            benchmark.canvas_width - w / 2.0 - margin,
        )
        placement[i, 1] = torch.clamp(
            placement[i, 1],
            h / 2.0 + margin,
            benchmark.canvas_height - h / 2.0 - margin,
        )

    if before:
        print(f"[bounds] repaired {before} out-of-bounds macros", flush=True)

    return placement


def apply_dreamplace_positions_to_all_macros(placement, benchmark, meta, gp_pos):
    """
    Apply DREAMPlace .gp.pl positions to all movable macro-like objects:
    hard macros + soft macros.

    DREAMPlace outputs lower-left DBU coordinates.
    Challenge evaluator expects center coordinates.
    """
    placement = placement.clone()

    macro_names = meta["macro_names"]
    macros = meta["macros"]

    if hasattr(benchmark, "get_movable_mask"):
        movable_ids = torch.where(benchmark.get_movable_mask())[0].tolist()
    else:
        movable_ids = list(range(placement.shape[0]))

    n = min(len(movable_ids), len(macro_names))

    missing = 0
    bad_orient = 0

    for k in range(n):
        name = macro_names[k]
        safe = safe_name(name)

        if safe not in gp_pos:
            missing += 1
            continue

        gp_entry = gp_pos[safe]

        # Backward-compatible with older read_gp_pl returning (x, y).
        if len(gp_entry) == 2:
            x_ll_dbu, y_ll_dbu = gp_entry
            orient = "N"
        else:
            x_ll_dbu, y_ll_dbu, orient = gp_entry

        orient_norm = normalize_orientation(orient)
        if orient != orient_norm:
            bad_orient += 1

        m = macros[name]
        w = float(m["width"])
        h = float(m["height"])

        # No width/height swapping: rotations are disallowed.
        cx = x_ll_dbu / SCALE + w / 2.0
        cy = y_ll_dbu / SCALE + h / 2.0

        idx = movable_ids[k]
        placement[idx, 0] = float(cx)
        placement[idx, 1] = float(cy)

    if missing:
        print(f"[DREAMPlace map] missing {missing} gp.pl macro positions", flush=True)
    if bad_orient:
        print(f"[DREAMPlace map] normalized {bad_orient} orientations", flush=True)

    placement = final_bounds_repair(placement, benchmark)
    return placement

def _is_fixed_macro(benchmark, idx: int) -> bool:
    if hasattr(benchmark, "macro_fixed"):
        try:
            return bool(benchmark.macro_fixed[idx])
        except Exception:
            return False
    return False


def _clamp_macro_xy(placement, benchmark, idx: int, margin: float = 1e-4):
    w, h = benchmark.macro_sizes[idx]
    placement[idx, 0] = torch.clamp(
        placement[idx, 0],
        w / 2.0 + margin,
        benchmark.canvas_width - w / 2.0 - margin,
    )
    placement[idx, 1] = torch.clamp(
        placement[idx, 1],
        h / 2.0 + margin,
        benchmark.canvas_height - h / 2.0 - margin,
    )


def _pair_overlap(placement, sizes, i: int, j: int, margin: float = 1e-4):
    xi, yi = float(placement[i, 0]), float(placement[i, 1])
    xj, yj = float(placement[j, 0]), float(placement[j, 1])
    wi, hi = float(sizes[i, 0]), float(sizes[i, 1])
    wj, hj = float(sizes[j, 0]), float(sizes[j, 1])

    px = (wi + wj) / 2.0 + margin - abs(xi - xj)
    py = (hi + hj) / 2.0 + margin - abs(yi - yj)

    if px > 0 and py > 0:
        return px, py
    return 0.0, 0.0


def count_hard_overlaps(placement, benchmark):
    hard_ids = torch.where(benchmark.get_hard_macro_mask())[0].tolist()
    sizes = benchmark.macro_sizes
    count = 0

    for a in range(len(hard_ids)):
        i = hard_ids[a]
        for b in range(a + 1, len(hard_ids)):
            j = hard_ids[b]
            px, py = _pair_overlap(placement, sizes, i, j)
            if px > 0 and py > 0:
                count += 1

    return count


def legalize_hard_macros_pairwise(placement, benchmark, max_iters: int = 300):
    """
    Preserves DREAMPlace structure as much as possible by only pushing apart
    overlapping hard macros. This is a post-DREAMPlace overlap repair, not a
    full placer.
    """
    placement = placement.clone()
    sizes = benchmark.macro_sizes

    hard_mask = benchmark.get_hard_macro_mask()
    hard_ids = torch.where(hard_mask)[0].tolist()

    movable_mask = benchmark.get_movable_mask() if hasattr(benchmark, "get_movable_mask") else hard_mask

    margin = 1e-3

    for idx in hard_ids:
        _clamp_macro_xy(placement, benchmark, idx, margin)

    last_overlap_count = None
    stall = 0

    for _ in range(max_iters):
        moved = 0
        overlap_count = 0

        # Largest macros first tends to stabilize faster.
        hard_ids_sorted = sorted(
            hard_ids,
            key=lambda i: float(sizes[i, 0] * sizes[i, 1]),
            reverse=True,
        )

        for a in range(len(hard_ids_sorted)):
            i = hard_ids_sorted[a]
            for b in range(a + 1, len(hard_ids_sorted)):
                j = hard_ids_sorted[b]

                px, py = _pair_overlap(placement, sizes, i, j, margin)
                if px <= 0 or py <= 0:
                    continue

                overlap_count += 1

                i_fixed = _is_fixed_macro(benchmark, i) or not bool(movable_mask[i])
                j_fixed = _is_fixed_macro(benchmark, j) or not bool(movable_mask[j])

                if i_fixed and j_fixed:
                    continue

                xi, yi = float(placement[i, 0]), float(placement[i, 1])
                xj, yj = float(placement[j, 0]), float(placement[j, 1])

                # Push along smaller penetration axis.
                if px < py:
                    sign = 1.0 if xi >= xj else -1.0
                    delta = px + margin

                    if i_fixed:
                        placement[j, 0] -= sign * delta
                    elif j_fixed:
                        placement[i, 0] += sign * delta
                    else:
                        placement[i, 0] += sign * delta / 2.0
                        placement[j, 0] -= sign * delta / 2.0
                else:
                    sign = 1.0 if yi >= yj else -1.0
                    delta = py + margin

                    if i_fixed:
                        placement[j, 1] -= sign * delta
                    elif j_fixed:
                        placement[i, 1] += sign * delta
                    else:
                        placement[i, 1] += sign * delta / 2.0
                        placement[j, 1] -= sign * delta / 2.0

                _clamp_macro_xy(placement, benchmark, i, margin)
                _clamp_macro_xy(placement, benchmark, j, margin)
                moved += 1

        if overlap_count == 0:
            return placement

        if last_overlap_count == overlap_count:
            stall += 1
        else:
            stall = 0
        last_overlap_count = overlap_count

        # If pairwise repair stalls, stop and let fallback shelf legalizer handle it.
        if stall >= 20:
            break

        if moved == 0:
            break

    return placement


def legalize_hard_macros_shelf_fallback(placement, benchmark):
    """
    Conservative fallback: keeps fixed hard macros where they are, then repacks
    movable hard macros into non-overlapping shelves ordered by DREAMPlace x/y.
    This is more destructive to wirelength but should reduce invalid overlaps.
    """
    placement = placement.clone()
    sizes = benchmark.macro_sizes

    hard_mask = benchmark.get_hard_macro_mask()
    movable_mask = benchmark.get_movable_mask() if hasattr(benchmark, "get_movable_mask") else hard_mask

    hard_ids = torch.where(hard_mask)[0].tolist()
    movable_hard_ids = [
        i for i in hard_ids
        if bool(movable_mask[i]) and not _is_fixed_macro(benchmark, i)
    ]
    fixed_hard_ids = [i for i in hard_ids if i not in movable_hard_ids]

    margin = 1e-3

    # Keep DREAMPlace's broad ordering: bottom-to-top, then left-to-right.
    movable_hard_ids = sorted(
        movable_hard_ids,
        key=lambda i: (float(placement[i, 1]), float(placement[i, 0])),
    )

    placed = fixed_hard_ids[:]

    def overlaps_any(candidate_idx, cx, cy, placed_ids):
        wi, hi = float(sizes[candidate_idx, 0]), float(sizes[candidate_idx, 1])
        for j in placed_ids:
            xj, yj = float(placement[j, 0]), float(placement[j, 1])
            wj, hj = float(sizes[j, 0]), float(sizes[j, 1])
            if abs(cx - xj) < (wi + wj) / 2.0 + margin and abs(cy - yj) < (hi + hj) / 2.0 + margin:
                return True
        return False

    # Candidate grid based on current hard macro sizes.
    min_w = max(1e-3, min(float(sizes[i, 0]) for i in hard_ids))
    min_h = max(1e-3, min(float(sizes[i, 1]) for i in hard_ids))
    step_x = max(min_w / 2.0, benchmark.canvas_width / 100.0)
    step_y = max(min_h / 2.0, benchmark.canvas_height / 100.0)

    for i in movable_hard_ids:
        w, h = float(sizes[i, 0]), float(sizes[i, 1])

        original_x = float(placement[i, 0])
        original_y = float(placement[i, 1])

        best = None
        best_dist = float("inf")

        # Search candidate centers near the original DREAMPlace location first.
        radius_steps = 0
        max_radius_steps = 120

        while radius_steps <= max_radius_steps and best is None:
            xs = [
                original_x + dx * step_x
                for dx in range(-radius_steps, radius_steps + 1)
            ]
            ys = [
                original_y + dy * step_y
                for dy in range(-radius_steps, radius_steps + 1)
            ]

            for cx in xs:
                if cx < w / 2.0 + margin or cx > benchmark.canvas_width - w / 2.0 - margin:
                    continue
                for cy in ys:
                    if cy < h / 2.0 + margin or cy > benchmark.canvas_height - h / 2.0 - margin:
                        continue
                    if overlaps_any(i, cx, cy, placed):
                        continue

                    dist = (cx - original_x) ** 2 + (cy - original_y) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best = (cx, cy)

            radius_steps += 4

        # Absolute fallback: simple shelf scan.
        if best is None:
            y = h / 2.0 + margin
            found = False
            while y <= benchmark.canvas_height - h / 2.0 - margin and not found:
                x = w / 2.0 + margin
                while x <= benchmark.canvas_width - w / 2.0 - margin:
                    if not overlaps_any(i, x, y, placed):
                        best = (x, y)
                        found = True
                        break
                    x += step_x
                y += step_y

        if best is not None:
            placement[i, 0] = float(best[0])
            placement[i, 1] = float(best[1])
        else:
            _clamp_macro_xy(placement, benchmark, i, margin)

        placed.append(i)

    for idx in hard_ids:
        _clamp_macro_xy(placement, benchmark, idx, margin)

    return placement



def legalize_hard_macros_gpu_repulsion(placement, benchmark, max_iters: int = 2000):
    """
    GPU-vectorized hard-macro overlap reduction.
    Computes all hard-macro pair overlaps in batched torch tensors and applies
    parallel repulsion. This is intended to reduce overlap count before the
    slower CPU fallback legalizer.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_device = placement.device
    placement = placement.clone().to(device)
    sizes = benchmark.macro_sizes.to(device)

    hard_ids = torch.where(benchmark.get_hard_macro_mask())[0].to(device)
    if hard_ids.numel() <= 1:
        return placement.to(original_device)

    if hasattr(benchmark, "get_movable_mask"):
        movable_mask = benchmark.get_movable_mask().to(device)
    else:
        movable_mask = benchmark.get_hard_macro_mask().to(device)

    hard_movable = movable_mask[hard_ids].float()

    if hasattr(benchmark, "macro_fixed"):
        try:
            fixed_mask = benchmark.macro_fixed.to(device).bool()
            hard_movable = hard_movable * (~fixed_mask[hard_ids]).float()
        except Exception:
            pass

    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)

    margin = 1e-3
    lr = 0.20

    for it in range(max_iters):
        pos = placement[hard_ids]
        wh = sizes[hard_ids]

        x = pos[:, 0]
        y = pos[:, 1]
        w = wh[:, 0]
        h = wh[:, 1]

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]

        req_x = (w[:, None] + w[None, :]) / 2.0 + margin
        req_y = (h[:, None] + h[None, :]) / 2.0 + margin

        pen_x = req_x - torch.abs(dx)
        pen_y = req_y - torch.abs(dy)

        overlap = (pen_x > 0) & (pen_y > 0)
        eye = torch.eye(hard_ids.numel(), dtype=torch.bool, device=device)
        overlap = overlap & (~eye)

        overlap_count = int(overlap.sum().item() // 2)
        if overlap_count == 0:
            break

        push_x_axis = pen_x < pen_y

        sign_x = torch.sign(dx)
        sign_y = torch.sign(dy)
        sign_x = torch.where(sign_x == 0, torch.ones_like(sign_x), sign_x)
        sign_y = torch.where(sign_y == 0, torch.ones_like(sign_y), sign_y)

        fx_pair = torch.where(push_x_axis, sign_x * pen_x, torch.zeros_like(pen_x))
        fy_pair = torch.where(~push_x_axis, sign_y * pen_y, torch.zeros_like(pen_y))

        fx_pair = torch.where(overlap, fx_pair, torch.zeros_like(fx_pair))
        fy_pair = torch.where(overlap, fy_pair, torch.zeros_like(fy_pair))

        fx = fx_pair.sum(dim=1)
        fy = fy_pair.sum(dim=1)
        force = torch.stack([fx, fy], dim=1)

        norm = torch.norm(force, dim=1, keepdim=True).clamp_min(1e-6)
        max_step = torch.maximum(w, h).unsqueeze(1) * 0.25
        step = force / norm * torch.minimum(norm * lr, max_step)

        step = step * hard_movable[:, None]
        placement[hard_ids] += step

        half = sizes[hard_ids] / 2.0
        placement[hard_ids, 0] = torch.clamp(
            placement[hard_ids, 0],
            half[:, 0] + margin,
            canvas_w - half[:, 0] - margin,
        )
        placement[hard_ids, 1] = torch.clamp(
            placement[hard_ids, 1],
            half[:, 1] + margin,
            canvas_h - half[:, 1] - margin,
        )

        if it % 200 == 199:
            lr *= 0.75

    return placement.to(original_device)

def legalize_hard_macros_post_dreamplace(placement, benchmark):
    before = count_hard_overlaps(placement, benchmark)
    print(f"[DREAMPlace legalizer] hard overlaps before: {before}", flush=True)

    placement = legalize_hard_macros_gpu_repulsion(placement, benchmark, max_iters=2000)
    mid_gpu = count_hard_overlaps(placement, benchmark)
    print(f"[DREAMPlace legalizer] after GPU repulsion: {mid_gpu}", flush=True)

    if mid_gpu > 0:
        if os.environ.get("SKIP_PAIRWISE_POLISH", "0") == "1":
            print("[DREAMPlace legalizer] skipping pairwise polish", flush=True)
            mid = mid_gpu
        else:
            pairwise_iters = int(os.environ.get("PAIRWISE_POLISH_ITERS", "100"))
            print(f"[DREAMPlace legalizer] running pairwise polish iters={pairwise_iters}", flush=True)
            placement = legalize_hard_macros_pairwise(placement, benchmark, max_iters=pairwise_iters)
            mid = count_hard_overlaps(placement, benchmark)
            print(f"[DREAMPlace legalizer] after pairwise polish: {mid}", flush=True)
    else:
        mid = mid_gpu

    if mid > 0:
        if os.environ.get("SKIP_SHELF_FALLBACK", "0") == "1":
            print("[DREAMPlace legalizer] skipping shelf fallback", flush=True)
            after = mid
        else:
            placement = legalize_hard_macros_shelf_fallback(placement, benchmark)
            after = count_hard_overlaps(placement, benchmark)
            print(f"[DREAMPlace legalizer] after shelf fallback: {after}", flush=True)
    else:
        after = mid

    return placement


def _hard_overlap_pairs(placement, benchmark):
    hard_ids = torch.where(benchmark.get_hard_macro_mask())[0].tolist()
    sizes = benchmark.macro_sizes
    pairs = []

    for a in range(len(hard_ids)):
        i = hard_ids[a]
        for b in range(a + 1, len(hard_ids)):
            j = hard_ids[b]
            px, py = _pair_overlap(placement, sizes, i, j)
            if px > 0 and py > 0:
                pairs.append((i, j, px, py))

    return pairs


def legalize_hard_macros_shelf_fallback(placement, benchmark):
    """
    Fast targeted fallback legalizer.

    Instead of repacking all movable hard macros, it identifies the small set of
    macros still involved in overlaps after GPU repulsion + pairwise polish, removes
    only those movable macros, and reinserts them near their DREAMPlace positions
    using vectorized candidate scoring.
    """
    placement = placement.clone()
    sizes = benchmark.macro_sizes
    device = placement.device

    hard_mask = benchmark.get_hard_macro_mask()
    hard_ids = torch.where(hard_mask)[0].tolist()

    if hasattr(benchmark, "get_movable_mask"):
        movable_mask = benchmark.get_movable_mask()
    else:
        movable_mask = hard_mask

    margin = 1e-3

    def is_movable_hard(i):
        return bool(hard_mask[i]) and bool(movable_mask[i]) and not _is_fixed_macro(benchmark, i)

    # Try a few targeted passes. Usually only a small number of overlaps remain.
    for pass_id in range(6):
        pairs = _hard_overlap_pairs(placement, benchmark)
        if not pairs:
            break

        conflict = set()
        for i, j, _, _ in pairs:
            if is_movable_hard(i):
                conflict.add(i)
            if is_movable_hard(j):
                conflict.add(j)

        if not conflict:
            break

        # Reinsert larger conflict macros first.
        conflict_ids = sorted(
            conflict,
            key=lambda i: float(sizes[i, 0] * sizes[i, 1]),
            reverse=True,
        )

        placed_ids = [i for i in hard_ids if i not in conflict]

        for i in conflict_ids:
            w = float(sizes[i, 0])
            h = float(sizes[i, 1])
            orig_x = float(placement[i, 0])
            orig_y = float(placement[i, 1])

            best = None
            best_dist = float("inf")

            # Candidate spacing tied to this macro size and canvas.
            step_x = max(w * 0.25, benchmark.canvas_width / 120.0)
            step_y = max(h * 0.25, benchmark.canvas_height / 120.0)

            # Local expanding square search. Vectorized per radius.
            for radius in range(0, 70, 2):
                xs = torch.arange(-radius, radius + 1, 1, dtype=placement.dtype, device=device) * step_x + orig_x
                ys = torch.arange(-radius, radius + 1, 1, dtype=placement.dtype, device=device) * step_y + orig_y

                if xs.numel() == 0 or ys.numel() == 0:
                    continue

                gx, gy = torch.meshgrid(xs, ys, indexing="ij")
                cand = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)

                # Bounds.
                in_bounds = (
                    (cand[:, 0] >= w / 2.0 + margin) &
                    (cand[:, 0] <= benchmark.canvas_width - w / 2.0 - margin) &
                    (cand[:, 1] >= h / 2.0 + margin) &
                    (cand[:, 1] <= benchmark.canvas_height - h / 2.0 - margin)
                )

                cand = cand[in_bounds]
                if cand.numel() == 0:
                    continue

                if placed_ids:
                    placed_t = torch.tensor(placed_ids, dtype=torch.long, device=device)
                    ppos = placement[placed_t]
                    psz = sizes[placed_t].to(device)

                    dx = torch.abs(cand[:, 0:1] - ppos[:, 0].unsqueeze(0))
                    dy = torch.abs(cand[:, 1:2] - ppos[:, 1].unsqueeze(0))

                    req_x = (w + psz[:, 0]).unsqueeze(0) / 2.0 + margin
                    req_y = (h + psz[:, 1]).unsqueeze(0) / 2.0 + margin

                    overlap = (dx < req_x) & (dy < req_y)
                    ok = ~overlap.any(dim=1)
                    cand = cand[ok]

                    if cand.numel() == 0:
                        continue

                dist = (cand[:, 0] - orig_x) ** 2 + (cand[:, 1] - orig_y) ** 2
                k = int(torch.argmin(dist).item())
                d = float(dist[k].item())

                if d < best_dist:
                    best_dist = d
                    best = (float(cand[k, 0].item()), float(cand[k, 1].item()))

                if best is not None:
                    break

            # Absolute fallback: coarser whole-canvas grid, vectorized in chunks.
            if best is None:
                grid_step_x = max(w * 0.5, benchmark.canvas_width / 80.0)
                grid_step_y = max(h * 0.5, benchmark.canvas_height / 80.0)

                xs = torch.arange(
                    w / 2.0 + margin,
                    benchmark.canvas_width - w / 2.0 - margin,
                    grid_step_x,
                    dtype=placement.dtype,
                    device=device,
                )
                ys = torch.arange(
                    h / 2.0 + margin,
                    benchmark.canvas_height - h / 2.0 - margin,
                    grid_step_y,
                    dtype=placement.dtype,
                    device=device,
                )

                if xs.numel() > 0 and ys.numel() > 0:
                    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
                    cand_all = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)

                    # Sort by distance to original, then test in chunks.
                    dist_all = (cand_all[:, 0] - orig_x) ** 2 + (cand_all[:, 1] - orig_y) ** 2
                    order = torch.argsort(dist_all)
                    cand_all = cand_all[order]

                    placed_t = torch.tensor(placed_ids, dtype=torch.long, device=device) if placed_ids else None

                    for start in range(0, cand_all.shape[0], 4096):
                        cand = cand_all[start:start + 4096]

                        if placed_t is not None and placed_t.numel() > 0:
                            ppos = placement[placed_t]
                            psz = sizes[placed_t].to(device)

                            dx = torch.abs(cand[:, 0:1] - ppos[:, 0].unsqueeze(0))
                            dy = torch.abs(cand[:, 1:2] - ppos[:, 1].unsqueeze(0))

                            req_x = (w + psz[:, 0]).unsqueeze(0) / 2.0 + margin
                            req_y = (h + psz[:, 1]).unsqueeze(0) / 2.0 + margin

                            overlap = (dx < req_x) & (dy < req_y)
                            ok = ~overlap.any(dim=1)
                            cand = cand[ok]

                        if cand.numel() > 0:
                            best = (float(cand[0, 0].item()), float(cand[0, 1].item()))
                            break

            if best is not None:
                placement[i, 0] = best[0]
                placement[i, 1] = best[1]
            else:
                _clamp_macro_xy(placement, benchmark, i, margin)

            placed_ids.append(i)

        for idx in hard_ids:
            _clamp_macro_xy(placement, benchmark, idx, margin)

    return placement


def _profile_stage(stage_name, fn, *args, **kwargs):
    """
    Lightweight runtime profiler for major placer stages.
    """
    t0 = time.time()
    result = fn(*args, **kwargs)
    dt = time.time() - t0
    print(f"[profile] {stage_name} took {dt:.3f}s", flush=True)
    return result


def _profile_msg(stage_name, t0):
    dt = time.time() - t0
    print(f"[profile] {stage_name} took {dt:.3f}s", flush=True)
    return dt


def hard_macro_gap_audit(placement, benchmark, safety_gap: float = None):
    """
    Final hard-macro legality audit.

    Reports:
      - exact hard overlap count
      - near-overlap count under safety_gap
      - minimum signed gap across hard macro pairs

    Signed gap definition:
      For non-overlapping rectangles, gap is positive.
      For overlapping rectangles, gap is negative by penetration amount.
    """
    if safety_gap is None:
        safety_gap = float(os.environ.get("HARD_MACRO_SAFETY_GAP", "0.006"))

    hard_ids = torch.where(benchmark.get_hard_macro_mask())[0].tolist()
    sizes = benchmark.macro_sizes

    overlap_count = 0
    near_count = 0
    min_gap = float("inf")
    worst_pair = None

    for a in range(len(hard_ids)):
        i = hard_ids[a]
        xi = float(placement[i, 0])
        yi = float(placement[i, 1])
        wi = float(sizes[i, 0])
        hi = float(sizes[i, 1])

        for b in range(a + 1, len(hard_ids)):
            j = hard_ids[b]
            xj = float(placement[j, 0])
            yj = float(placement[j, 1])
            wj = float(sizes[j, 0])
            hj = float(sizes[j, 1])

            dx_gap = abs(xi - xj) - (wi + wj) / 2.0
            dy_gap = abs(yi - yj) - (hi + hj) / 2.0

            if dx_gap < 0 and dy_gap < 0:
                # True overlap; signed gap is negative penetration on the easier separating axis.
                pair_gap = max(dx_gap, dy_gap)
                overlap_count += 1
            else:
                # Non-overlap; separation exists on at least one axis.
                # Use the axis with positive separation if possible.
                if dx_gap >= 0 and dy_gap >= 0:
                    pair_gap = min(dx_gap, dy_gap)
                elif dx_gap >= 0:
                    pair_gap = dx_gap
                else:
                    pair_gap = dy_gap

                if pair_gap < safety_gap:
                    near_count += 1

            if pair_gap < min_gap:
                min_gap = pair_gap
                worst_pair = (i, j)

    if min_gap == float("inf"):
        min_gap = 0.0

    print(
        f"[legality] hard_overlaps={overlap_count} near_pairs={near_count} "
        f"min_gap={min_gap:.6f} safety_gap={safety_gap:.6f} worst_pair={worst_pair}",
        flush=True,
    )

    return overlap_count, near_count, min_gap


def final_legality_margin_repair(placement, benchmark):
    """
    Conservative final safety repair.

    If true overlaps or near-overlaps remain, run a small extra pairwise
    legalization pass with a safety margin, then re-audit.
    """
    safety_gap = float(os.environ.get("HARD_MACRO_SAFETY_GAP", "0.006"))

    overlaps, near_pairs, min_gap = hard_macro_gap_audit(
        placement, benchmark, safety_gap=safety_gap
    )

    # Only repair true overlaps. Near-pairs are reported for audit,
    # but repairing them is too expensive and usually unnecessary.
    if overlaps == 0:
        return placement

    print(
        f"[legality] running final safety repair for true overlaps={overlaps}",
        flush=True,
    )

    # Temporarily use the existing pairwise legalization machinery.
    # This may move macros slightly, so keep iterations modest.
    repaired = legalize_hard_macros_pairwise(
        placement,
        benchmark,
        max_iters=int(os.environ.get("FINAL_SAFETY_REPAIR_ITERS", "80")),
    )

    repaired = final_bounds_repair(repaired, benchmark)

    hard_macro_gap_audit(repaired, benchmark, safety_gap=safety_gap)
    return repaired


def legalization_displacement_audit(anchor_placement, final_placement, benchmark):
    """
    Measures how much legalization moved macros away from the DREAMPlace output.

    This is a preservation audit, not an optimizer. It helps identify when the
    legalizer is damaging DREAMPlace's placement structure.
    """
    hard_ids = torch.where(benchmark.get_hard_macro_mask())[0].tolist()

    if hasattr(benchmark, "get_movable_mask"):
        movable_ids = torch.where(benchmark.get_movable_mask())[0].tolist()
    else:
        movable_ids = list(range(final_placement.shape[0]))

    delta = final_placement - anchor_placement
    dist = torch.sqrt(torch.sum(delta * delta, dim=1))

    hard_dist = dist[hard_ids] if hard_ids else torch.tensor([], device=dist.device)
    movable_dist = dist[movable_ids] if movable_ids else torch.tensor([], device=dist.device)

    def stats(x):
        if x.numel() == 0:
            return 0.0, 0.0, 0.0, 0
        moved = int(torch.sum(x > 1e-6).item())
        return (
            float(torch.mean(x).item()),
            float(torch.max(x).item()),
            float(torch.quantile(x, 0.95).item()) if x.numel() >= 2 else float(torch.max(x).item()),
            moved,
        )

    hard_avg, hard_max, hard_p95, hard_moved = stats(hard_dist)
    mov_avg, mov_max, mov_p95, mov_moved = stats(movable_dist)

    print(
        f"[preserve] hard_moved={hard_moved}/{len(hard_ids)} "
        f"hard_avg_disp={hard_avg:.6f} hard_p95_disp={hard_p95:.6f} hard_max_disp={hard_max:.6f}",
        flush=True,
    )

    print(
        f"[preserve] movable_moved={mov_moved}/{len(movable_ids)} "
        f"movable_avg_disp={mov_avg:.6f} movable_p95_disp={mov_p95:.6f} movable_max_disp={mov_max:.6f}",
        flush=True,
    )

    return final_placement


def anchor_restore_hard_macros(anchor_placement, placement, benchmark):
    """
    Post-legalization preservation pass.

    The hard legalizer may move macros far from their DREAMPlace positions.
    This pass tries to move each hard macro back toward its DREAMPlace anchor,
    accepting only moves that keep hard macros non-overlapping and in-bounds.

    It does not create new overlaps.
    """
    placement = placement.clone()
    sizes = benchmark.macro_sizes

    hard_ids = torch.where(benchmark.get_hard_macro_mask())[0].tolist()

    if hasattr(benchmark, "get_movable_mask"):
        movable_mask = benchmark.get_movable_mask()
    else:
        movable_mask = benchmark.get_hard_macro_mask()

    margin = float(os.environ.get("ANCHOR_RESTORE_MARGIN", "0.0045"))
    min_restore_disp = float(os.environ.get("ANCHOR_RESTORE_MIN_DISP", "0.02"))
    max_passes = int(os.environ.get("ANCHOR_RESTORE_PASSES", "2"))

    def is_movable_hard(i):
        return bool(movable_mask[i]) and not _is_fixed_macro(benchmark, i)

    def in_bounds(i, x, y):
        w = float(sizes[i, 0])
        h = float(sizes[i, 1])
        return (
            x >= w / 2.0 + margin
            and x <= float(benchmark.canvas_width) - w / 2.0 - margin
            and y >= h / 2.0 + margin
            and y <= float(benchmark.canvas_height) - h / 2.0 - margin
        )

    def overlaps_any_hard(i, x, y):
        wi = float(sizes[i, 0])
        hi = float(sizes[i, 1])

        for j in hard_ids:
            if j == i:
                continue

            xj = float(placement[j, 0])
            yj = float(placement[j, 1])
            wj = float(sizes[j, 0])
            hj = float(sizes[j, 1])

            if abs(x - xj) < (wi + wj) / 2.0 + margin and abs(y - yj) < (hi + hj) / 2.0 + margin:
                return True

        return False

    def valid(i, x, y):
        return in_bounds(i, x, y) and not overlaps_any_hard(i, x, y)

    moved_total = 0
    total_restore_dist = 0.0

    for _pass in range(max_passes):
        # Recompute displacement from anchor each pass.
        delta = placement - anchor_placement
        dist = torch.sqrt(torch.sum(delta * delta, dim=1))

        candidates = [
            i for i in hard_ids
            if is_movable_hard(i) and float(dist[i]) > min_restore_disp
        ]

        # Restore largest displacements first.
        candidates.sort(key=lambda i: float(dist[i]), reverse=True)

        moved_this_pass = 0

        for i in candidates:
            cur_x = float(placement[i, 0])
            cur_y = float(placement[i, 1])
            anc_x = float(anchor_placement[i, 0])
            anc_y = float(anchor_placement[i, 1])

            best = None
            best_alpha = 0.0

            # Try increasingly conservative moves toward anchor.
            for alpha in [1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05]:
                x = cur_x + alpha * (anc_x - cur_x)
                y = cur_y + alpha * (anc_y - cur_y)

                if valid(i, x, y):
                    best = (x, y)
                    best_alpha = alpha
                    break

            if best is not None and best_alpha > 0:
                old_x, old_y = cur_x, cur_y
                placement[i, 0] = best[0]
                placement[i, 1] = best[1]
                moved_this_pass += 1
                moved_total += 1
                total_restore_dist += ((best[0] - old_x) ** 2 + (best[1] - old_y) ** 2) ** 0.5

        if moved_this_pass == 0:
            break

    print(
        f"[preserve] anchor_restore moved={moved_total} "
        f"total_restore_dist={total_restore_dist:.6f}",
        flush=True,
    )

    return placement


def preserve_run_artifacts(bench, out_dir, placement):
    """
    Save exact artifacts for this run.

    Enabled when:
      SAVE_RUN_ARTIFACTS=1

    Uses:
      ARTIFACT_RUN_ID=<name>
    if provided. Otherwise a timestamp is used.
    """
    if os.environ.get("SAVE_RUN_ARTIFACTS", "0") != "1":
        return placement

    run_id = os.environ.get("ARTIFACT_RUN_ID")
    if not run_id:
        run_id = time.strftime("%Y%m%d_%H%M%S")

    artifact_dir = OUT_ROOT / bench / "artifacts" / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Save final returned placement tensor.
    torch.save(placement.detach().cpu(), artifact_dir / "placement.pt")

    # Save CSV version too.
    with (artifact_dir / "placement.csv").open("w") as f:
        f.write("idx,x,y\n")
        cpu = placement.detach().cpu()
        for i in range(cpu.shape[0]):
            f.write(f"{i},{float(cpu[i,0])},{float(cpu[i,1])}\n")

    # Copy converter / DREAMPlace config files.
    for name in [
        f"{bench}.json",
        f"{bench}.aux",
        f"{bench}.nodes",
        f"{bench}.nets",
        f"{bench}.pl",
        f"{bench}.scl",
        f"{bench}.wts",
        "meta.json",
        "run_from_evaluator.log",
    ]:
        src = out_dir / name
        if src.exists():
            shutil.copy2(src, artifact_dir / name)

    # Copy DREAMPlace gp.pl if it exists.
    gp = out_dir / "results" / bench / f"{bench}.gp.pl"
    if gp.exists():
        shutil.copy2(gp, artifact_dir / f"{bench}.gp.pl")

    # Save environment/config snapshot.
    with (artifact_dir / "run_env.txt").open("w") as f:
        keys = [
            "DREAMPLACE_FORCE",
            "DREAMPLACE_TIMEOUT_SEC",
            "DREAMPLACE_RANDOM_SEED",
            "DREAMPLACE_TARGET_DENSITY",
            "DREAMPLACE_DENSITY_WEIGHT",
            "DREAMPLACE_OMIT_DENSITY_WEIGHT",
            "DREAMPLACE_DETERMINISTIC",
            "SOFT_MODE",
            "EXPLICIT_ENV_OVERRIDE",
            "CHALLENGE_ROOT",
            "DREAMPLACE_ROOT",
            "DREAMPLACE_PYTHON",
        ]
        for k in keys:
            f.write(f"{k}={os.environ.get(k, '')}\n")

    print(f"[artifacts] saved run artifacts to {artifact_dir}", flush=True)

    return placement



def soft_dreamplace_refine(placement, benchmark, bench, out_dir, meta):
    """
    Optional second DREAMPlace pass for soft macros.

    Enable with:
        SOFT_DREAMPLACE_REFINE=1

    Flow:
      - hard macros stay fixed at the current legalized positions
      - soft macros remain movable
      - DREAMPlace is rerun on a copied Bookshelf problem
      - only soft macro positions are copied back
    """
    if os.environ.get("SOFT_DREAMPLACE_REFINE", "0") != "1":
        return placement

    t0 = time.time()
    refine_dir = out_dir / "soft_refine"
    refine_dir.mkdir(parents=True, exist_ok=True)

    # Copy base Bookshelf files.
    for suffix in ["aux", "nodes", "nets", "scl", "wts"]:
        src = out_dir / f"{bench}.{suffix}"
        if src.exists():
            shutil.copy2(src, refine_dir / f"{bench}.{suffix}")

    # Load current DREAMPlace JSON and repoint paths.
    src_json = out_dir / f"{bench}.json"
    cfg = json.loads(src_json.read_text())
    cfg["aux_input"] = str((refine_dir / f"{bench}.aux").resolve())
    cfg["result_dir"] = str((refine_dir / "results").resolve())
    (refine_dir / f"{bench}.json").write_text(json.dumps(cfg, indent=2))

    macro_names = meta["macro_names"]
    macros = meta["macros"]

    if hasattr(benchmark, "get_movable_mask"):
        movable_ids = torch.where(benchmark.get_movable_mask())[0].tolist()
    else:
        movable_ids = list(range(placement.shape[0]))

    hard_mask = benchmark.get_hard_macro_mask()
    hard_ids = set(torch.where(hard_mask)[0].tolist())

    n = min(len(movable_ids), len(macro_names))
    idx_to_name = {movable_ids[k]: macro_names[k] for k in range(n)}
    name_to_idx = {v: k for k, v in idx_to_name.items()}

    hard_names = set()
    soft_names = set()

    for idx, name in idx_to_name.items():
        if idx in hard_ids:
            hard_names.add(safe_name(name))
        else:
            soft_names.add(safe_name(name))

    # Rewrite .nodes so hard macros become terminals.
    nodes_path = refine_dir / f"{bench}.nodes"
    if nodes_path.exists():
        lines = nodes_path.read_text().splitlines()
        new_lines = []

        for line in lines:
            parts = line.split()
            if parts and parts[0] in hard_names and "terminal" not in parts:
                new_lines.append(line + " terminal")
            else:
                new_lines.append(line)

        nodes_path.write_text("\n".join(new_lines) + "\n")

    # Parse original .pl so ports/fixed terminals are preserved.
    original_pl = out_dir / f"{bench}.pl"
    original_lines = []
    if original_pl.exists():
        original_lines = original_pl.read_text().splitlines()

    line_by_name = {}
    for line in original_lines:
        parts = line.split()
        if len(parts) >= 3 and not line.startswith("UCLA"):
            line_by_name[parts[0]] = line

    # Rewrite .pl using current placement for all macros.
    pl_path = refine_dir / f"{bench}.pl"
    with pl_path.open("w") as f:
        f.write("UCLA pl 1.0\n\n")

        written = set()

        for k in range(n):
            idx = movable_ids[k]
            name = macro_names[k]
            safe = safe_name(name)
            m = macros[name]

            w = float(m["width"])
            h = float(m["height"])
            cx = float(placement[idx, 0])
            cy = float(placement[idx, 1])
            x_ll = (cx - w / 2.0) * SCALE
            y_ll = (cy - h / 2.0) * SCALE

            if safe in hard_names:
                f.write(f"{safe} {x_ll:.6f} {y_ll:.6f} : N /FIXED\n")
            else:
                f.write(f"{safe} {x_ll:.6f} {y_ll:.6f} : N\n")

            written.add(safe)

        # Preserve ports and other fixed terminals from original .pl.
        for safe, line in line_by_name.items():
            if safe not in written:
                f.write(line + "\n")

    # Run DREAMPlace on the soft-refine copy.
    cmd = [
        str(DREAMPLACE_PYTHON),
        str(DREAMPLACE_ROOT / "dreamplace/Placer.py"),
        str(refine_dir / f"{bench}.json"),
    ]

    log_path = refine_dir / "soft_refine_dreamplace.log"
    timeout_sec = float(os.environ.get("SOFT_DREAMPLACE_TIMEOUT_SEC", "900"))

    gp_pl = refine_dir / "results" / bench / f"{bench}.gp.pl"

    try:
        with log_path.open("w") as log:
            subprocess.run(
                cmd,
                cwd=str(DREAMPLACE_ROOT),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=timeout_sec,
            )
    except Exception as e:
        print(f"[soft_dp] soft DREAMPlace refine failed: {type(e).__name__}: {e}", flush=True)
        return placement

    if not gp_pl.exists():
        print(f"[soft_dp] no soft-refine gp.pl produced at {gp_pl}", flush=True)
        return placement

    gp_pos = read_gp_pl(gp_pl)
    refined = placement.clone()

    changed = 0
    for k in range(n):
        idx = movable_ids[k]
        name = macro_names[k]
        safe = safe_name(name)

        if idx in hard_ids:
            continue

        if safe not in gp_pos:
            continue

        entry = gp_pos[safe]
        if len(entry) == 2:
            x_ll_dbu, y_ll_dbu = entry
        else:
            x_ll_dbu, y_ll_dbu, _orient = entry

        m = macros[name]
        w = float(m["width"])
        h = float(m["height"])

        cx = x_ll_dbu / SCALE + w / 2.0
        cy = y_ll_dbu / SCALE + h / 2.0

        refined[idx, 0] = float(cx)
        refined[idx, 1] = float(cy)
        changed += 1

    refined = final_bounds_repair(refined, benchmark)

    print(
        f"[soft_dp] updated {changed} soft macros using second DREAMPlace pass "
        f"in {time.time() - t0:.3f}s",
        flush=True,
    )

    return refined



# =============================================================================
# Fast spatial-hash hard macro legalizer
# =============================================================================

def _spatial_hash_macro_sizes(benchmark):
    if hasattr(benchmark, "macro_sizes"):
        return benchmark.macro_sizes
    if hasattr(benchmark, "node_sizes"):
        return benchmark.node_sizes
    raise AttributeError("Could not find macro/node sizes on benchmark")


def _spatial_hash_is_fixed_macro(benchmark, idx: int) -> bool:
    # Reuse existing helper if present.
    if "_is_fixed_macro" in globals():
        try:
            return bool(_is_fixed_macro(benchmark, idx))
        except Exception:
            pass

    try:
        return bool(benchmark.macro_fixed[idx])
    except Exception:
        return False


def _spatial_hash_bounds_repair(placement, benchmark, ids):
    sizes = _spatial_hash_macro_sizes(benchmark)
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)

    out = placement

    for i in ids:
        w = float(sizes[i, 0])
        h = float(sizes[i, 1])

        xmin = w / 2.0
        xmax = canvas_w - w / 2.0
        ymin = h / 2.0
        ymax = canvas_h - h / 2.0

        out[i, 0] = float(max(xmin, min(xmax, float(out[i, 0]))))
        out[i, 1] = float(max(ymin, min(ymax, float(out[i, 1]))))

    return out


def _spatial_hash_hard_pairs(placement, benchmark, hard_ids, safety_gap=0.0):
    """
    Build candidate hard-hard pairs using spatial hashing.
    This avoids checking all hard macro pairs repeatedly.
    """
    sizes = _spatial_hash_macro_sizes(benchmark)

    if not hard_ids:
        return []

    widths = [float(sizes[i, 0]) for i in hard_ids]
    heights = [float(sizes[i, 1]) for i in hard_ids]

    med_w = sorted(widths)[len(widths) // 2]
    med_h = sorted(heights)[len(heights) // 2]
    cell = max(1e-6, 0.75 * max(med_w, med_h) + float(safety_gap))

    bins = {}
    rects = {}

    for i in hard_ids:
        x = float(placement[i, 0])
        y = float(placement[i, 1])
        w = float(sizes[i, 0])
        h = float(sizes[i, 1])

        xmin = x - w / 2.0 - safety_gap
        xmax = x + w / 2.0 + safety_gap
        ymin = y - h / 2.0 - safety_gap
        ymax = y + h / 2.0 + safety_gap

        rects[i] = (xmin, xmax, ymin, ymax)

        bx0 = int(xmin // cell)
        bx1 = int(xmax // cell)
        by0 = int(ymin // cell)
        by1 = int(ymax // cell)

        for bx in range(bx0, bx1 + 1):
            for by in range(by0, by1 + 1):
                bins.setdefault((bx, by), []).append(i)

    pairs = set()

    for members in bins.values():
        n = len(members)
        if n <= 1:
            continue
        for a in range(n):
            ia = members[a]
            for b in range(a + 1, n):
                ib = members[b]
                if ia == ib:
                    continue
                pairs.add((ia, ib) if ia < ib else (ib, ia))

    return list(pairs)


def _spatial_hash_count_overlaps(placement, benchmark, hard_ids, safety_gap=0.0):
    sizes = _spatial_hash_macro_sizes(benchmark)
    pairs = _spatial_hash_hard_pairs(placement, benchmark, hard_ids, safety_gap=safety_gap)

    overlap_pairs = []
    total_area = 0.0
    min_gap = 1e9

    for i, j in pairs:
        xi = float(placement[i, 0])
        yi = float(placement[i, 1])
        wi = float(sizes[i, 0])
        hi = float(sizes[i, 1])

        xj = float(placement[j, 0])
        yj = float(placement[j, 1])
        wj = float(sizes[j, 0])
        hj = float(sizes[j, 1])

        dx = abs(xi - xj)
        dy = abs(yi - yj)

        ox = (wi + wj) / 2.0 + safety_gap - dx
        oy = (hi + hj) / 2.0 + safety_gap - dy

        if ox > 0 and oy > 0:
            area = ox * oy
            overlap_pairs.append((i, j, ox, oy, area))
            total_area += area
            min_gap = min(min_gap, min(ox, oy))

    return overlap_pairs, total_area, min_gap


def legalize_hard_macros_spatial_hash(placement, benchmark):
    """
    Fast hard macro legalization using:
    - spatial hash pair generation
    - vector-free minimum-axis pushes
    - anchor preservation
    - bounded iteration count

    This is meant to replace slow CPU pairwise/shelf behavior on hard cases.
    It is intentionally conservative and exact legality is still audited later.
    """
    t0 = time.time()

    original_device = placement.device
    out = placement.detach().clone().cpu()
    anchor = out.clone()

    hard_mask = benchmark.get_hard_macro_mask()
    if hasattr(hard_mask, "detach"):
        hard_mask = hard_mask.detach().cpu()

    hard_ids = [int(i) for i in torch.nonzero(hard_mask, as_tuple=False).flatten().tolist()]

    movable = {}
    for i in hard_ids:
        movable[i] = not _spatial_hash_is_fixed_macro(benchmark, i)

    sizes = _spatial_hash_macro_sizes(benchmark)
    if hasattr(sizes, "detach"):
        sizes_cpu = sizes.detach().cpu()
    else:
        sizes_cpu = sizes

    # Move smaller hard macros more than larger hard macros.
    move_weight = {}
    for i in hard_ids:
        w = float(sizes_cpu[i, 0])
        h = float(sizes_cpu[i, 1])
        move_weight[i] = max(1e-9, (w * h) ** 0.5)

    max_iters = int(os.environ.get("SPATIAL_HASH_LEGALIZER_ITERS", "80"))
    safety_gap = float(os.environ.get("SPATIAL_HASH_SAFETY_GAP", "0.0"))
    step_scale = float(os.environ.get("SPATIAL_HASH_STEP_SCALE", "0.65"))
    max_step_frac = float(os.environ.get("SPATIAL_HASH_MAX_STEP_FRAC", "0.04"))
    anchor_pull = float(os.environ.get("SPATIAL_HASH_ANCHOR_PULL", "0.02"))

    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    max_step = max_step_frac * max(canvas_w, canvas_h)

    before_pairs, before_area, _ = _spatial_hash_count_overlaps(out, benchmark, hard_ids, safety_gap=0.0)
    print(
        f"[spatial_hash_legalizer] hard overlaps before: {len(before_pairs)} area={before_area:.6f}",
        flush=True,
    )

    for it in range(max_iters):
        pairs, area, _ = _spatial_hash_count_overlaps(out, benchmark, hard_ids, safety_gap=safety_gap)

        if not pairs:
            print(f"[spatial_hash_legalizer] converged at iter={it}", flush=True)
            break

        disp = {i: [0.0, 0.0] for i in hard_ids}
        counts = {i: 0 for i in hard_ids}

        for i, j, ox, oy, _area in pairs:
            xi = float(out[i, 0])
            yi = float(out[i, 1])
            xj = float(out[j, 0])
            yj = float(out[j, 1])

            mi = movable.get(i, True)
            mj = movable.get(j, True)

            if not mi and not mj:
                continue

            # Push along minimum overlap axis.
            if ox <= oy:
                sign = 1.0 if xi >= xj else -1.0
                push_x = sign * ox * step_scale
                push_y = 0.0
            else:
                sign = 1.0 if yi >= yj else -1.0
                push_x = 0.0
                push_y = sign * oy * step_scale

            wi = move_weight[i]
            wj = move_weight[j]

            if mi and mj:
                # Larger/important macros move less.
                total = wi + wj
                frac_i = wj / total
                frac_j = wi / total
            elif mi:
                frac_i = 1.0
                frac_j = 0.0
            else:
                frac_i = 0.0
                frac_j = 1.0

            if mi:
                disp[i][0] += push_x * frac_i
                disp[i][1] += push_y * frac_i
                counts[i] += 1

            if mj:
                disp[j][0] -= push_x * frac_j
                disp[j][1] -= push_y * frac_j
                counts[j] += 1

        moved = 0
        for i in hard_ids:
            if not movable.get(i, True):
                continue

            dx, dy = disp[i]
            if counts[i] > 0:
                dx /= counts[i]
                dy /= counts[i]

            # Mild anchor pull to reduce legalizer damage.
            dx += anchor_pull * (float(anchor[i, 0]) - float(out[i, 0]))
            dy += anchor_pull * (float(anchor[i, 1]) - float(out[i, 1]))

            norm = (dx * dx + dy * dy) ** 0.5
            if norm > max_step:
                scale = max_step / max(norm, 1e-12)
                dx *= scale
                dy *= scale

            if abs(dx) > 1e-12 or abs(dy) > 1e-12:
                out[i, 0] += dx
                out[i, 1] += dy
                moved += 1

        _spatial_hash_bounds_repair(out, benchmark, hard_ids)

        if it % 10 == 0 or it == max_iters - 1:
            cur_pairs, cur_area, _ = _spatial_hash_count_overlaps(out, benchmark, hard_ids, safety_gap=0.0)
            print(
                f"[spatial_hash_legalizer] iter={it} moved={moved} "
                f"hard_overlaps={len(cur_pairs)} area={cur_area:.6f}",
                flush=True,
            )

    after_pairs, after_area, _ = _spatial_hash_count_overlaps(out, benchmark, hard_ids, safety_gap=0.0)

    disp_vec = out[hard_ids] - anchor[hard_ids]
    disp_norm = torch.norm(disp_vec, dim=1)

    moved_count = int(torch.sum(disp_norm > 1e-9).item()) if len(hard_ids) else 0
    avg_disp = float(torch.mean(disp_norm).item()) if len(hard_ids) else 0.0
    max_disp = float(torch.max(disp_norm).item()) if len(hard_ids) else 0.0

    print(
        f"[spatial_hash_legalizer] after: hard_overlaps={len(after_pairs)} "
        f"area={after_area:.6f} moved={moved_count}/{len(hard_ids)} "
        f"avg_disp={avg_disp:.6f} max_disp={max_disp:.6f} "
        f"time={time.time() - t0:.3f}s",
        flush=True,
    )

    return out.to(original_device)




# =============================================================================
# Candidate-repair hard macro legalizer
# =============================================================================

def _candidate_repair_involved_macros(overlap_pairs):
    involved = set()
    for item in overlap_pairs:
        i, j = int(item[0]), int(item[1])
        involved.add(i)
        involved.add(j)
    return sorted(involved)


def _candidate_repair_candidate_positions(out, anchor, benchmark, macro_id, blockers, safety_gap):
    sizes = _spatial_hash_macro_sizes(benchmark)
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)

    x = float(out[macro_id, 0])
    y = float(out[macro_id, 1])
    ax = float(anchor[macro_id, 0])
    ay = float(anchor[macro_id, 1])

    w = float(sizes[macro_id, 0])
    h = float(sizes[macro_id, 1])

    # Candidate positions.
    cand = []

    def add(nx, ny, tag):
        xmin = w / 2.0
        xmax = canvas_w - w / 2.0
        ymin = h / 2.0
        ymax = canvas_h - h / 2.0

        nx = max(xmin, min(xmax, float(nx)))
        ny = max(ymin, min(ymax, float(ny)))

        cand.append((nx, ny, tag))

    # Current and anchor.
    add(x, y, "current")
    add(ax, ay, "anchor")

    # Small local nudges.
    base = max(canvas_w, canvas_h)
    for frac in [0.005, 0.01, 0.02, 0.04, 0.08]:
        step = frac * base
        add(x + step, y, f"R@{frac}")
        add(x - step, y, f"L@{frac}")
        add(x, y + step, f"U@{frac}")
        add(x, y - step, f"D@{frac}")

    # Positions just outside blockers.
    for b in blockers:
        bx = float(out[b, 0])
        by = float(out[b, 1])
        bw = float(sizes[b, 0])
        bh = float(sizes[b, 1])

        # Place macro just outside blocker in each direction.
        add(bx - (bw + w) / 2.0 - safety_gap, y, f"left_of_{b}")
        add(bx + (bw + w) / 2.0 + safety_gap, y, f"right_of_{b}")
        add(x, by - (bh + h) / 2.0 - safety_gap, f"below_{b}")
        add(x, by + (bh + h) / 2.0 + safety_gap, f"above_{b}")

        # Same outside moves but preserving the other coordinate closer to anchor.
        add(bx - (bw + w) / 2.0 - safety_gap, ay, f"left_of_{b}_anchor_y")
        add(bx + (bw + w) / 2.0 + safety_gap, ay, f"right_of_{b}_anchor_y")
        add(ax, by - (bh + h) / 2.0 - safety_gap, f"below_{b}_anchor_x")
        add(ax, by + (bh + h) / 2.0 + safety_gap, f"above_{b}_anchor_x")

    # Deduplicate.
    seen = set()
    out_cands = []
    for nx, ny, tag in cand:
        key = (round(nx, 9), round(ny, 9))
        if key in seen:
            continue
        seen.add(key)
        out_cands.append((nx, ny, tag))

    return out_cands


def _candidate_repair_score(out, anchor, benchmark, hard_ids, macro_id, nx, ny, safety_gap):
    old_x = float(out[macro_id, 0])
    old_y = float(out[macro_id, 1])

    out[macro_id, 0] = nx
    out[macro_id, 1] = ny

    pairs, area, _ = _spatial_hash_count_overlaps(out, benchmark, hard_ids, safety_gap=0.0)

    # Anchor displacement cost for this macro only.
    adx = float(out[macro_id, 0]) - float(anchor[macro_id, 0])
    ady = float(out[macro_id, 1]) - float(anchor[macro_id, 1])
    anchor_dist = (adx * adx + ady * ady) ** 0.5

    # Also penalize safety-gap near pairs lightly.
    near_pairs, near_area, _ = _spatial_hash_count_overlaps(out, benchmark, hard_ids, safety_gap=safety_gap)

    out[macro_id, 0] = old_x
    out[macro_id, 1] = old_y

    score = (
        100000.0 * len(pairs)
        + 1000.0 * float(area)
        + 10.0 * len(near_pairs)
        + 0.25 * float(anchor_dist)
    )

    return score, len(pairs), area, len(near_pairs), anchor_dist


def legalize_hard_macros_candidate_repair(placement, benchmark):
    """
    Stronger hard macro legalizer:
      1. spatial hash force pass
      2. candidate repair for unresolved overlap macros

    Goal:
      legalize hard macros faster than slow CPU pairwise/shelf path,
      while preserving the DREAMPlace anchor as much as possible.
    """
    t0 = time.time()

    original_device = placement.device

    # First do the fast force pass.
    out = legalize_hard_macros_spatial_hash(placement, benchmark).detach().clone().cpu()
    anchor = placement.detach().clone().cpu()

    hard_mask = benchmark.get_hard_macro_mask()
    if hasattr(hard_mask, "detach"):
        hard_mask = hard_mask.detach().cpu()

    hard_ids = [int(i) for i in torch.nonzero(hard_mask, as_tuple=False).flatten().tolist()]
    movable = {i: (not _spatial_hash_is_fixed_macro(benchmark, i)) for i in hard_ids}

    safety_gap = float(os.environ.get("CANDIDATE_REPAIR_SAFETY_GAP", "0.006"))
    max_rounds = int(os.environ.get("CANDIDATE_REPAIR_ROUNDS", "80"))
    max_macros_per_round = int(os.environ.get("CANDIDATE_REPAIR_MAX_MACROS_PER_ROUND", "32"))

    pairs, area, _ = _spatial_hash_count_overlaps(out, benchmark, hard_ids, safety_gap=0.0)
    print(
        f"[candidate_repair] start hard_overlaps={len(pairs)} area={area:.6f}",
        flush=True,
    )

    if not pairs:
        print(f"[candidate_repair] no repair needed time={time.time()-t0:.3f}s", flush=True)
        return out.to(original_device)

    for rnd in range(max_rounds):
        pairs, area, _ = _spatial_hash_count_overlaps(out, benchmark, hard_ids, safety_gap=0.0)

        if not pairs:
            print(f"[candidate_repair] converged round={rnd}", flush=True)
            break

        involved = _candidate_repair_involved_macros(pairs)
        involved = [i for i in involved if movable.get(i, True)]

        if not involved:
            print("[candidate_repair] no movable involved macros remain", flush=True)
            break

        # Prioritize macros that appear in many overlap pairs.
        degree = {i: 0 for i in involved}
        blockers = {i: set() for i in involved}
        for item in pairs:
            i, j = int(item[0]), int(item[1])
            if i in degree:
                degree[i] += 1
                blockers[i].add(j)
            if j in degree:
                degree[j] += 1
                blockers[j].add(i)

        involved.sort(key=lambda i: degree.get(i, 0), reverse=True)
        involved = involved[:max_macros_per_round]

        before_key = (len(pairs), float(area))
        best_global = None

        for i in involved:
            candidates = _candidate_repair_candidate_positions(
                out=out,
                anchor=anchor,
                benchmark=benchmark,
                macro_id=i,
                blockers=sorted(blockers.get(i, [])),
                safety_gap=safety_gap,
            )

            best = None
            for nx, ny, tag in candidates:
                score, overlap_count, overlap_area, near_count, anchor_dist = _candidate_repair_score(
                    out=out,
                    anchor=anchor,
                    benchmark=benchmark,
                    hard_ids=hard_ids,
                    macro_id=i,
                    nx=nx,
                    ny=ny,
                    safety_gap=safety_gap,
                )

                item = (score, overlap_count, overlap_area, near_count, anchor_dist, i, nx, ny, tag)
                if best is None or item < best:
                    best = item

            if best is not None:
                if best_global is None or best < best_global:
                    best_global = best

        if best_global is None:
            print("[candidate_repair] no candidate generated", flush=True)
            break

        score, overlap_count, overlap_area, near_count, anchor_dist, i, nx, ny, tag = best_global

        # Apply only if it improves lexicographic overlap state.
        if (overlap_count, overlap_area) < before_key:
            old_x = float(out[i, 0])
            old_y = float(out[i, 1])
            out[i, 0] = nx
            out[i, 1] = ny
            _spatial_hash_bounds_repair(out, benchmark, [i])

            print(
                f"[candidate_repair] round={rnd} macro={i} tag={tag} "
                f"old=({old_x:.6f},{old_y:.6f}) new=({nx:.6f},{ny:.6f}) "
                f"overlaps {before_key[0]}->{overlap_count} "
                f"area {before_key[1]:.6f}->{overlap_area:.6f} "
                f"near={near_count} anchor_dist={anchor_dist:.6f}",
                flush=True,
            )
        else:
            print(
                f"[candidate_repair] stalled round={rnd} "
                f"best_candidate_overlaps={overlap_count} area={overlap_area:.6f} "
                f"current_overlaps={before_key[0]} area={before_key[1]:.6f}",
                flush=True,
            )
            break

    final_pairs, final_area, _ = _spatial_hash_count_overlaps(out, benchmark, hard_ids, safety_gap=0.0)

    disp_vec = out[hard_ids] - anchor[hard_ids]
    disp_norm = torch.norm(disp_vec, dim=1)

    moved_count = int(torch.sum(disp_norm > 1e-9).item()) if len(hard_ids) else 0
    avg_disp = float(torch.mean(disp_norm).item()) if len(hard_ids) else 0.0
    max_disp = float(torch.max(disp_norm).item()) if len(hard_ids) else 0.0

    print(
        f"[candidate_repair] after hard_overlaps={len(final_pairs)} area={final_area:.6f} "
        f"moved={moved_count}/{len(hard_ids)} avg_disp={avg_disp:.6f} "
        f"max_disp={max_disp:.6f} time={time.time()-t0:.3f}s",
        flush=True,
    )

    return out.to(original_device)




# =============================================================================
# Anchor-preserving local Tetris / candidate-repair hard macro legalizer
# =============================================================================

def _lt_is_fixed_macro(benchmark, idx: int) -> bool:
    if "_is_fixed_macro" in globals():
        try:
            return bool(_is_fixed_macro(benchmark, idx))
        except Exception:
            pass
    try:
        return bool(benchmark.macro_fixed[idx])
    except Exception:
        return False


def _lt_sizes_cpu(benchmark):
    sizes = benchmark.macro_sizes
    if hasattr(sizes, "detach"):
        return sizes.detach().cpu()
    return sizes


def _lt_hard_ids(benchmark):
    hard_mask = benchmark.get_hard_macro_mask()
    if hasattr(hard_mask, "detach"):
        hard_mask = hard_mask.detach().cpu()
    return [int(i) for i in torch.nonzero(hard_mask, as_tuple=False).flatten().tolist()]


def _lt_bounds_repair_one(out, benchmark, i):
    sizes = _lt_sizes_cpu(benchmark)
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)

    w = float(sizes[i, 0])
    h = float(sizes[i, 1])

    xmin = w / 2.0
    xmax = canvas_w - w / 2.0
    ymin = h / 2.0
    ymax = canvas_h - h / 2.0

    out[i, 0] = max(xmin, min(xmax, float(out[i, 0])))
    out[i, 1] = max(ymin, min(ymax, float(out[i, 1])))


def _lt_overlap_pairs(out, benchmark, hard_ids, gap=0.0):
    sizes = _lt_sizes_cpu(benchmark)
    pairs = []
    total_area = 0.0

    for a in range(len(hard_ids)):
        i = hard_ids[a]
        xi = float(out[i, 0])
        yi = float(out[i, 1])
        wi = float(sizes[i, 0])
        hi = float(sizes[i, 1])

        for b in range(a + 1, len(hard_ids)):
            j = hard_ids[b]
            xj = float(out[j, 0])
            yj = float(out[j, 1])
            wj = float(sizes[j, 0])
            hj = float(sizes[j, 1])

            ox = (wi + wj) / 2.0 + gap - abs(xi - xj)
            oy = (hi + hj) / 2.0 + gap - abs(yi - yj)

            if ox > 0 and oy > 0:
                area = ox * oy
                pairs.append((i, j, ox, oy, area))
                total_area += area

    return pairs, total_area


def _lt_candidate_positions(out, anchor, benchmark, i, blockers, gap):
    sizes = _lt_sizes_cpu(benchmark)
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)

    x = float(out[i, 0])
    y = float(out[i, 1])
    ax = float(anchor[i, 0])
    ay = float(anchor[i, 1])
    w = float(sizes[i, 0])
    h = float(sizes[i, 1])

    base = max(canvas_w, canvas_h)
    candidates = []

    def add(nx, ny, tag):
        xmin = w / 2.0
        xmax = canvas_w - w / 2.0
        ymin = h / 2.0
        ymax = canvas_h - h / 2.0

        nx = max(xmin, min(xmax, float(nx)))
        ny = max(ymin, min(ymax, float(ny)))
        candidates.append((nx, ny, tag))

    # Always include current and anchor.
    add(x, y, "current")
    add(ax, ay, "anchor")

    # Small anchor-preserving nudges.
    for frac in [0.0025, 0.005, 0.01, 0.02, 0.04]:
        step = frac * base
        add(x + step, y, f"R@{frac}")
        add(x - step, y, f"L@{frac}")
        add(x, y + step, f"U@{frac}")
        add(x, y - step, f"D@{frac}")

        # also from anchor
        add(ax + step, ay, f"anchor_R@{frac}")
        add(ax - step, ay, f"anchor_L@{frac}")
        add(ax, ay + step, f"anchor_U@{frac}")
        add(ax, ay - step, f"anchor_D@{frac}")

    # Local Tetris positions just outside blockers.
    for b in blockers:
        bx = float(out[b, 0])
        by = float(out[b, 1])
        bw = float(sizes[b, 0])
        bh = float(sizes[b, 1])

        add(bx - (bw + w) / 2.0 - gap, y, f"left_of_{b}")
        add(bx + (bw + w) / 2.0 + gap, y, f"right_of_{b}")
        add(x, by - (bh + h) / 2.0 - gap, f"below_{b}")
        add(x, by + (bh + h) / 2.0 + gap, f"above_{b}")

        # Tetris move but preserve other coordinate from anchor.
        add(bx - (bw + w) / 2.0 - gap, ay, f"left_of_{b}_anchor_y")
        add(bx + (bw + w) / 2.0 + gap, ay, f"right_of_{b}_anchor_y")
        add(ax, by - (bh + h) / 2.0 - gap, f"below_{b}_anchor_x")
        add(ax, by + (bh + h) / 2.0 + gap, f"above_{b}_anchor_x")

    # Deduplicate.
    seen = set()
    result = []
    for nx, ny, tag in candidates:
        key = (round(nx, 9), round(ny, 9))
        if key in seen:
            continue
        seen.add(key)
        result.append((nx, ny, tag))
    return result


def _lt_score_candidate(out, anchor, benchmark, hard_ids, i, nx, ny, gap):
    old_x = float(out[i, 0])
    old_y = float(out[i, 1])

    out[i, 0] = nx
    out[i, 1] = ny
    _lt_bounds_repair_one(out, benchmark, i)

    pairs, area = _lt_overlap_pairs(out, benchmark, hard_ids, gap=0.0)
    near_pairs, near_area = _lt_overlap_pairs(out, benchmark, hard_ids, gap=gap)

    adx = float(out[i, 0]) - float(anchor[i, 0])
    ady = float(out[i, 1]) - float(anchor[i, 1])
    anchor_dist = (adx * adx + ady * ady) ** 0.5

    out[i, 0] = old_x
    out[i, 1] = old_y

    # Lexicographic-like scalar. Legal first, then area, then displacement.
    score = (
        1_000_000.0 * len(pairs)
        + 10_000.0 * float(area)
        + 10.0 * len(near_pairs)
        + 0.05 * float(anchor_dist)
    )
    return score, len(pairs), area, len(near_pairs), anchor_dist


def legalize_hard_macros_local_tetris(placement, benchmark):
    """
    Anchor-preserving local Tetris / candidate repair.

    Intended usage:
      run after DREAMPlace mapping.
      use GPU repulsion first to reduce broad overlaps.
      then repair only overlap components with local candidate moves.
    """
    t0 = time.time()

    original_device = placement.device
    out = placement.detach().clone().cpu()
    anchor = out.clone()

    hard_ids = _lt_hard_ids(benchmark)
    movable = {i: (not _lt_is_fixed_macro(benchmark, i)) for i in hard_ids}

    # First reduce broad overlaps with existing GPU repulsion.
    repulse_iters = int(os.environ.get("LOCAL_TETRIS_GPU_REPULSE_ITERS", "2000"))
    out = legalize_hard_macros_gpu_repulsion(out.to(original_device), benchmark, max_iters=repulse_iters)
    out = out.detach().clone().cpu()

    max_rounds = int(os.environ.get("LOCAL_TETRIS_ROUNDS", "80"))
    max_macros_per_round = int(os.environ.get("LOCAL_TETRIS_MAX_MACROS_PER_ROUND", "24"))
    gap = float(os.environ.get("LOCAL_TETRIS_GAP", "0.006"))

    pairs, area = _lt_overlap_pairs(out, benchmark, hard_ids, gap=0.0)
    print(f"[local_tetris] start overlaps={len(pairs)} area={area:.6f}", flush=True)

    for rnd in range(max_rounds):
        pairs, area = _lt_overlap_pairs(out, benchmark, hard_ids, gap=0.0)
        if not pairs:
            print(f"[local_tetris] converged round={rnd}", flush=True)
            break

        degree = {}
        blockers = {}
        for i, j, ox, oy, a in pairs:
            if movable.get(i, True):
                degree[i] = degree.get(i, 0) + 1
                blockers.setdefault(i, set()).add(j)
            if movable.get(j, True):
                degree[j] = degree.get(j, 0) + 1
                blockers.setdefault(j, set()).add(i)

        involved = sorted(degree, key=lambda k: degree[k], reverse=True)
        involved = involved[:max_macros_per_round]

        if not involved:
            print("[local_tetris] no movable overlap macros", flush=True)
            break

        before = (len(pairs), float(area))
        best = None

        for i in involved:
            cands = _lt_candidate_positions(out, anchor, benchmark, i, sorted(blockers.get(i, [])), gap)

            for nx, ny, tag in cands:
                item = _lt_score_candidate(out, anchor, benchmark, hard_ids, i, nx, ny, gap)
                score, overlap_count, overlap_area, near_count, anchor_dist = item
                candidate = (score, overlap_count, overlap_area, near_count, anchor_dist, i, nx, ny, tag)
                if best is None or candidate < best:
                    best = candidate

        if best is None:
            print("[local_tetris] no candidate found", flush=True)
            break

        score, overlap_count, overlap_area, near_count, anchor_dist, i, nx, ny, tag = best

        # Only apply if it improves overlap count or overlap area.
        if (overlap_count, overlap_area) < before:
            old_x = float(out[i, 0])
            old_y = float(out[i, 1])
            out[i, 0] = nx
            out[i, 1] = ny
            _lt_bounds_repair_one(out, benchmark, i)

            print(
                f"[local_tetris] round={rnd} macro={i} tag={tag} "
                f"overlaps {before[0]}->{overlap_count} area {before[1]:.6f}->{overlap_area:.6f} "
                f"anchor_dist={anchor_dist:.6f}",
                flush=True,
            )
        else:
            print(
                f"[local_tetris] stalled round={rnd} current_overlaps={before[0]} area={before[1]:.6f} "
                f"best_overlaps={overlap_count} best_area={overlap_area:.6f}",
                flush=True,
            )
            break

    final_pairs, final_area = _lt_overlap_pairs(out, benchmark, hard_ids, gap=0.0)

    disp = out[hard_ids] - anchor[hard_ids]
    disp_norm = torch.norm(disp, dim=1)
    moved = int(torch.sum(disp_norm > 1e-9).item()) if hard_ids else 0
    avg_disp = float(torch.mean(disp_norm).item()) if hard_ids else 0.0
    p95_disp = float(torch.quantile(disp_norm, 0.95).item()) if hard_ids else 0.0
    max_disp = float(torch.max(disp_norm).item()) if hard_ids else 0.0

    print(
        f"[local_tetris] after overlaps={len(final_pairs)} area={final_area:.6f} "
        f"moved={moved}/{len(hard_ids)} avg_disp={avg_disp:.6f} "
        f"p95_disp={p95_disp:.6f} max_disp={max_disp:.6f} "
        f"time={time.time()-t0:.3f}s",
        flush=True,
    )

    return out.to(original_device)




# =============================================================================
# Hard macro anchor rollback polish
# =============================================================================

def hard_macro_anchor_rollback_polish(placement, benchmark, anchor):
    """
    After hard macro legalization, try to move hard macros partially back toward
    their DREAMPlace anchor positions. Accept a rollback only if hard overlaps
    remain zero. This is a legality-gated proxy-preservation polish.
    """
    t0 = time.time()

    original_device = placement.device
    out = placement.detach().clone().cpu()
    anc = anchor.detach().clone().cpu()

    hard_mask = benchmark.get_hard_macro_mask()
    if hasattr(hard_mask, "detach"):
        hard_mask = hard_mask.detach().cpu()

    hard_ids = [int(i) for i in torch.nonzero(hard_mask, as_tuple=False).flatten().tolist()]

    if not hard_ids:
        return placement

    max_macros = int(os.environ.get("ANCHOR_ROLLBACK_MAX_MACROS", "80"))
    min_disp = float(os.environ.get("ANCHOR_ROLLBACK_MIN_DISP", "0.005"))
    fractions_raw = os.environ.get("ANCHOR_ROLLBACK_FRACTIONS", "0.25,0.5,0.75,1.0")
    fractions = [float(x) for x in fractions_raw.split(",") if x.strip()]

    # Start only from valid hard-overlap state.
    start_overlaps = count_hard_overlaps(out.to(original_device), benchmark)
    if start_overlaps != 0:
        print(
            f"[anchor_rollback] skip because start hard_overlaps={start_overlaps}",
            flush=True,
        )
        return placement

    disp = out[hard_ids] - anc[hard_ids]
    disp_norm = torch.norm(disp, dim=1)

    ranked = []
    for local_idx, macro_id in enumerate(hard_ids):
        d = float(disp_norm[local_idx].item())
        if d >= min_disp:
            # Avoid fixed macros when helper exists.
            fixed = False
            if "_is_fixed_macro" in globals():
                try:
                    fixed = bool(_is_fixed_macro(benchmark, macro_id))
                except Exception:
                    fixed = False
            if not fixed:
                ranked.append((d, macro_id))

    ranked.sort(reverse=True)
    ranked = ranked[:max_macros]

    accepted = 0
    tested = 0

    for _, i in ranked:
        cur_x = float(out[i, 0])
        cur_y = float(out[i, 1])
        anc_x = float(anc[i, 0])
        anc_y = float(anc[i, 1])

        best_pos = None
        best_dist = ((cur_x - anc_x) ** 2 + (cur_y - anc_y) ** 2) ** 0.5

        for frac in fractions:
            nx = cur_x + frac * (anc_x - cur_x)
            ny = cur_y + frac * (anc_y - cur_y)

            trial = out.clone()
            trial[i, 0] = nx
            trial[i, 1] = ny

            trial = final_bounds_repair(trial.to(original_device), benchmark).detach().cpu()

            tested += 1
            overlaps = count_hard_overlaps(trial.to(original_device), benchmark)

            if overlaps == 0:
                new_dist = ((float(trial[i, 0]) - anc_x) ** 2 + (float(trial[i, 1]) - anc_y) ** 2) ** 0.5

                if new_dist < best_dist:
                    best_dist = new_dist
                    best_pos = (float(trial[i, 0]), float(trial[i, 1]), frac)

        if best_pos is not None:
            old_dist = ((cur_x - anc_x) ** 2 + (cur_y - anc_y) ** 2) ** 0.5
            out[i, 0] = best_pos[0]
            out[i, 1] = best_pos[1]
            accepted += 1

            print(
                f"[anchor_rollback] macro={i} frac={best_pos[2]} "
                f"anchor_dist {old_dist:.6f}->{best_dist:.6f}",
                flush=True,
            )

    final_overlaps = count_hard_overlaps(out.to(original_device), benchmark)

    hard_disp = out[hard_ids] - anc[hard_ids]
    hard_norm = torch.norm(hard_disp, dim=1)

    avg_disp = float(torch.mean(hard_norm).item()) if len(hard_ids) else 0.0
    p95_disp = float(torch.quantile(hard_norm, 0.95).item()) if len(hard_ids) else 0.0
    max_disp = float(torch.max(hard_norm).item()) if len(hard_ids) else 0.0

    print(
        f"[anchor_rollback] tested={tested} accepted={accepted} "
        f"final_hard_overlaps={final_overlaps} "
        f"avg_disp={avg_disp:.6f} p95_disp={p95_disp:.6f} max_disp={max_disp:.6f} "
        f"time={time.time() - t0:.3f}s",
        flush=True,
    )

    if final_overlaps != 0:
        print("[anchor_rollback] rollback produced overlaps; reverting", flush=True)
        return placement

    return out.to(original_device)


class MyPlacer:
    def __init__(self):
        pass

    def place(self, benchmark):
        total_t0 = time.time()
        bench = infer_benchmark_name(benchmark)
        apply_benchmark_env_config(bench, benchmark)

        out_dir, meta = _profile_stage("export_bookshelf", export_bookshelf_via_tool, bench)
        placement = benchmark.macro_positions.clone()

        placement = benchmark.macro_positions.clone()

        gp_pl = _profile_stage("run_dreamplace", run_dreamplace, bench, out_dir)

        if gp_pl is None:
            print(
                f"[fallback] using benchmark initial placement for {bench}",
                flush=True,
            )
            placement = _profile_stage(
                "fallback_bounds_repair",
                final_bounds_repair,
                placement,
                benchmark,
            )
        else:
            gp_pos = _profile_stage("read_gp_pl", read_gp_pl, gp_pl)
            placement = _profile_stage(
                "map_dreamplace_positions",
                apply_dreamplace_positions_to_all_macros,
                placement,
                benchmark,
                meta,
                gp_pos,
            )

        dreamplace_anchor = placement.clone()
        fast_sweep_mode = os.environ.get("FAST_SWEEP_MODE", "0") == "1"

        if fast_sweep_mode:
            print("[fast_sweep] running GPU repulsion only; skipping pairwise/shelf hard legalizer", flush=True)

            fast_gpu_iters = int(os.environ.get("FAST_SWEEP_GPU_REPULSE_ITERS", "2000"))

            placement = _profile_stage(
                "fast_sweep_gpu_repulsion",
                legalize_hard_macros_gpu_repulsion,
                placement,
                benchmark,
                fast_gpu_iters,
            )

            fast_overlaps_after_gpu = count_hard_overlaps(placement, benchmark)
            print(
                f"[fast_sweep] hard overlaps after GPU repulsion: {fast_overlaps_after_gpu}",
                flush=True,
            )

            placement = _profile_stage(
                "fast_sweep_bounds_repair",
                final_bounds_repair,
                placement,
                benchmark,
            )
        else:
            if os.environ.get("USE_LOCAL_TETRIS_LEGALIZER", "0") == "1":
                placement = _profile_stage(
                    "local_tetris_hard_macro_legalizer",
                    legalize_hard_macros_local_tetris,
                    placement,
                    benchmark,
                )
            elif os.environ.get("USE_CANDIDATE_REPAIR_LEGALIZER", "0") == "1":
                placement = _profile_stage(
                    "candidate_repair_hard_macro_legalizer",
                    legalize_hard_macros_candidate_repair,
                    placement,
                    benchmark,
                )
            elif os.environ.get("USE_SPATIAL_HASH_LEGALIZER", "0") == "1":
                placement = _profile_stage(
                    "spatial_hash_hard_macro_legalizer",
                    legalize_hard_macros_spatial_hash,
                    placement,
                    benchmark,
                )
            else:
                placement = _profile_stage(
                    "hard_macro_legalizer",
                    legalize_hard_macros_post_dreamplace,
                    placement,
                    benchmark,
                )

        if fast_sweep_mode:
            print("[fast_sweep] skipping soft_dreamplace_refine", flush=True)
        else:
            placement = _profile_stage(
                "soft_dreamplace_refine",
                soft_dreamplace_refine,
                placement,
                benchmark,
                bench,
                out_dir,
                meta,
            )

        placement = _profile_stage(
            "legalization_displacement_audit",
            legalization_displacement_audit,
            dreamplace_anchor,
            placement,
            benchmark,
        )

        soft_iters = int(os.environ.get("SOFT_SPREAD_ITERS", "120"))
        placement = gpu_soft_macro_spread_post_dreamplace(
            placement, benchmark, iters=soft_iters
        )

        placement = _profile_stage("final_bounds_repair", final_bounds_repair, placement, benchmark)

        if fast_sweep_mode:
            print("[fast_sweep] skipping final_legality_margin_repair", flush=True)
        else:
            placement = _profile_stage(
                "final_legality_margin_repair",
                final_legality_margin_repair,
                placement,
                benchmark,
            )
        if (not fast_sweep_mode) and os.environ.get("USE_ANCHOR_ROLLBACK_POLISH", "0") == "1":
            placement = _profile_stage(
                "hard_macro_anchor_rollback_polish",
                hard_macro_anchor_rollback_polish,
                placement,
                benchmark,
                dreamplace_anchor,
            )

        placement = preserve_run_artifacts(bench, out_dir, placement)

        print(f"[profile] total placer time for {bench}: {time.time() - total_t0:.3f}s", flush=True)

        return placement
