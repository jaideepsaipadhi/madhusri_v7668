#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import re
import os

SCALE = 1000


def safe_name(name: str) -> str:
    return name.replace("/", "__")


def normalize_orientation(orient: str) -> str:
    """
    DREAMPlace/Bookshelf orientation normalization.

    Challenge hard macros allow N, FN, FS, S.
    90-degree rotations are not allowed, so E/W-style rotations are converted to N.
    """
    if orient is None:
        return "N"

    orient = str(orient).strip().upper()

    allowed = {"N", "FN", "FS", "S"}
    if orient in allowed:
        return orient

    # Do not allow 90-degree rotations.
    return "N"


def split_pb_nodes(text: str):
    """Split text-format protobuf into complete node {...} blocks using brace counting."""
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
    cols = rows = None
    positions = {}

    for line in plc_path.read_text(errors="ignore").splitlines():
        if line.startswith("# Columns"):
            m = re.search(r"Columns\s*:\s*(\d+)\s+Rows\s*:\s*(\d+)", line)
            if m:
                cols = int(m.group(1))
                rows = int(m.group(2))

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
    if cols is None or rows is None:
        # fallback only; real IBM PLCs include this metadata
        cols, rows = 100, 100

    return width, height, cols, rows, positions


def classify_nodes(nodes):
    macros = {}
    ports = {}
    macro_pins = {}

    for n in nodes:
        attrs = n["attrs"]
        name = n["name"]
        typ = attrs.get("type", "")

        has_size = "width" in attrs and "height" in attrs
        is_macro_like = has_size and typ not in {"PORT", "MACRO_PIN", "HARD_MACRO_PIN", "SOFT_MACRO_PIN"}

        # In these IBM PB files, hard macros show type MACRO.
        # Soft macros / groups may not always have the exact same type label,
        # so size-bearing non-pin nodes are treated as movable macro-like objects.
        if is_macro_like:
            macros[name] = {
                "idx": n["idx"],
                "type": typ,
                "width": float(attrs.get("width", 0.01)),
                "height": float(attrs.get("height", 0.01)),
                "x": float(attrs.get("x", 0.0)),
                "y": float(attrs.get("y", 0.0)),
                "orientation": normalize_orientation(attrs.get("orientation", "N")),
            }

        elif typ == "PORT":
            ports[name] = {
                "idx": n["idx"],
                "x": float(attrs.get("x", 0.0)),
                "y": float(attrs.get("y", 0.0)),
                "side": attrs.get("side", ""),
            }

        elif typ.endswith("PIN") or "macro_name" in attrs:
            macro_pins[name] = {
                "idx": n["idx"],
                "macro_name": attrs.get("macro_name", ""),
                "x": float(attrs.get("x", 0.0)),
                "y": float(attrs.get("y", 0.0)),
                "x_offset": float(attrs.get("x_offset", 0.0)),
                "y_offset": float(attrs.get("y_offset", 0.0)),
                "type": typ,
            }

    return macros, ports, macro_pins


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


def build_nets(nodes, macros, ports, macro_pins):
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

    return nets


def write_bookshelf(bench, nodes, plc_w, plc_h, cols, rows, plc_positions, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    macros, ports, macro_pins = classify_nodes(nodes)
    nets = build_nets(nodes, macros, ports, macro_pins)

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
                f"{safe_name(name)} "
                f"{int(round(m['width'] * SCALE))} "
                f"{int(round(m['height'] * SCALE))}\n"
            )

        for name in ports:
            f.write(f"{safe_name(name)} 1 1 terminal\n")

    # .pl, lower-left DBU for movable macros; fixed DBU for ports
    with (out_dir / f"{bench}.pl").open("w") as f:
        f.write("UCLA pl 1.0\n\n")

        for name, m in macros.items():
            cx = m["x"]
            cy = m["y"]
            orient = m.get("orientation", "N")

            if m["idx"] in plc_positions:
                cx, cy, orient_from_plc, fixed = plc_positions[m["idx"]]
                if orient_from_plc != "-":
                    orient = normalize_orientation(orient_from_plc)

            x_ll = cx - m["width"] / 2.0
            y_ll = cy - m["height"] / 2.0
            f.write(
                f"{safe_name(name)} "
                f"{int(round(x_ll * SCALE))} "
                f"{int(round(y_ll * SCALE))} : {normalize_orientation(orient)}\n"
            )

        for name, p in ports.items():
            cx = p["x"]
            cy = p["y"]
            orient = "N"

            if p["idx"] in plc_positions:
                cx, cy, orient_from_plc, fixed = plc_positions[p["idx"]]
                if orient_from_plc != "-":
                    orient = normalize_orientation(orient_from_plc)

            f.write(
                f"{safe_name(name)} "
                f"{int(round(cx * SCALE))} "
                f"{int(round(cy * SCALE))} : {normalize_orientation(orient)} /FIXED\n"
            )

    # .scl using actual PLC grid dimensions
    scaled_w = int(round(plc_w * SCALE))
    scaled_h = int(round(plc_h * SCALE))
    row_height = max(1, scaled_h // rows)
    site_width = max(1, scaled_w // cols)

    with (out_dir / f"{bench}.scl").open("w") as f:
        f.write("UCLA scl 1.0\n\n")
        f.write(f"NumRows : {rows}\n\n")

        for r in range(rows):
            y = r * row_height
            f.write("CoreRow Horizontal\n")
            f.write(f"  Coordinate    : {y}\n")
            f.write(f"  Height        : {row_height}\n")
            f.write(f"  Sitewidth     : {site_width}\n")
            f.write(f"  Sitespacing   : {site_width}\n")
            f.write("  Siteorient    : N\n")
            f.write("  Sitesymmetry  : Y\n")
            f.write(f"  SubrowOrigin  : 0 NumSites : {cols}\n")
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
                    f"{int(round(xoff * SCALE))} "
                    f"{int(round(yoff * SCALE))}\n"
                )

    # DREAMPlace JSON: global placement only
    cfg = {
        "aux_input": str((out_dir / f"{bench}.aux").resolve()),
        "gpu": 1,
        "num_bins_x": int(os.environ.get("DREAMPLACE_NUM_BINS_X", "512")),
        "num_bins_y": int(os.environ.get("DREAMPLACE_NUM_BINS_Y", "512")),
        "global_place_stages": [
            {
                "num_bins_x": int(os.environ.get("DREAMPLACE_NUM_BINS_X", "512")),
                "num_bins_y": int(os.environ.get("DREAMPLACE_NUM_BINS_Y", "512")),
                "iteration": int(os.environ.get("DREAMPLACE_ITERATIONS", "1000")),
                "learning_rate": 0.01,
                "wirelength": "weighted_average",
                "optimizer": "nesterov",
                "Llambda_density_weight_iteration": 1,
                "Lsub_iteration": 1,
            }
        ],
        "target_density": float(os.environ.get("DREAMPLACE_TARGET_DENSITY", "1.0")),
        "density_weight": float(os.environ.get("DREAMPLACE_DENSITY_WEIGHT", "8e-5")),
        "random_seed": int(os.environ.get("DREAMPLACE_RANDOM_SEED", "1000")),
        "result_dir": str((out_dir / "results").resolve()),
        "scale_factor": 1.0,
        "ignore_net_degree": 100,
        "enable_fillers": 1,
        "global_place_flag": 1,
        "legalize_flag": 0,
        "abacus_legalize_flag": 0,
        "detailed_place_flag": 0,
        "plot_flag": 0,
        "num_threads": 8,
        "deterministic_flag": int(os.environ.get("DREAMPLACE_DETERMINISTIC", "1")),
    }

    (out_dir / f"{bench}.json").write_text(json.dumps(cfg, indent=2))

    meta = {
        "bench": bench,
        "scale": SCALE,
        "plc_width": plc_w,
        "plc_height": plc_h,
        "cols": cols,
        "rows": rows,
        "macro_names": macro_names,
        "macros": macros,
        "ports": ports,
        "macro_pins": macro_pins,
        "num_nets": len(nets),
    }

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"[OK] wrote {out_dir}")
    print(f"  macros/movable nodes: {len(macros)}")
    print(f"  ports/fixed terminals: {len(ports)}")
    print(f"  macro pins: {len(macro_pins)}")
    print(f"  nets: {len(nets)}")
    print(f"  grid: {cols} x {rows}")
    print(f"  canvas: {plc_w} x {plc_h}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--src-root", default="external/MacroPlacement/Testcases/ICCAD04")
    ap.add_argument("--out-root", default="dreamplace_ibm")
    args = ap.parse_args()

    bench = args.bench
    src = Path(args.src_root) / bench
    pb = src / "netlist.pb.txt"
    plc = src / "initial.plc"

    if not pb.exists():
        raise FileNotFoundError(pb)
    if not plc.exists():
        raise FileNotFoundError(plc)

    nodes = parse_netlist(pb)
    w, h, cols, rows, positions = parse_plc(plc)

    out_dir = Path(args.out_root) / bench
    write_bookshelf(bench, nodes, w, h, cols, rows, positions, out_dir)


if __name__ == "__main__":
    main()
