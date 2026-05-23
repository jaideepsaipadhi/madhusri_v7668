# Post-deadline Runtime Isolation Note

After the deadline, I pushed one narrow runtime-stability fix for all-benchmark evaluation mode. The fix only isolates temporary benchmark run paths when multiple benchmarks are evaluated consecutively.

The issue: during `--all` evaluation, consecutive benchmarks could reuse stale DREAMPlace/export paths. In one test, an `ibm04` conversion attempted to read/write under an `ibm03` run root, causing a runtime path error before placement logic could complete.

The fix:
- Adds a unique process/time suffix to the temporary `/dev/shm/dreamplace_final_placer_runs/<bench>...` directory.
- Clears stale `dreamplace_ibm/<bench>` export folders at the start of each benchmark.

This does **not** change:
- placement search parameters,
- DREAMPlace target-density or density-weight configurations,
- candidate scoring logic,
- legalization logic,
- LNS strategy,
- final selection criteria,
- or any benchmark-specific score optimization.

It is intended only to make the submitted code run reliably in judge-style `--all` evaluation.

# HRT Macro Placement Challenge Submission

## Entry point

    uv run evaluate submissions/final_placer/placer.py --all

## Main submission files

- submissions/final_placer/placer.py
- submissions/final_placer/lns_engine.py
- submissions/final_placer/lns_adapter.py
- submissions/dreamplace_only/placer.py
- tools/pb_plc_to_bookshelf.py

## Algorithm summary

- DREAMPlace-based 27-config L1 portfolio
- Official scoring of all candidates
- Finalist selection from best valid / lowest-overlap invalid / lowest-proxy invalid candidates
- Quick GPU zero-overlap repair
- Old DREAMPlace hard-macro legalizer fallback
- Layer-1-only LNS polish with capped accepts/rejects

## Runtime target

Under 1 hour per benchmark.

## Environment notes

This submission uses DREAMPlace. In our verified environment:

    DREAMPLACE_ROOT=/workspace/DREAMPlace/install
    DREAMPLACE_PYTHON=/workspace/dreamplace_env/bin/python

The DREAMPlace Python environment must include:

    matplotlib scipy networkx

The intended evaluation command is:

    uv run evaluate submissions/final_placer/placer.py --all

## DREAMPlace dependency pin

DREAMPlace must run with NumPy 1.x. NumPy 2.x removes `np.string_`, which causes this DREAMPlace version to fail while loading Bookshelf benchmarks.

Install DREAMPlace Python dependencies with:

    python -m pip install -r requirements-dreamplace.txt

The important pin is:

    numpy<2

If installing `ncg_optimizer`, install it without letting it upgrade Torch:

    python -m pip install ncg_optimizer --no-deps

## DREAMPlace modern CUDA/CUB build note

On modern CUDA images, DREAMPlace may fail to compile because its old `utils_cub.cuh` wraps CUB inside the `DreamPlace::cub` namespace. If this happens, patch:

    DREAMPlace/dreamplace/ops/utility/src/utils_cub.cuh

to include CUB globally:

    #include "utility/src/namespace.h"
    #include <cub/cub.cuh>
    #define CUB_NS_QUALIFIER cub

and remove the old block:

    #define CUB_NS_PREFIX namespace DREAMPLACE_NAMESPACE {
    #define CUB_NS_POSTFIX }
    #define CUB_NS_QUALIFIER DREAMPLACE_NAMESPACE::cub
    #include "cub/cub.cuh"
    #undef CUB_NS_QUALIFIER
    #undef CUB_NS_POSTFIX
    #undef CUB_NS_PREFIX

This is a DREAMPlace build compatibility patch, not a change to the submitted placer logic.
