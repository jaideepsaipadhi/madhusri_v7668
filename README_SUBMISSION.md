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
