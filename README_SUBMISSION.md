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

---

## Runtime Isolation Fix Diff Proof

The following is the exact code diff for the post-deadline runtime isolation fix. It shows that the change only makes the temporary run directory unique and clears stale per-benchmark DREAMPlace export folders.

```diff
diff --git a/submissions/final_placer/placer.py b/submissions/final_placer/placer.py
index f17c0e3..21a1a6f 100644
--- a/submissions/final_placer/placer.py
+++ b/submissions/final_placer/placer.py
@@ -1844,7 +1844,7 @@ class FinalPlacer:
         deadline = time.time() + timeout_sec
 
         root_base = Path(os.environ.get("FINAL_PLACER_ROOT", "/dev/shm/dreamplace_final_placer_runs"))
-        root = root_base / f"{bench}_{time.strftime('%Y%m%d_%H%M%S')}"
+        root = root_base / f"{bench}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{time.time_ns() % 1000000}"
         root.mkdir(parents=True, exist_ok=True)
 
         benchmark_root = ICCAD_ROOT / bench
@@ -1853,6 +1853,12 @@ class FinalPlacer:
         placer_mod = load_dreamplace_placer_module()
 
         print(f"[final_placer] bench={bench}", flush=True)
+        # Defensive --all isolation: remove stale DREAMPlace export for this benchmark.
+        try:
+            import shutil
+            shutil.rmtree(Path("dreamplace_ibm") / str(bench), ignore_errors=True)
+        except Exception as e:
+            print(f"[final_placer] warning: could not clean dreamplace_ibm/{bench}: {e!r}", flush=True)
         print(f"[final_placer] root={root}", flush=True)
         print(f"[final_placer] timeout_sec={timeout_sec}", flush=True)
 
```
