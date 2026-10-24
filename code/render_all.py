#!/usr/bin/env python3
"""
render_all.py -- render every figure panel to vector PDF and EPS.

    python render_all.py                 # every figure
    python render_all.py fig1 fig6       # selected figures
    python render_all.py --list          # list panels without rendering
    python render_all.py --timeout 120   # per-panel limit in seconds

This script discovers panels as the non-underscore .py files in each figure
directory and runs one per subprocess, so that the global state a panel sets
(matplotlib rcParams, the figure registry) cannot affect the next panel and a
failure or timeout in one does not end the run. Each panel writes its output
through gt_saveFig. The exit status is non-zero if any panel failed.

fig4/fig4_i_glmhmm.py and edfig7/edfig7_g_glmhmm.py plot a GLM-HMM fit that is
stored in data, so they render in seconds like every other panel.
GT_REFIT=1 refits from xy_type*.mat. The refit takes several minutes and
overwrites the stored fit.
"""
import argparse, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
ORDER = ["fig1","fig2","fig3","fig4","fig5","fig6"] + \
        [f"edfig{i}" for i in range(1, 15)]


def discover(dirs):
    """Return (directory, filename) for every panel script in dirs, sorted
    by name. Files beginning with an underscore are helpers."""
    out = []
    for d in dirs:
        full = os.path.join(ROOT, d)
        if not os.path.isdir(full):
            continue
        for f in sorted(os.listdir(full)):
            if f.endswith(".py") and not f.startswith("_"):
                out.append((d, f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()

    dirs = a.dirs or ORDER
    panels = discover(dirs)
    if a.list:
        for d, f in panels:
            print(f"{d}/{f}")
        print(f"\n{len(panels)} panels")
        return

    fails, t0 = [], time.time()
    for d, f in panels:
        s = time.time()
        try:
            p = subprocess.run([sys.executable, f], cwd=os.path.join(ROOT, d),
                               capture_output=True, text=True, timeout=a.timeout)
            ok = p.returncode == 0
            if not ok:
                fails.append((f"{d}/{f}", (p.stderr or "").strip().split("\n")[-1]))
        except subprocess.TimeoutExpired:
            ok = False
            fails.append((f"{d}/{f}", f"timeout after {a.timeout}s"))
        print(f"{'ok  ' if ok else 'FAIL'} {time.time()-s:6.1f}s  {d}/{f}", flush=True)

    print(f"\n{len(panels)-len(fails)}/{len(panels)} panels rendered "
          f"in {time.time()-t0:.0f}s")
    for name, err in fails:
        print(f"  FAIL {name}: {err}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
