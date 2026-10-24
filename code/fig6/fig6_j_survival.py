"""
fig6_j_survival.py

Figure 6, panel j: Kaplan-Meier survival of the SHANK3 marmoset colony, as a
safety readout.

Data. survival_data.csv, covering the whole colony, including animals with no
Unique_ID in the master colony list. Neonatal deaths are excluded from the
input file. Health-related euthanasia counts as an event; planned terminal
study sacrifices are censored. Groups are WT, SHANK3 het (untreated) and
SHANK3 with gene therapy.

Statistics. Kaplan-Meier product-limit estimator: at each observed event time
survival is multiplied by 1 - d/at_risk, where d is the number of events at
that time and at_risk the number of animals still under observation. Censored
animals remain at risk up to their own censoring time.

Display. Three panels, one per group. Each panel draws its own survival curve
in the group color, with the other two groups behind it as faint reference
traces, event markers labeled by Unique_ID (or by name where no ID exists)
and censoring marked with a plus. Age is in years on the x axis and survival
probability in per cent on the y axis; each title gives the group size,
number of events and final survival.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from gtstyle import central_output_dir  # noqa: E402
from gtstyle import gt_init, gt_style  # noqa: E402

STY = gt_style()

DATA = os.path.join(HERE, "..", "..", "data", "fig6", "fig6_j_survival")

GROUPS = ["WT", "SHANK3_het", "SHANK3_GT"]
GROUP_LABELS = ["WT", "SH", "GT"]
COLORS = np.array([
    [0.420, 0.447, 0.502],
    [0.961, 0.620, 0.043],
    [0.231, 0.510, 0.965],
])


def _km(times, events):
    """Kaplan-Meier product-limit estimator. Returns the survival step
    function, the censoring times with their survival values, and the group
    size, event count and final survival."""
    order = np.argsort(times, kind="stable")
    times = times[order]
    events = events[order]
    event_times = np.unique(times[events == 1])

    surv = 1.0
    t_out = [0.0]
    s_out = [1.0]
    for t in event_times:
        at_risk = np.sum(times >= t)
        d = np.sum((times == t) & (events == 1))
        surv *= (1 - d / at_risk)
        t_out.append(t)
        s_out.append(surv)

    t_out = np.array(t_out)
    s_out = np.array(s_out)
    max_time = times.max()
    last_surv = surv

    censor_times = times[events == 0]
    censor_surv = np.zeros(censor_times.size)
    for j, ct in enumerate(censor_times):
        s = 1.0
        for k in range(t_out.size):
            if t_out[k] <= ct:
                s = s_out[k]
        censor_surv[j] = s

    return dict(time=t_out, surv=s_out, max_time=max_time, last_surv=last_surv,
               n=times.size, n_events=int(events.sum()),
               censor_times=censor_times, censor_surv=censor_surv)


def _stairs(km):
    t, s = km["time"], km["surv"]
    x, y = [], []
    for j in range(t.size):
        if j > 0:
            x.append(t[j]); y.append(s[j - 1])
        x.append(t[j]); y.append(s[j])
    x.append(km["max_time"]); y.append(km["last_surv"])
    return np.array(x), np.array(y)


def main(data_dir=DATA, out_dir=central_output_dir(HERE), show=False):
    gt_init()
    T = pd.read_csv(os.path.join(data_dir, "survival_data.csv"))

    def _fmt_uid(v):
        if pd.isna(v):
            return ""
        if isinstance(v, float) and v == int(v):
            return str(int(v))
        return str(v)

    T["Unique_ID"] = T["Unique_ID"].apply(_fmt_uid)

    km = {}
    for g in GROUPS:
        mask = T["Group"] == g
        times = T.loc[mask, "Duration_days"].to_numpy(float)
        events = T.loc[mask, "Event"].to_numpy(int)
        k = _km(times, events)
        k["event_uids"] = T.loc[mask & (T["Event"] == 1), "Unique_ID"].to_numpy()
        km[g] = k

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    y_limits = (0.75, 1.03)
    x_limits = (0, 2100)
    year_ticks = np.arange(0, 6) * 365.25

    for gi, g in enumerate(GROUPS):
        ax = axes[gi]
        for gj, g2 in enumerate(GROUPS):
            if g2 == g:
                continue
            xr, yr = _stairs(km[g2])
            # The other two groups are drawn faintly, as reference traces.
            ax.plot(xr, yr, "-", color=(*COLORS[gj], 0.2), lw=STY.lw.trace, zorder=1)

        xs, ys = _stairs(km[g])
        # This panel's own curve is the primary series.
        ax.plot(xs, ys, "-", color=COLORS[gi], lw=STY.lw.fit, zorder=3)
        ax.plot(km[g]["censor_times"], km[g]["censor_surv"], "+", color=COLORS[gi],
                ms=6, mew=1.2, zorder=4)

        if km[g]["n_events"] > 0:
            ev_idx = np.flatnonzero(km[g]["time"] > 0)
            et, es = km[g]["time"][ev_idx], km[g]["surv"][ev_idx]
            ax.plot(et, es, "o", ms=8, mfc=COLORS[gi], mec="w", lw=1.5, zorder=5)
            for j in range(et.size):
                if j < km[g]["event_uids"].size:
                    uid = km[g]["event_uids"][j]
                    lbl = f"#{uid}"
                    ax.text(et[j] + 40, es[j] - 0.012, lbl, fontsize=9, fontweight="bold",
                            color=COLORS[gi] * 0.7)

        for yr in range(1, 6):
            ax.axvline(yr * 365.25, ls=":", color=(0.92, 0.92, 0.92), zorder=0)

        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.xaxis.set_major_locator(FixedLocator(year_ticks))
        ax.xaxis.set_major_formatter(FixedFormatter(["0", "1", "2", "3", "4", "5"]))
        yticks = np.arange(0.75, 1.001, 0.05)
        ax.yaxis.set_major_locator(FixedLocator(yticks))
        ax.yaxis.set_major_formatter(FixedFormatter([f"{round(v*100)}%" for v in yticks]))
        ax.set_xlabel("Age (years)", fontsize=STY.font.lab)
        if gi == 0:
            ax.set_ylabel("Survival Probability", fontsize=STY.font.lab)
        s_txt = "s" if km[g]["n_events"] != 1 else ""
        ax.set_title(f"{GROUP_LABELS[gi]}\nn={km[g]['n']}, {km[g]['n_events']} event{s_txt}, "
                    f"S(t)={km[g]['last_surv']*100:.1f}%", fontsize=STY.font.title, color=COLORS[gi])
        # Uniform tick and body text size across the panel set.
        ax.tick_params(direction="out", labelsize=STY.font.tick)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.text(0.5, 0.01,
             "Neonatal deaths excluded. Health-related euthanasia counted as events. "
             "Planned terminal sacrifices censored. Faint lines = other groups for reference.",
             fontsize=7.5, color=(0.55, 0.55, 0.55), ha="center")
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    os.makedirs(out_dir, exist_ok=True)
    p1 = os.path.join(out_dir, "fig6_j_survival.png")
    fig.savefig(p1, dpi=300, facecolor="w")
    fig.savefig(p1.replace(".png", ".pdf"), facecolor="w")   # vector PDF for figure assembly
    print(f"Saved: {p1}")

    print(f"\n{'Group':<30s}  {'n':>5s}  {'Events':>6s}  {'S(t)':>10s}")
    print("-" * 55)
    for gi, g in enumerate(GROUPS):
        print(f"{GROUP_LABELS[gi]:<30s}  {km[g]['n']:5d}  {km[g]['n_events']:6d}  "
              f"{km[g]['last_surv']*100:9.1f}%")

    if not show:
        plt.close(fig)
    return fig


if __name__ == "__main__":
    main()
