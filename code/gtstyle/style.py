"""
gtstyle.style -- figure appearance constants shared by every panel: colors,
typography, line weights, marker sizes and pinned panel geometry.

gt_style() returns a nested attribute bag:

    S = gt_style()
    S.clr.wt          group color, wild type
    S.cmean[g]        group color by 0-based row index g
    S.font.tick       tick / body font size, points

Row order is WT, SH, GT throughout. Panel scripts index these arrays with the
group loop variable, so the order is load-bearing.
"""

from dataclasses import dataclass, field
import numpy as np


# ---------------------------------------------------------------------------
# Attribute bag: a nested namespace with dotted access, e.g. S.clr.wt.
# ---------------------------------------------------------------------------
class _Bag:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        keys = ", ".join(sorted(self.__dict__))
        return f"<{keys}>"


def gt_style():
    """Return the style bag holding every appearance constant used by the
    panel scripts."""
    S = _Bag()

    # ---- group colors --------------------------------------------------
    # WT is 20% gray rather than true black: it stays legible against the
    # pale bar fill and reproduces on non-white stock.
    clr = _Bag()
    clr.wt = np.array([0.200, 0.200, 0.200])
    clr.sh = np.array([0.753, 0.098, 0.145])
    clr.gt = np.array([0.286, 0.278, 0.729])

    # Near-black, used only for the GT-versus-WT contrast series in the
    # Figure 6 family. Distinct from clr.wt.
    clr.gtwt = np.array([0.15, 0.15, 0.15])

    # Three progressively paler tiers of the same hues, not interchangeable:
    #   scat    -- individual data points over a bar or box
    #   fill    -- mid-pale, for bar/box bodies that carry points on top
    #   barfill -- very pale, for bar bodies in dense multi-group panels
    clr.scat = np.array([[0.760, 0.760, 0.760],
                         [0.889, 0.594, 0.615],
                         [0.679, 0.675, 0.878]])
    clr.fill = np.array([[0.856, 0.856, 0.856],
                         [0.956, 0.838, 0.846],
                         [0.871, 0.870, 0.951]])
    clr.barfill = np.array([[0.936, 0.936, 0.936],
                            [0.980, 0.928, 0.932],
                            [0.943, 0.942, 0.978]])

    # ---- axis chrome ----------------------------------------------------
    # Dark gray rather than pure black for axis lines and ticks: at 0.6 pt a
    # black frame competes with the data for attention.
    clr.axis = np.array([0.15, 0.15, 0.15])
    clr.gray_hair = np.array([0.72, 0.72, 0.72])   # thin connector / hairline
    clr.gray_mid = np.array([0.45, 0.45, 0.45])
    clr.gray_lite = np.array([0.88, 0.88, 0.88])
    clr.gray_ref = np.array([0.60, 0.60, 0.60])    # reference / target lines
    clr.ns_txt = np.array([0.45, 0.45, 0.45])      # "ANOVA n/s" and similar
    clr.n_txt = np.array([0.55, 0.55, 0.55])       # the "n=" line under x axis
    clr.hilite = np.array([0.98, 0.90, 0.62])      # cog4 highlight band (fig6)
    clr.hilite_tx = np.array([0.55, 0.36, 0.02])

    # ---- semantic pairs -------------------------------------------------
    # Non-group encodings, shared by the Figure 3 and Figure 4 panels and
    # their Extended Data counterparts.
    #
    # This module defines two distinct green/magenta encodings: toward/away
    # uses the muted pair, approach/avoid the saturated pair. They label
    # different quantities and are not interchangeable.
    clr.toward = np.array([0.20, 0.70, 0.35])
    clr.away = np.array([0.55, 0.30, 0.80])
    clr.pos = np.array([0.20, 0.70, 0.35])         # alias of toward
    clr.neg = np.array([0.55, 0.30, 0.80])         # alias of away
    clr.pairline = np.array([0.20, 0.20, 0.20])
    clr.approach = np.array([0.00, 0.80, 0.00])
    clr.avoid = np.array([0.80, 0.00, 0.80])
    clr.neutral = np.array([0.80, 0.80, 0.80])
    # GLM-HMM colors: purple encodes bias, green encodes state usage.
    clr.bias = np.array([0.588, 0.271, 0.722])     # purple
    clr.usage = np.array([0.282, 0.675, 0.431])    # green
    clr.pt_gray = np.array([0.70, 0.70, 0.70])     # neutral data point
    S.clr = clr

    # Row-indexed forms (0-based rows: 0=WT, 1=SH, 2=GT).
    S.cmean = np.vstack([clr.wt, clr.sh, clr.gt])
    S.cscat = clr.scat
    S.cfill = clr.fill
    S.cbarfill = clr.barfill
    S.labels = ["WT", "SH", "GT"]

    S.alpha = _Bag(band=0.20, trace=0.15)

    # ---- typography -----------------------------------------------------
    # One type scale for every panel in the paper. Sizes are in points.
    S.font = _Bag(
        name="Arial",
        tick=7,         # body text and tick numbers
        lab=7,          # axis labels
        title=8,        # panel titles
        star=8,         # significance asterisks
        ns=6,           # "n/s" text, one step below body
        n=6,            # "n=" annotations
        letter=12,      # panel letter
        leg=6,          # legend text
    )

    # ---- line weights, in points ----------------------------------------
    S.lw = _Bag(
        axis=0.6,
        bar=1.0,        # bar outline
        se=0.9,         # SEM whisker and cap
        ci=0.9,         # forest-plot CI
        mean=1.5,       # mean / median marker rule
        sig=0.5,        # significance bracket
        hair=0.5,       # connectors, reference lines
        thin=0.5,
        trace=0.4,      # individual-subject traces
        pair=0.6,       # paired-data connecting lines
    )

    # ---- marker sizes ---------------------------------------------------
    # Scatter sizes are AREA in points^2, not diameter. Every individual data
    # point (swarm point, correlation dot, paired-plot point) uses S.mk.pt,
    # and every mean or median marker uses S.mk.mean, so a data point is the
    # same size in every figure. 6.0 pt^2 is a diameter of about 2.8 pt:
    # visible in a sparse scatter, compact in a dense swarm.
    S.mk = _Bag(swarm=6.0, pt=6.0, mean=20, leg=30, alpha=0.55, ew=0.7)

    # Weight of regression and model-fit lines on the correlation panels.
    S.lw.fit = 1.0

    # ---- tick geometry, in points (72 pt = 1 in) ------------------------
    # len = 2.7 pt (0.0375 in), shorter than matplotlib's 3.5 pt default:
    # these axes are only about 1.0 in wide, so a shorter tick returns width
    # to the plotting area while staying clearly longer than the 0.6 pt axis
    # line is thick. pad = 1.5 pt is the tick-to-label gap; at zero the digits
    # touch the tick and the axis reads as a smear at print size.
    S.tick = _Bag(len=2.7, pad=1.5, width=None)   # width=None -> use lw.axis

    # ---- panel geometry -------------------------------------------------
    # bar_w is in DATA units, so the printed bar width is
    # bar_w * (axes width in inches / x-range). ax_w is therefore fixed:
    # changing the axes width would change how wide the bars print.
    geom = _Bag()
    geom.bar_w = 0.66
    geom.swarm_w = 0.20
    geom.cap_hw = 0.09          # SEM cap half-width, data units

    # Title offset above the axes, in points. Larger than matplotlib's 6.0
    # default because the top significance bracket's 'n/s' label is drawn
    # with clipping off and rises above the axes frame into the title band;
    # 11 pt clears it.
    geom.title_pad = 11.0

    # Pinned three-group bar panel. xlim is [-0.6, 2.6] = 3.2 x-units at
    # 0.319 inches per x-unit, giving ax_w = 1.0208 in.
    geom.in_per_xunit = 0.319
    geom.xspan = 3.2
    geom.xlim3 = (-0.6, 2.6)
    geom.ax_w = geom.xspan * geom.in_per_xunit      # 1.0208 in
    geom.ax_h = 1.17
    geom.margin_l = 0.70        # y-label + tick labels
    geom.margin_r = 0.10
    geom.margin_b = 0.55        # x tick labels + the "n=" line
    geom.margin_t = 0.28        # title

    geom.fig_w = geom.margin_l + geom.ax_w + geom.margin_r   # 1.8208
    geom.fig_h = geom.margin_b + geom.ax_h + geom.margin_t   # 2.0000

    # Nature Neuroscience column widths, for sizing multi-panel assemblies.
    geom.col_single_mm = 89
    geom.col_double_mm = 183
    geom.mm_per_in = 25.4
    S.geom = geom

    return S


def gt_ablend(rgb, a):
    """Blend an RGB color toward white.

    rgb: length-3 sequence in [0, 1]. a: blend factor, 1 = full color,
    0 = white. Returns a length-3 array.

    Used in place of alpha wherever the result must survive EPS export: EPS
    has no alpha channel, so a genuinely transparent patch either forces the
    whole figure to rasterize or drops out. Pre-blending against white gives
    the same appearance on a white background, in pure vector art.
    """
    rgb = np.asarray(rgb, dtype=float)
    return a * rgb + (1 - a) * np.ones(3)
