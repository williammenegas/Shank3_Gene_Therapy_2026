"""
gtstyle.plotting -- drawing helpers shared by the figure panel scripts.

Contents
--------
    gt_init                 global rcParams and font resolution
    gt_beeswarm             deterministic center-out swarm offsets
    gt_jitter_binned        density-scaled random horizontal jitter
    gt_ticks_matlab         restricted "nice" tick-step locator
    gt_axes, gt_figStyle    axis and figure-wide typography
    gt_panelLetter          bold panel letter outside the axes
    gt_drawSig              significance bracket with a star or 'n/s' label
    gt_saveFig              vector PDF/EPS export at exact physical size
    central_output_dir      figure-code directory -> central output directory
    gt_barpanel_figure      pinned three-group bar figure and axes
    gt_bar3                 bar + swarm + SEM panel body
    gt_offscale_note        annotation for a capped y axis
    gt_boxscatter_figure    figure geometry for the box + scatter family
    gt_colorboxplot         colored box + quartiles + jittered scatter
    gt_boxsig               significance brackets for a box + scatter panel
    gt_meanerror            mean and error-bar dot plot
    gt_color_xticklabels    color WT/SH/GT tick labels in the group colors

Axes boxes are pinned in inches. matplotlib's fig.add_axes takes
figure-relative units, so panel geometry is divided through by the figure
size to give the intended physical dimensions. Scatter marker sizes are areas
in points^2 throughout.
"""

import os
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches
from matplotlib import font_manager, ticker

from .style import gt_style, gt_ablend
from .stats import gt_compare3, gt_compare3_np, gt_pstar

_FONT_WARNED = False


# ---------------------------------------------------------------------------
# Global rcParams: exported art stays true vector and text stays as text.
# ---------------------------------------------------------------------------
def gt_init(S=None):
    """Set the rcParams every panel depends on, and return the style bag.

    Call once before drawing. S: an existing style bag, or None to build one
    with gt_style().

    pdf.fonttype and ps.fonttype = 42 keep glyphs as editable TrueType text
    in the exported PDF and EPS instead of converting them to outlines, so
    labels remain editable in Illustrator. The assembly scripts rely on
    this. svg.fonttype = 'none' does the same for SVG.
    """
    if S is None:
        S = gt_style()

    # Font fallback chain. Arial is the target face. Liberation Sans and
    # Nimbus Sans are metric-compatible with Arial (identical advance
    # widths), so a panel laid out against either lands identically when
    # rendered with Arial itself. DejaVu Sans is matplotlib's default and is
    # NOT metric-compatible; it is last so that rendering never fails
    # outright, and a figure that falls through to it has wider labels. See
    # the warning emitted below.
    matplotlib.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": [S.font.name, "Arial", "Liberation Sans",
                            "Helvetica", "Nimbus Sans", "DejaVu Sans"],
        "font.size": S.font.tick,
        # axes.titlesize defaults to 'large' (1.2x base = 8.4 pt), which a
        # bare ax.set_title() re-applies on every call, overriding any
        # earlier set_fontsize() from gt_axes. Pinned here so a bare
        # set_title lands on the 8 pt title size with no per-call argument.
        "axes.titlesize": S.font.title,
        "axes.labelsize": S.font.lab,
        "xtick.labelsize": S.font.tick,
        "ytick.labelsize": S.font.tick,
        # Tick geometry is pinned here as well as in gt_axes, so panels that
        # build their own axes without calling gt_axes still get the shared
        # length and pad rather than matplotlib's 3.5 pt defaults.
        "xtick.major.size": S.tick.len,
        "ytick.major.size": S.tick.len,
        "xtick.minor.size": S.tick.len * 0.6,
        "ytick.minor.size": S.tick.len * 0.6,
        "xtick.major.pad": S.tick.pad,
        "ytick.major.pad": S.tick.pad,
        "xtick.major.width": S.lw.axis,
        "ytick.major.width": S.lw.axis,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.unicode_minus": False,   # ASCII hyphen-minus in tick labels
        "figure.facecolor": "w",
        "savefig.facecolor": "w",
    })

    # matplotlib emits one findfont warning per text object when Arial is
    # absent, which buries the panel's printed statistics. Suppress the
    # per-object messages and report the substitution once instead.
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    # Resolve S.font.name to an installed face rather than relying on the
    # rcParams fallback chain. Every panel and every gt_* helper passes
    # fontname=S.font.name explicitly, and when a Text object names a family
    # findfont cannot resolve, matplotlib does not walk
    # rcParams['font.sans-serif']. It falls straight through to DejaVu
    # Sans. DejaVu is not metric-compatible with Arial, so its wider labels
    # would be clipped by axes pinned in inches and would shift the panel
    # letters. Rewriting S.font.name to the best available
    # metric-compatible face preserves the layout.
    global _FONT_WARNED
    avail = {f.name for f in font_manager.fontManager.ttflist}
    if S.font.name not in avail:
        sub = next((n for n in ("Liberation Sans", "Nimbus Sans", "Helvetica",
                                "Arimo") if n in avail), None)
        if sub is None:
            sub = "DejaVu Sans"
        S.font.name = sub
        matplotlib.rcParams["font.sans-serif"] = (
            [sub] + [f for f in matplotlib.rcParams["font.sans-serif"] if f != sub]
        )
        if not _FONT_WARNED:
            _FONT_WARNED = True
            if sub == "DejaVu Sans":
                print("[gtstyle] WARNING: Arial not installed and no "
                      "metric-compatible substitute found; using DejaVu Sans. "
                      "Label widths WILL differ from the published figures. "
                      "Install fonts-liberation, or generate final art on a "
                      "machine with Arial.")
            else:
                print(f"[gtstyle] Arial not installed; using {sub} "
                      f"(metric-compatible - layout preserved). Generate "
                      f"final submission art on a machine with real Arial.")
    return S


# ---------------------------------------------------------------------------
# gt_beeswarm
# ---------------------------------------------------------------------------
def gt_beeswarm(v, center=0.0, halfw=0.20):
    """Symmetric center-out beeswarm offsets for a 1-D sample.

    v: values to be plotted. center: x coordinate of the swarm axis.
    halfw: maximum horizontal offset, in data units.

    Values are binned along y; within each bin the points fan out from the
    center alternately (0, +1, -1, +2, -2, ...), so the horizontal offset
    encodes local density only and never the value or rank of the point.
    Ordering the offsets by value instead would couple x to y and render as a
    diagonal stagger within each bin, which reads as structure that is not in
    the data.

    Returns an array the same length as v, in the original point order.
    """
    v = np.asarray(v, dtype=float).ravel()
    n = v.size
    xs = np.zeros(n)
    if n == 0:
        return xs

    # Bin count grows as sqrt(n): fine enough to resolve local density,
    # coarse enough that isolated points do not each claim their own bin.
    nb = max(6, int(round(2.2 * np.sqrt(n))))

    lo, hi = np.min(v), np.max(v)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        # All points identical (or a single point): nothing to spread.
        return xs + center

    edges = np.linspace(lo, hi + np.spacing(1.0), nb + 1)

    # Right-open intervals with the last bin closed; indices clamped to the
    # valid range at both ends.
    b = np.digitize(v, edges) - 1        # 0-based
    b[b < 0] = 0
    b[b > nb - 1] = nb - 1

    counts = np.bincount(b, minlength=nb)
    kmax = counts.max()

    for bi in range(nb):
        idx = np.flatnonzero(b == bi)
        k = idx.size
        if k == 0:
            continue

        # Denser bins spread wider, but sub-linearly, so a single crowded bin
        # does not blow the swarm out to the full half-width.
        sp = halfw * np.sqrt(k / max(kmax, 1))

        # Center-out ranks 0,1,2,... then signed so pairs straddle the
        # center: 0, +1, -1, +2, -2, ...
        r = np.arange(k)
        s = ((-1.0) ** r) * np.ceil(r / 2)
        if k > 1:
            denom = np.max(np.abs(s))
            off = s / denom * sp if denom > 0 else np.zeros(k)
        else:
            off = np.zeros(1)

        # Assigned in the bin's original index order, not sorted by value.
        xs[idx] = off

    return xs + center


# ---------------------------------------------------------------------------
# gt_axes / gt_figStyle
# ---------------------------------------------------------------------------
def gt_ticks_matlab(ax=None, axis="both", nbins=7):
    """Apply the shared tick locator: steps restricted to 1, 2, 5 and 10.

    ax: axes to modify, or None for the current axes. axis: 'x', 'y' or
    'both'. nbins: target number of intervals.

    matplotlib's default MaxNLocator also admits a step of 2.5, which on the
    small axes used here places the topmost tick short of the data ceiling.
    Restricting the step set keeps ticks on round multiples.

    nbins is fixed rather than 'auto' because 'auto' scales the tick count to
    the physical length of the axes, and these axes are only about 1.17 in
    tall, which collapses a panel to three ticks. A categorical x axis (one
    with a FixedLocator already set by the panel) is left untouched.

    Called from gt_axes, so panels receive it automatically.
    """
    if ax is None:
        ax = plt.gca()
    loc = lambda: ticker.MaxNLocator(nbins=nbins, steps=[1, 2, 5, 10])
    if axis in ("both", "y"):
        ax.yaxis.set_major_locator(loc())
    if axis in ("both", "x"):
        # Leave a categorical x axis alone.
        if not isinstance(ax.xaxis.get_major_locator(), ticker.FixedLocator):
            ax.xaxis.set_major_locator(loc())
    return ax


def gt_axes(ax=None, S=None, matlab_ticks=True):
    """Apply the shared axis styling to ax.

    Hides the top and right spines, sets ticks outward, draws the remaining
    spines and ticks at 0.6 pt in dark gray, and applies the 7 pt label and
    8 pt title sizes. Labels and titles do not inherit font size from the
    axes, so gt_axes sets both explicitly.

    matlab_ticks=True also applies gt_ticks_matlab to the tick locators. Pass
    False for a panel that sets its own tick positions, such as a categorical
    or matrix axis.
    """
    if ax is None:
        ax = plt.gca()
    if S is None:
        S = gt_style()

    if matlab_ticks:
        gt_ticks_matlab(ax)

    axc = tuple(S.clr.axis)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(S.lw.axis)
        ax.spines[side].set_color(axc)

    ax.tick_params(direction="out", width=S.lw.axis, colors=axc,
                   labelsize=S.font.tick,
                   length=S.tick.len, pad=S.tick.pad)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontname(S.font.name)
        lbl.set_fontsize(S.font.tick)
        lbl.set_color(axc)

    # Labels and title do not inherit font size from the axes.
    ax.xaxis.label.set_fontname(S.font.name)
    ax.xaxis.label.set_fontsize(S.font.lab)
    ax.yaxis.label.set_fontname(S.font.name)
    ax.yaxis.label.set_fontsize(S.font.lab)
    ax.title.set_fontname(S.font.name)
    ax.title.set_fontsize(S.font.title)

    ax.set_axisbelow(False)     # axis lines drawn over the data
    ax.grid(False)
    return ax


def gt_figStyle(fig=None, S=None):
    """Set the figure font on every text object in fig.

    Font family does not propagate to every artist type, so a single object
    created outside gt_axes can carry a fallback face into the exported art.
    Call immediately before gt_saveFig.
    """
    if fig is None:
        fig = plt.gcf()
    if S is None:
        S = gt_style()

    fig.patch.set_facecolor("w")
    for t in fig.findobj(matplotlib.text.Text):
        t.set_fontname(S.font.name)
    return fig


# ---------------------------------------------------------------------------
# gt_panelLetter / gt_drawSig
# ---------------------------------------------------------------------------
def gt_panelLetter(ax, letter, S=None, lower=True, dx=-0.02, dy=1.10, fs=None):
    """Draw the bold panel letter just outside the top-left of the axes.

    letter: the panel letter; an empty string or None draws nothing.
    dx, dy: position in axes-relative coordinates. fs: font size in points,
    or None for S.font.letter. lower: lowercase the letter (the default;
    Nature sets panel letters lowercase bold).

    The text is right-aligned at dx rather than left-aligned, which pins the
    letter's right edge: left-aligning at a fixed negative offset would make
    the gap between letter and axes depend on the glyph, so 'i' and 'm' would
    sit at different distances across an assembled figure.

    Returns the Text artist.
    """
    if S is None:
        S = gt_style()
    if letter is None or letter == "":
        return None
    if fs is None:
        fs = S.font.letter
    if lower:
        letter = str(letter).lower()

    return ax.text(dx, dy, letter,
                   transform=ax.transAxes,
                   fontname=S.font.name, fontweight="bold", fontsize=fs,
                   ha="right", va="center", color="k",
                   clip_on=False)


def gt_drawSig(ax, x1, x2, y, h, txt, S=None):
    """Draw a significance bracket from x1 to x2 at height y, labeled txt.

    h: bracket height in data units, used to offset the label. txt: the label
    from gt_pstar, typically a run of asterisks or 'n/s'.

    Asterisks and text labels are positioned differently: a run of '*' glyphs
    sits high in its em box and is therefore nudged down onto the bracket,
    whereas a text label is set one size smaller and sits above it.
    """
    if S is None:
        S = gt_style()

    INSET = 0.10
    a = x1 + INSET
    b = x2 - INSET

    ax.plot([a, b], [y, y], "-", color="k", linewidth=S.lw.sig,
            solid_capstyle="butt", clip_on=False)

    is_star = bool(txt) and all(ch == "*" for ch in txt)
    if is_star:
        dy, fs = -0.9 * h, S.font.star
    else:
        dy, fs = 0.4 * h, S.font.ns

    ax.text((a + b) / 2, y + dy, txt,
            ha="center", va="baseline",
            fontsize=fs, fontname=S.font.name, clip_on=False)


# ---------------------------------------------------------------------------
# gt_saveFig
# ---------------------------------------------------------------------------
def central_output_dir(path, name=None):
    """Map a panel's code directory to the repository's figure output tree:

        <repo>/code/figN[/...]    ->  <repo>/figures/figN
        <repo>/code/edfigN[/...]  ->  <repo>/supplemental_figures/edfigN

    The repository root is the parent of the `code` directory, so the mapping
    holds wherever the repository is checked out, rather than depending on any
    one directory name above it.

    A path with no `code/figN` or `code/edfigN` component is returned
    unchanged, so a script that passes an explicit output directory elsewhere
    is unaffected. `name` is accepted for call compatibility and is not used:
    the code directory alone determines the destination.
    """
    import re
    from pathlib import Path
    p = Path(path).resolve()
    parts = p.parts
    if "code" not in parts:
        return str(path)
    ci = len(parts) - 1 - list(reversed(parts)).index("code")   # last "code"
    repo = Path(*parts[:ci]) if ci > 0 else Path(p.anchor)
    for comp in parts[ci + 1:]:
        if re.fullmatch(r"edfig\d+", comp):
            return str(repo / "supplemental_figures" / comp)
        if re.fullmatch(r"fig\d+", comp):
            return str(repo / "figures" / comp)
    return str(path)


def gt_saveFig(fig, name, outdir=None, also_pdf=True, also_eps=True):
    """Export a figure as vector art at its exact physical size.

    name: output stem, or a path whose directory is used when outdir is None.
    outdir: destination directory; routed through central_output_dir.
    also_pdf, also_eps: which formats to write. Returns the list of paths
    written.

    bbox_inches=None is passed explicitly so that an rcParams default of
    'tight' cannot crop the output to its drawn content, which would discard
    the pinned panel geometry.

    EPS has no alpha channel, so a panel drawn with transparency is faithful
    only in the PDF; use gt_ablend rather than alpha where both formats must
    match.
    """
    if outdir is None:
        p, n = os.path.split(name)
        stem, ext = os.path.splitext(n)
        if ext.lower() in (".eps", ".pdf"):
            name = stem
        else:
            name = n
        outdir = p if p else os.getcwd()

    outdir = central_output_dir(outdir, name)
    os.makedirs(outdir, exist_ok=True)
    written = []

    if also_eps:
        fpath = os.path.join(outdir, name + ".eps")
        fig.savefig(fpath, format="eps", bbox_inches=None, facecolor="w")
        print(f"  saved {fpath}")
        written.append(fpath)

    if also_pdf:
        ppath = os.path.join(outdir, name + ".pdf")
        fig.savefig(ppath, format="pdf", bbox_inches=None, facecolor="w")
        print(f"  saved {ppath}")
        written.append(ppath)

    return written


# ---------------------------------------------------------------------------
# Shared bar-panel geometry
# ---------------------------------------------------------------------------
def gt_barpanel_figure(S=None):
    """Build the pinned three-group bar figure and axes shared by every bar
    panel.

    The axes box is pinned in inches (S.geom.ax_w by S.geom.ax_h) and the
    figure is sized around it, with margins wide enough on the left for the
    y-label and a five-digit tick. ax_w fixes the printed bar width, since
    S.geom.bar_w is in data units.

    Returns (fig, ax), with x limits already set to S.geom.xlim3.
    """
    if S is None:
        S = gt_style()
    g = S.geom

    fig = plt.figure(figsize=(g.fig_w, g.fig_h), facecolor="w")
    ax = fig.add_axes([
        g.margin_l / g.fig_w,
        g.margin_b / g.fig_h,
        g.ax_w / g.fig_w,
        g.ax_h / g.fig_h,
    ])
    ax.set_xlim(*g.xlim3)
    return fig, ax


def gt_bar3(ax, G, S=None, bar_w=None, swarm_w=None, pt_size=None, ymax=None):
    """Draw the three-group bar, swarm and SEM panel body.

    Per group: a pale bar at the group mean, a beeswarm of the individual
    points, a standard-error whisker with caps, and a colored outline traced
    round the bar as a polyline rather than a patch edge, so the baseline
    stays open.

    G: length-3 sequence of cleaned 1-D arrays in WT, SH, GT order.
    bar_w, swarm_w, pt_size: override the corresponding style constants.

    ymax: optional display ceiling. The bar height, the standard error and
    every statistic are always computed from the full data; ymax affects only
    which individual points are drawn. Points strictly above it are omitted
    from the swarm and should be declared on the panel (see
    gt_offscale_note). Points exactly at the ceiling are drawn whole, because
    the swarm is plotted with clipping off; otherwise a value sitting on the
    ceiling would render as a half dot bisected by the top spine.

    Returns a length-3 list of the number of points omitted per group.
    """
    if S is None:
        S = gt_style()
    bar_w = S.geom.bar_w if bar_w is None else bar_w
    swarm_w = S.geom.swarm_w if swarm_w is None else swarm_w
    pt_size = S.mk.swarm if pt_size is None else pt_size
    cap_hw = S.geom.cap_hw

    n_off = [0, 0, 0]
    for g in range(3):
        v = np.asarray(G[g], dtype=float).ravel()
        m = v.mean()
        se = v.std(ddof=1) / np.sqrt(v.size)     # sample SD -> standard error
        x = g

        # Swarm positions are computed from the full vector, so the packing
        # and hence every point's x offset is the same whether or not a
        # ceiling is applied; only the drawn subset changes.
        xs = gt_beeswarm(v, x, swarm_w)
        if ymax is None:
            vis = np.ones(v.size, dtype=bool)
        else:
            vis = v <= ymax
            n_off[g] = int((~vis).sum())

        ax.bar(x, m, width=bar_w, color=S.cbarfill[g], edgecolor="none",
               zorder=1)
        ax.scatter(xs[vis], v[vis], s=pt_size,
                   c=[S.cscat[g]], edgecolors="none", zorder=2,
                   clip_on=(ymax is None))

        col = S.cmean[g]
        ax.plot([x, x], [m - se, m + se], "-", color=col, lw=S.lw.se, zorder=3)
        ax.plot([x - cap_hw, x + cap_hw], [m - se, m - se], "-", color=col,
                lw=S.lw.se, zorder=3)
        ax.plot([x - cap_hw, x + cap_hw], [m + se, m + se], "-", color=col,
                lw=S.lw.se, zorder=3)
        ax.plot([x - bar_w / 2, x - bar_w / 2, x + bar_w / 2, x + bar_w / 2],
                [0, m, m, 0], "-", color=col, lw=S.lw.bar, zorder=3)

    return n_off


def gt_offscale_note(ax, n_off, gnames, ymax, S=None, fmt="{:g}"):
    """Annotate a panel whose y axis was capped, stating what is not shown.

    Prints a line such as "2 GT > 12000 (off scale)" below the axes, in
    the same gray used for the n= row, so that a truncated axis declares
    its truncation. The wording is the published one, parentheses and all.

    n_off: length-3 counts, as returned by gt_bar3(..., ymax=...).
    gnames: group names in the same order. ymax: the display ceiling.
    fmt: format string for ymax.

    Returns the note string, empty if nothing was clipped.
    """
    if S is None:
        S = gt_style()
    n_off = list(n_off)
    if sum(n_off) == 0:
        return ""

    parts = [f"{n} {g}" for n, g in zip(n_off, gnames) if n]
    txt = f"{', '.join(parts)} > {fmt.format(ymax)} (off scale)"

    # Centered on the figure rather than the axes: the wide left margin for
    # the y label puts the axes center well right of the page center, and a
    # note centered on the axes would run off the right edge of a panel only
    # about 131 pt wide.
    fig = ax.figure
    bb = ax.get_position()
    t = fig.text(0.5, max(bb.y0 - 0.20, 0.005), txt,
                 ha="center", va="top", fontname=S.font.name,
                 fontsize=S.font.n, color=S.clr.n_txt)

    # Shrink the type to fit the figure width rather than overflow it.
    try:
        fig.canvas.draw()
        fw = fig.get_size_inches()[0] * fig.dpi
        for _ in range(6):
            w = t.get_window_extent(fig.canvas.get_renderer()).width
            if w <= 0.96 * fw or t.get_fontsize() <= 4.0:
                break
            t.set_fontsize(t.get_fontsize() - 0.5)
    except Exception:
        pass          # no renderer available; leave the size as set

    return txt


# ---------------------------------------------------------------------------
# Box + jittered-scatter panel family
# ---------------------------------------------------------------------------
def gt_jitter_binned(y, num_bins=20, min_offset=0.1, max_offset=0.2, rng=None):
    """Density-scaled random horizontal jitter for a 1-D sample.

    y is binned into num_bins equal-width bins along its range. Points in a
    bin holding k of the maximum count receive a uniform random offset within
    +/- (min_offset + (k/max_count) * (max_offset - min_offset)), so denser
    bins spread wider.

    This is not gt_beeswarm: it is uniform random jitter within a
    density-scaled band rather than a deterministic center-out fan, and it is
    used only by the box + scatter panels.

    rng: a NumPy Generator. The jitter is cosmetic and affects no plotted
    y-value or printed statistic, but passing a seeded generator makes the
    exact x positions reproducible; otherwise a fresh default_rng() is used.

    Returns an array of offsets the same length as y.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    offsets = np.zeros(n)
    if n == 0:
        return offsets
    if rng is None:
        rng = np.random.default_rng()

    lo, hi = y.min(), y.max()
    if hi <= lo:
        return offsets
    edges = np.linspace(lo, hi, num_bins + 1)
    bins = np.clip(np.digitize(y, edges[1:-1], right=False), 0, num_bins - 1)
    counts = np.bincount(bins, minlength=num_bins)
    max_count = counts.max() if counts.max() > 0 else 1

    for b in range(num_bins):
        idx = np.flatnonzero(bins == b)
        k = idx.size
        if k == 0:
            continue
        spread = min_offset + (k / max_count) * (max_offset - min_offset)
        offsets[idx] = (rng.random(k) - 0.5) * 2 * spread
    return offsets


def gt_meanerror(ax, G, S=None, positions=(1, 2, 3), err="sd", rng=None,
                 zero_line=False):
    """Draw a mean and error-bar dot plot for three groups.

    Per group: faint jittered individual points in the group's scatter
    color, a vertical error bar, and a large marker at the mean in the
    group's mean color (WT dark gray, SH red, GT blue).

    G: length-3 sequence of 1-D arrays; NaNs are dropped.
    positions: x coordinate per group.
    err: 'sd' for +/- one standard deviation (default) or 'sem' for +/- one
    standard error. zero_line: draw a dashed reference line at y = 0.

    Returns a list of per-group (mean, error) tuples.
    """
    if S is None:
        S = gt_style()
    if rng is None:
        rng = np.random.default_rng()

    if zero_line:
        ax.axhline(0.0, ls="--", color="k", lw=S.lw.hair, zorder=1)

    out = []
    for g in range(3):
        v = np.asarray(G[g], dtype=float).ravel()
        v = v[~np.isnan(v)]
        x = positions[g]
        m = v.mean()
        e = v.std(ddof=1) if err == "sd" else v.std(ddof=1) / np.sqrt(v.size)
        out.append((m, e))

        col = S.cmean[g]
        ptcol = S.cscat[g]

        # faint jittered points behind
        xj = x + gt_jitter_binned(v, rng=rng)
        ax.scatter(xj, v, s=S.mk.pt, c=[ptcol], edgecolors="none",
                   alpha=0.6, zorder=2)

        # vertical error bar in the group color
        ax.plot([x, x], [m - e, m + e], "-", color=col, lw=S.lw.se, zorder=3)

        # large mean marker on top
        ax.scatter([x], [m], s=S.mk.mean, c=[col], edgecolors="none", zorder=4)

    return out


def gt_color_xticklabels(ax, S=None, labels=("WT", "SH", "GT")):
    """Color the WT/SH/GT x-tick labels in their group colors.

    Call after the tick labels are set. Label text is matched against S.cmean,
    so WT reads dark gray, SH red and GT blue; GT dose variants map to the GT
    color. Labels that match nothing are left unchanged.
    """
    if S is None:
        S = gt_style()
    colmap = {"WT": S.cmean[0], "SH": S.cmean[1], "GT": S.cmean[2],
              "GT-HD": S.cmean[2], "GT-LD": S.cmean[2],
              "GT-high": S.cmean[2], "GT (HD)": S.cmean[2],
              "GT (LD)": S.cmean[2]}
    for t in ax.get_xticklabels():
        c = colmap.get(t.get_text())
        if c is not None:
            t.set_color(c)
    return ax


def gt_boxscatter_figure(S=None, fig_w=1.35, fig_h=1.804):
    """Build the figure and axes for a box + scatter panel.

    fig_w, fig_h: figure size in inches. Returns (fig, ax).

    Unlike gt_barpanel_figure, the axes box is positioned in figure-relative
    units. The left margin is wide enough for the long rotated y-axis labels
    this family uses; the figure width compensates so the plotting area keeps
    a constant width.
    """
    if S is None:
        S = gt_style()
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="w")
    ax = fig.add_axes([0.40, 0.13, 0.56, 0.80])
    return fig, ax


def gt_colorboxplot(ax, G, S=None, positions=(1, 2, 3), box_width=0.6075,
                     whisker=0.75, rng=None):
    """Draw a colored box, median and density-jittered scatter for three
    groups.

    The box spans the first to the third quartile, with the median as a rule
    across it. Whiskers extend to the most extreme observation within
    `whisker` times the interquartile range of the nearer box edge; this
    family of panels uses a multiplier of 0.75 rather than the conventional
    1.5.

    G: length-3 sequence of 1-D arrays in WT, SH, GT order; NaNs are dropped.
    positions: x coordinate per group. box_width: box width in data units.
    rng: generator for the point jitter (see gt_jitter_binned).

    The cleaned groups and their x positions are stored on the axes so that
    gt_boxsig can add significance brackets afterwards without the panel
    passing the data a second time.

    Returns a list of per-group (q1, med, q3, lo, hi) tuples.
    """
    if S is None:
        S = gt_style()
    if rng is None:
        rng = np.random.default_rng()

    hw = box_width / 2.0
    # Whisker cap half-width is fixed rather than derived from box_width, so
    # that widening the box leaves the whisker ticks unchanged.
    cap_hw = 0.125
    stats = []
    for g in range(3):
        v = np.asarray(G[g], dtype=float).ravel()
        v = v[~np.isnan(v)]
        x = positions[g]

        q1, med, q3 = np.percentile(v, [25, 50, 75])
        iqr = q3 - q1
        wlen = whisker * iqr
        lo = max(v.min(), q1 - wlen)
        hi = min(v.max(), q3 + wlen)
        stats.append((q1, med, q3, lo, hi))

        # Layering, back to front:
        #   1. box fill, a light tint of the group hue, no edge
        #   2. jittered data points, over the fill so they stay visible
        #   3. box edge, whiskers and median in black, over the points
        # The box edges are Q1 and Q3, so no separate quartile lines are drawn.
        fillcol = gt_ablend(S.cmean[g], 0.22)
        ptcol = S.cscat[g]

        # 1) box fill only (behind the points)
        ax.add_patch(matplotlib.patches.Rectangle(
            (x - hw, q1), box_width, q3 - q1,
            facecolor=fillcol, edgecolor="none", zorder=1))

        # 2) jittered points, on top of the fill
        xj = x + gt_jitter_binned(v, rng=rng)
        ax.scatter(xj, v, s=S.mk.pt, c=[ptcol], edgecolors="none",
                   alpha=0.6, zorder=2)

        # 3) black structure lines over the points: whiskers with short caps
        ax.plot([x, x], [q3, hi], "-", color="k", lw=S.lw.hair, zorder=3)
        ax.plot([x, x], [q1, lo], "-", color="k", lw=S.lw.hair, zorder=3)
        ax.plot([x - cap_hw, x + cap_hw], [hi, hi], "-", color="k",
                lw=S.lw.hair, zorder=3)
        ax.plot([x - cap_hw, x + cap_hw], [lo, lo], "-", color="k",
                lw=S.lw.hair, zorder=3)
        # box outline; the fill was drawn in step 1
        ax.add_patch(matplotlib.patches.Rectangle(
            (x - hw, q1), box_width, q3 - q1,
            facecolor="none", edgecolor="k", linewidth=S.lw.axis, zorder=3))
        # median, same weight as the box edges and whiskers
        ax.plot([x - hw, x + hw], [med, med], "-", color="k",
                lw=S.lw.axis, zorder=4)

    ax._gt_box_groups = [np.asarray(g, float).ravel()[~np.isnan(np.asarray(g, float).ravel())]
                         for g in G]
    ax._gt_box_pos = list(positions)
    return stats


def gt_boxsig(ax, S=None, gnames=("WT", "SH", "GT"), gate=True,
              y_frac=0.90, step_frac=0.075, test="parametric"):
    """Add significance brackets to a three-group box + scatter panel.

    Reads the cleaned groups that gt_colorboxplot stored on the axes, runs
    the three-group comparison and draws pairwise brackets labeled by
    gt_pstar. Call after set_ylim: bracket heights are placed relative to the
    final y limits. This is why the brackets are not drawn inside
    gt_colorboxplot, which runs before the panel fixes its limits.

    test: 'parametric' (default) uses gt_compare3 with the Levene-gated
    route (Levene's test for homogeneity of variance, then one-way ANOVA
    with Tukey-Kramer when equal variance holds or Welch's ANOVA with
    Games-Howell when it is rejected). 'nonparametric' uses gt_compare3_np
    (Kruskal-Wallis omnibus, Mann-Whitney pairwise), as in the Figure 5 fMRI
    panels.

    gate=True draws pairwise brackets only when the omnibus test reaches
    p < 0.05, since pairwise labels under a null omnibus are uncontrolled
    comparisons; otherwise a single small 'n/s' note is printed in place of
    the brackets. gate=False draws all three brackets regardless.

    y_frac: height of the lower bracket row as a fraction of the y range.
    step_frac: additional height of the WT-GT bracket above that row.

    Returns the comparison result dict, or None if fewer than three usable
    groups are present.
    """
    if S is None:
        S = gt_style()
    G = getattr(ax, "_gt_box_groups", None)
    pos = getattr(ax, "_gt_box_pos", [1, 2, 3])
    if G is None or len(G) != 3 or any(g.size < 2 for g in G):
        return None

    if test == "nonparametric":
        # Holm step-down across the three pairwise contrasts, matching the
        # adjusted values reported in Supplementary Table 1.
        R = gt_compare3_np(G, "holm")
        omnibus = "K-W n/s"
    else:
        R = gt_compare3(G, "levene")   # Levene-gated ANOVA / Welch's ANOVA
        omnibus = "ANOVA n/s"
    yl = ax.get_ylim()
    rng = yl[1] - yl[0]
    h = 0.012 * rng
    base = yl[0] + y_frac * rng

    if gate and R["p_anova"] >= 0.05:
        ax.text((pos[0] + pos[2]) / 2.0, base + 0.02 * rng, omnibus,
                ha="center", va="baseline", fontsize=S.font.ns,
                color=S.clr.ns_txt, fontname=S.font.name, clip_on=False)
        return R

    # (WT,SH) and (SH,GT) on the lower row; (WT,GT) one step above.
    gt_drawSig(ax, pos[0], pos[1], base, h, R["labels"][0], S)
    gt_drawSig(ax, pos[1], pos[2], base, h, R["labels"][1], S)
    gt_drawSig(ax, pos[0], pos[2], base + step_frac * rng, h, R["labels"][2], S)
    return R
