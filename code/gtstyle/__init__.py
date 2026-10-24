"""
gtstyle -- shared style and statistics library for the figure panel scripts
of the SHANK3 marmoset gene-therapy paper.

Modules
-------
    style.py     colors, typography, line weights, pinned panel geometry
    stats.py     ANOVA, post-hoc contrasts, effect sizes, permutation tests,
                 FDR adjustment, t/F/beta distribution functions
    plotting.py  beeswarm, axes styling, significance brackets, panel
                 letters, vector export
    color.py     sRGB <-> CIELAB conversion and Lab-space colormaps
    matio.py     .mat loaders (v7 and v7.3)

Usage
-----
    from gtstyle import *

    S = gt_init()                       # style struct + rcParams
    fig, ax = gt_barpanel_figure(S)     # pinned three-group geometry
    gt_bar3(ax, [wt, sh, gt], S)

    p, F, df1, df2 = gt_anova1([wt, sh, gt])
    _, p_adj = gt_posthocT([wt, sh, gt], 'none')

    gt_axes(ax, S)
    gt_panelLetter(ax, 'a', S)
    gt_figStyle(fig, S)
    gt_saveFig(fig, 'main_a_locomotion', outdir)
"""

from .style import gt_style, gt_ablend, _Bag

from .color import gt_rgb2lab, gt_lab2rgb, gt_lab_colormap

from .stats import (
    gt_betainc, gt_tcdf, gt_fcdf,
    gt_anova1, gt_posthocT,
    gt_cohenD, gt_hedges, gt_permp, gt_bh,
    gt_pstar, gt_stars, gt_pfmt, gt_compare3, gt_compare3_np, gt_tukey, gt_pairedt,
    gt_holm,
    PSTAR_T1, PSTAR_T2, PSTAR_T3,
)

from .plotting import (
    gt_init, gt_beeswarm, gt_axes, gt_figStyle,
    gt_panelLetter, gt_drawSig, gt_saveFig, central_output_dir,
    gt_barpanel_figure, gt_bar3, gt_offscale_note,
    gt_jitter_binned, gt_boxscatter_figure, gt_colorboxplot, gt_boxsig,
    gt_meanerror, gt_color_xticklabels,
)

from .matio import loadmat, load_struct, MatStruct

__all__ = [
    "gt_style", "gt_ablend",
    "gt_rgb2lab", "gt_lab2rgb", "gt_lab_colormap",
    "gt_betainc", "gt_tcdf", "gt_fcdf",
    "gt_anova1", "gt_posthocT",
    "gt_cohenD", "gt_hedges", "gt_permp", "gt_bh",
    "gt_pstar", "gt_stars", "gt_pfmt", "gt_compare3", "gt_compare3_np", "gt_tukey", "gt_pairedt",
    "gt_holm",
    "PSTAR_T1", "PSTAR_T2", "PSTAR_T3",
    "gt_init", "gt_beeswarm", "gt_axes", "gt_figStyle",
    "gt_panelLetter", "gt_drawSig", "gt_saveFig", "central_output_dir",
    "gt_barpanel_figure", "gt_bar3", "gt_offscale_note",
    "gt_jitter_binned", "gt_boxscatter_figure", "gt_colorboxplot", "gt_boxsig",
    "gt_meanerror", "gt_color_xticklabels",
    "loadmat", "load_struct", "MatStruct",
]

__version__ = "1.0.0"
