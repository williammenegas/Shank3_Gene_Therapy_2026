# Uniform house style (v2)

**Rule: a given visual role has ONE size everywhere.** When a panel's code sets
a size, it should reference the `gtstyle` constant for that role, not a literal.

## The system

| Role | Constant | Value | Notes |
|---|---|---|---|
| body text, tick numbers, axis labels | `S.font.tick` / `S.font.lab` | **7 pt** | matches clean F5/F6 |
| panel title | `S.font.title` | **8 pt** | matches clean .ai (was 9) |
| significance stars | `S.font.star` | 8 pt | |
| "n/s" text | `S.font.ns` | 6 pt | |
| "n=" annotations | `S.font.n` | 6 pt | |
| panel letter | `S.font.letter` | **12 pt** | every .ai uses 12 |
| individual data point (swarm, scatter, paired) | `S.mk.pt` (= `S.mk.swarm`) | **6 pt²** | ONE size for all data dots |
| mean / median marker | `S.mk.mean` | 20 pt² | ONE size for all mean dots |
| legend marker | `S.mk.leg` | 30 pt² | |
| axis line | `S.lw.axis` | 0.6 pt | |
| bar outline | `S.lw.bar` | 1.0 pt | |
| SEM / CI whisker | `S.lw.se` / `S.lw.ci` | 0.9 pt | |
| mean rule | `S.lw.mean` | 1.5 pt | |
| significance bracket | `S.lw.sig` | 0.5 pt | |
| connector / reference hairline | `S.lw.hair` | 0.5 pt | |
| individual-subject trace | `S.lw.trace` | 0.4 pt | |
| paired connector | `S.lw.pair` | 0.6 pt | |
| regression / model-fit line | `S.lw.fit` | 1.0 pt | new; replaces per-panel 1.0/1.8 |

## What changes in the panels

The bespoke panels (correlation scatters, paired plots, trajectory plots in
figs 2-5; the new-data panels; a few sup panels) carry LITERAL sizes
copied from their original source (e.g. `s=50`, `s=36`, `s=22`, `lw=1.8`,
`fontsize=13`, `labelsize=9`). Each are rewritten to the role constant:

* individual data points → `s=S.mk.pt`
* mean / median dots → `s=S.mk.mean`
* regression / fit lines → `lw=S.lw.fit`
* connectors → `lw=S.lw.pair` or `S.lw.hair`
* any explicit `fontsize=`/`labelsize=` on ticks/labels → `S.font.tick` (7)
* any explicit title fontsize → `S.font.title` (8)

Panels that already go through the `gtstyle` helpers (`gt_bar3`,
`gt_colorboxplot`, `gt_boxscatter_figure`, `gt_axes`) inherit the new values
automatically once `gt_style` is updated with no per-panel edit needed.

## What does NOT change

* Colors (the palette is already unified in `gt_style`).
* Data, statistics, panel geometry, axis limits, number of sub-panels.
* Panel content. Only element sizes are normalized to the house values so that
  all panels match across figures.

## Result

Every panel was re-rendered under the uniform style and re-assembled
(`revision_output/assembly_v6/`). Measured across all main + supplement panels:

* **Text:** 7 pt dominates (2987 spans), then 8 pt titles (128), 6 pt small
  (208), 12 pt letters. No stray 13/14/15/22 pt.
* **Data dots:** collapse to one dominant size (2.4 pt diameter = `S.mk.pt`,
  6 pt²) with a single mean-marker size (4.5 pt = `S.mk.mean`, 20 pt²). E.g.
  Figure 2e's correlation scatter, previously `s=50`, is now the same 2.4 pt
  as every swarm point in the set.

Residual, by design or pending:
* Figure 5b's 74-node connectome keeps 4 pt ROI labels and data-scaled node
  sizes / edge widths (those encode statistics, not house-style roles).
* Three slow panels (fig4i GLM-HMM, S4c/d 6 GB stream, S7g GLM-HMM) have the
  edits in place but were not re-rendered in this pass; re-run them in a
  longer session to refresh their PDFs.

## Deliberately not matched

The distorted F2/F3/F4 `.ai` panels carry fractional font sizes (6.4, 5.7 …)
and dots up to 6.7 pt from being scaled ~65 % by hand in Illustrator. Those are
the artefact this whole exercise removes; the target is the clean, consistent
look of F1/F5/F6 at 7 pt, not the distorted panels.
