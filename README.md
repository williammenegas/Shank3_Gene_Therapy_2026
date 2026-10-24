# Gene therapy improves behavioral and neural phenotypes in a SHANK3 marmoset model for autism

Data and code to reproduce the figures from this paper.

## Usage

```bash
pip install -r code/requirements.txt
python code/tools/build_data.py     # once, to populate data/
python code/render_all.py                  # render every panel
python code/render_all.py fig1 fig6        # or a subset
python code/render_all.py --list           # list panels without rendering
```

Each script writes a vector `.pdf` and `.eps` into `figures/` or
`supplemental_figures/`, and prints the statistics it computed. Scripts can
also be run individually:

```bash
cd code/fig1 && python fig1_d_locomotion.py
```

## Layout

```
code/
  fig1/ … fig6/        main figure panels
  edfig1/ … edfig14/   extended data panels
  gtstyle/             shared style and statistics library
  tools/               build_data.py, make_reduced_inputs.py
  render_all.py        batch renderer
  requirements.txt     package versions
data/           per-panel input data and manifest.json
figures/               rendered main figure panels
supplemental_figures/  rendered extended data panels
```

Files with a leading underscore are helpers: `_f6common.py`, `_traj_common.py`,
`_common.py`, `_s4cd_common.py`, `_classifier_check.py`, and the
`edfig10/_build_subsets.py` endpoint-subset builder. The panels that need these
invoke them automatically.

## Data

Every panel names the folder it reads from explicitly, as a `DATA`
constant near the top of the script:

```python
DATA = os.path.join(HERE, "..", "..", "data", "fig1", "fig1_b_radar")
```

Nothing is resolved by basename and nothing is searched for at run time: the
path in the script is the path that is opened.

Figures whose data all belongs to a single panel keep their files
directly in the figure's folder, and those scripts name one level less:

```python
DATA = os.path.join(HERE, "..", "..", "data", "edfig1")
```

Each distinct file content is stored once, in the folder of the first panel
that uses it in figure order. A panel that reads a file owned by another
panel points at that panel's folder, so a few scripts carry a second named
constant (`DATA_HLABEL`, `DATA_STATE`, ...) alongside `DATA`; the file each
constant serves is named at the call site. `code/tools/_ownership_map.json` records
which folder owns each file.

`data/manifest.json` records, for every panel, the files it reads and
their original location under `new_code/` or `new_data/`. It documents
provenance and is read by `tools/build_data.py`; no panel reads it.

The bundle is self-contained: every panel runs against what is in
`data`, with no separate download.


Three types of files:

- `_reduced` - 28 files that stand in for much larger originals, mostly raw
  position tracking and fMRI data. They are written by 
  `tools/make_reduced_inputs.py` in the original format and under the
  original basename, keeping only what the panel consumes; each extractor's
  docstring states why the panel's printed numbers and rendered figure are
  unchanged.
- `_trim` - `HD_LD_Merged_Data.xlsx`, copied with only the columns the three
  Extended Data Fig. 2 endpoints require.
- `_derived` - the two GLM-HMM fits described under Notes below.

`code/tools/build_data.py` populates the folder from a checkout that also
holds `new_code/` and `new_data/`. `--max-mb N` leaves files above a size
threshold at their original location, `--dry-run` reports without copying.

## Statistics

Group comparisons are made using `gtstyle.stats`:

- `gt_compare3` - one-way ANOVA with Tukey-Kramer post-hoc. With
  `correction="levene"`, homogeneity of variance is tested first (Levene,
  median-centered); Welch's ANOVA with Games-Howell is used where equal variance
  is rejected at p < 0.05.
- `gt_compare3_np` - Kruskal-Wallis with Mann-Whitney post-hoc, for endpoints
  not assumed normal.
- `gt_hedges` - Hedges' g with the small-sample correction and a bootstrap 95%
  confidence interval.

Figure 6 and Extended Data Fig. 9 use permutation p-values with
Benjamini-Hochberg FDR across the 16 endpoints within each contrast, and a
ridge-penalized logistic classifier fit on the wild-type versus mutant contrast
with treated animals held out.

Sample sizes vary by endpoint and each panel prints its own n (number of
animals).

## Notes

- Figures are designed in Arial. Without it, `gtstyle` falls back to Liberation
  Sans, which is metric-compatible, and prints a notice.
- `fig4/fig4_i_glmhmm.py` and `edfig7/edfig7_g_glmhmm.py` fit a GLM-HMM, which
  takes several minutes, so the converged fit is stored in `data` and
  both scripts plot from it and run in seconds. `GT_REFIT=1` refits from
  `xy_type*.mat` and overwrites the stored fit; `GT_MAX_EM` caps the number of
  EM iterations for a quick preview, in which case the model is not converged.
  The two fits are listed in the `_derived` section of
  `data/manifest.json`. `fig4_g_touch_distributions.py` and
  `edfig7_e_touch_distributions.py` read the same files to draw their
  MAP-state lines.
- Cross-validation folds in the classifier panels are unseeded, so
  cross-validated AUROC varies slightly between runs.

## Related repositories

The homecage behavior behind Figures 2 and 3 and Extended Data Figs. 4, 5, 6
and 10 is produced by three separate repositories. This repository
does not depend on them. The `hlabel_*.csv` and `hcoord_*.csv` files in 
`data` are the pipeline's per-frame output. They are listed 
for anyone reproducing the behavioral analysis from raw video.

[marm_behavior](https://github.com/williammenegas/marm_behavior) is the
behavioral pipeline itself. One command runs six stages over overhead stereo
video of four differently marked marmosets in the homecage: DeepLabCut pose
estimation, per-animal body-part tracks, per-frame behavioral features, and a
projection of those features into a learned behavioral cluster space. The 80
states the panels here plot, and the per-frame `hlabel` files they read, are
its output.

[marm_filt](https://github.com/williammenegas/marm_filt) denoises the
body-part tracks. A graph neural network with temporal convolutions, 
written in TensorFlow and trained on the tracks themselves, smooths each 
animal's skeleton over time and writes a filtered copy next to each original.

[marm_viz](https://github.com/williammenegas/marm_viz) is the general-purpose
viewer for the outputs. Pointed at a recording folder it draws ten plot types, 
among them spatial occupancy heatmaps, smoothed density maps, ethograms, 
state-usage and t-SNE summaries, head-direction polar histograms, 
behavioral-category transition matrices and cagemate state-usage correlations. 
Several panels in this repository are the published form of one of those 
views, fixed to a single figure and cohort.

## Author

William Menegas

Code and data related to the figures in the accompanying manuscript (Gene
therapy improves behavioral and neural phenotypes in a SHANK3 marmoset model
for autism).

## License

The code in this repository is released under the MIT License; see LICENSE.
