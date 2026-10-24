"""Extended Data Figure 5, panel c: example trajectory and state-density maps,
treat and object conditions.

A 2 x 3 grid of single-animal trajectory and behavioral-state density maps.
The top row is drawn from `xy_treat.mat` (treat, the control condition) and the
bottom row from `xy_object.mat` (object, the test condition). Each row shows
the same three example animals in the order WT (animal index 242), SH (110)
and GT (115).

The drawing routine is shared with panel h and lives in `_traj_common.py`; see
that module's docstring for the state groupings, the color layers, the
orientation of the density layers and the rendering parameters.

The panel is descriptive; no statistical test is applied.
"""

import os

from _traj_common import build_figure, HERE

DATA = os.path.join(HERE, "..", "..", "data", "edfig5", "edfig5_c_example_traces")


def main(data_dir=DATA, out_dir=HERE):
    return build_figure(
        "edfig5_c_example_traces", data_dir, out_dir,
        block1_file="xy_treat.mat", block2_file="xy_object.mat",
        wt_ind=242, sh_ind=110, gt_ind=115,
        block1_label="treat", block2_label="object",
    )


if __name__ == "__main__":
    main()
