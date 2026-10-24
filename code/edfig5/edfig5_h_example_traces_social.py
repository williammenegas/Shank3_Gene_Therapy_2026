"""Extended Data Figure 5, panel h: example trajectory and state-density maps,
box and face conditions.

A 2 x 3 grid of single-animal trajectory and behavioral-state density maps.
The top row is drawn from `xy_box.mat` (a box displayed on screen, the control
condition) and the bottom row from `xy_face.mat` (a face displayed on screen,
the test condition). Each row shows the same three example animals in the order
WT (animal index 212), SH (110) and GT (115).

The drawing routine is shared with panel c and lives in `_traj_common.py`; see
that module's docstring for the state groupings, the color layers, the
orientation of the density layers and the rendering parameters.

The panel is descriptive; no statistical test is applied.
"""

import os

from _traj_common import build_figure, HERE

DATA = os.path.join(HERE, "..", "..", "data", "edfig5", "edfig5_h_example_traces_social")


def main(data_dir=DATA, out_dir=HERE):
    return build_figure(
        "edfig5_h_example_traces_social", data_dir, out_dir,
        block1_file="xy_box.mat", block2_file="xy_face.mat",
        wt_ind=212, sh_ind=110, gt_ind=115,
        block1_label="box", block2_label="face",
    )


if __name__ == "__main__":
    main()
