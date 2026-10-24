"""
Figure 2, panel b: per-frame cage position traces for one example animal per
genotype, in the control and test conditions.

Six panels, one per genotype (WT, SH, GT) x condition (control, test). Each
shows the animal's tracked (x, y) position for every frame of the session as
a semi-transparent line plus a scatter of the same points, in normalized cage
coordinates spanning [0, X_MAX] on both axes, on square axes 1.1 in on a
side. Control panels are drawn in green, test panels in purple. Example
animals are identified by the ID values in EXAMPLES.

The script reads traces from the frame-by-frame position arrays, drops frames
with non-positive cagemate distance, truncates the series at MAX_FRAMES,
shifts coordinates by ADD_OFFSET, divides them by SCALE_DIV, and reflects
them to match the cage orientation used in the figure.

Rendering uses plain light axes rather than the house style, and the point
clouds are rasterized inside the vector files. Output is written at 300 dpi
in PDF, EPS and PNG.

No statistical test is applied; the panels are descriptive.

Data: xy_control.mat, xy_test.mat
"""

import os, sys, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
import scipy.io as sio
mpl.rcParams["savefig.dpi"]=300; mpl.rcParams["font.size"]=7
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from gtstyle import central_output_dir  # noqa: E402

DATA = os.path.join(HERE, "..", "..", "data", "fig2", "fig2_b_traces")

EXAMPLES = [
    (1, 245, "WT"), (2, 107, "SH"), (3, 115, "GT"),
]

X_MAX = 0.8          # axis limit in normalized cage coordinates
ADD_OFFSET = 400     # offset added to raw pixel coordinates before scaling
SCALE_DIV = 1000     # divisor mapping offset pixel coordinates to [0, 1]
MAX_FRAMES = 108000  # frames retained per session

def _select_data(mat_path, id_type, ind_to_try):
    S = sio.loadmat(mat_path, variable_names=[
        f"u_condition_{id_type}", f"v_condition_{id_type}",
        f"Bmat_condition_{id_type}", f"Dist_condition_{id_type}",
        f"ID_condition_{id_type}"])
    U = S[f"u_condition_{id_type}"]
    V = S[f"v_condition_{id_type}"]
    D = S[f"Dist_condition_{id_type}"]
    ID = S[f"ID_condition_{id_type}"].ravel()

    rows = np.flatnonzero(ID == ind_to_try)
    if rows.size == 0:
        raise ValueError(f"No rows where ID_condition_{id_type} == {ind_to_try}")
    row = rows[0]
    return U[row, :], V[row, :], D[row, :]



OUT=central_output_dir(HERE)
ttl={"WT":"WT","SH":"SH","GT":"GT"}
HUE={"green":(0.10,0.55,0.30),"purple":(0.45,0.20,0.65)}

def xy(mat_path,id_type,ind):
    R1,R2,dist=_select_data(mat_path,id_type,ind)
    Red=np.column_stack([R1,R2])[dist>0]; Red=Red[:min(MAX_FRAMES,Red.shape[0])]
    ok=np.isfinite(Red[:,0])&np.isfinite(Red[:,1])
    X=(Red[ok,1]+ADD_OFFSET)/SCALE_DIV; Y=(Red[ok,0]+ADD_OFFSET)/SCALE_DIV
    X0,Y0=X,Y; X=1-Y0; Y=X0; X=1-X
    return X,Y

def make(mat,color,name,k):
    X,Y=xy(mat,name[0],name[1]); c=HUE[color]
    # The data box (0 to X_MAX on both axes) is 1.1 in square; the remaining
    # margins hold the ticks and title.
    AX=1.1; L,R,Bm,T=0.34,0.10,0.30,0.26
    FW=L+AX+R; FH=Bm+AX+T
    fig=plt.figure(figsize=(FW,FH),facecolor="w")
    ax=fig.add_axes([L/FW, Bm/FH, AX/FW, AX/FH])
    ax.plot(X,Y,'-',color=c,lw=0.15,alpha=0.35,rasterized=True)
    ax.scatter(X,Y,s=0.4,c=[c],alpha=0.30,edgecolors="none",rasterized=True)
    ax.set_xlim(0,X_MAX); ax.set_ylim(0,X_MAX); ax.set_aspect("equal")
    ax.set_xticks([0,0.4,0.8]); ax.set_yticks([0,0.4,0.8]); ax.tick_params(labelsize=6, width=0.6, length=2.5)
    for s in ax.spines.values(): s.set_linewidth(0.6); s.set_color((0.2,0.2,0.2))
    ax.set_title(ttl[name[2]], fontsize=8, fontweight="bold")
    for ext in ("pdf","eps","png"):
        fig.savefig(os.path.join(OUT,f"fig2_b_traces_p{k}.{ext}"), dpi=300, bbox_inches=None, facecolor="w")
    plt.close(fig)
    print(f"fig2_b_traces_p{k} ({ttl[name[2]]}, {color}): n={X.size} frames @300dpi")

CTRL=os.path.join(DATA,"xy_control.mat"); TEST=os.path.join(DATA,"xy_test.mat")
k=1
for id_type,ind,g in EXAMPLES: make(CTRL,"green",(id_type,ind,g),k); k+=1
for id_type,ind,g in EXAMPLES: make(TEST,"purple",(id_type,ind,g),k); k+=1
print("saved 6 panels to",OUT)
