import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from pathlib import Path
import random, math

mpl.rcParams["savefig.format"] = "pdf"
OUT_DIR = Path("figures_pdf_png")
OUT_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.size": 16,
    "font.family": "Times New Roman",
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "legend.title_fontsize": 16,
    "figure.titlesize": 18,
})

SAMPLE_FRAC = 0.02
RANDOM_SEED = 42
PLANE_W, PLANE_H = 20.0, 8.0
LAYER_GAP = 5.0
PLANE_ALPHA = 0.40

NODE_R_REAL = 0.1
NODE_ALPHA_REAL = 1.0
NODE_R_FAKE = 0.3
NODE_ALPHA_FAKE = 1.0
PROD_HALF = 0.1
PROD_NODE_ALPHA = 1.0

EDGE_W_REAL = 0.01
EDGE_ALPHA_REAL = 0.5
EDGE_W_FAKE = 0.6
EDGE_ALPHA_FAKE = 0.9

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

df = pd.read_excel("data.xlsx")
df["is_real"] = df["Label_user"].astype(int) == 0

user_avg = df.groupby("Reviewer_id")["Rating"].mean().round().clip(1, 5).astype(int)
real_ids = df.loc[df["is_real"], "Reviewer_id"].unique()
fake_ids = df.loc[~df["is_real"], "Reviewer_id"].unique()

def stratified(ids):
    keep = []
    for r in range(1, 6):
        grp = [u for u in ids if user_avg.get(u) == r]
        k = max(1, int(len(grp) * SAMPLE_FRAC))
        if grp:
            keep.extend(np.random.choice(grp, min(k, len(grp)), replace=False))
    return keep

keep_real = stratified(real_ids)
keep_real = random.sample(keep_real, len(keep_real) // 2)
keep_fake = stratified(fake_ids)
keep_u = set(keep_real + keep_fake)
df = df[df["Reviewer_id"].isin(keep_u)]
prod_ids = df["Product_id"].unique()

print("Selected Real users:", len(keep_real))
print("Selected Fake users:", len(keep_fake))
print("Number of products involved:", len(prod_ids))

X_SHIFT = 35.0
TEXT_X = 0.0005
TEXT_Y = {
    "Fake": 0.70,
    "Product": 0.38,
    "Real": 0.08
}

def rand_uv(n):
    return np.column_stack((
        np.random.uniform(-PLANE_W / 2, PLANE_W / 2, n),
        np.random.uniform(-PLANE_H / 2, PLANE_H / 2, n)
    ))

uv_real = dict(zip(keep_real, rand_uv(len(keep_real))))
uv_fake = dict(zip(keep_fake, rand_uv(len(keep_fake))))
uv_prod = dict(zip(prod_ids, rand_uv(len(prod_ids))))

uv_real = {u: (x + X_SHIFT, y) for u, (x, y) in uv_real.items()}
uv_fake = {u: (x + X_SHIFT, y) for u, (x, y) in uv_fake.items()}
uv_prod = {p: (x + X_SHIFT, y) for p, (x, y) in uv_prod.items()}

rating_color = {1: "#E64B35", 2: "#F7C242", 3: "#4DBBD5", 4: "#00A087", 5: "#7E57C2"}

def circle_poly(c, r, z, segs=16):
    cx, cy = c
    return [(cx + r * math.cos(a), cy + r * math.sin(a), z) for a in np.linspace(0, 2 * math.pi, segs, False)]

def square_poly(c, hs, z):
    cx, cy = c
    return [(cx - hs, cy - hs, z), (cx + hs, cy - hs, z), (cx + hs, cy + hs, z), (cx - hs, cy + hs, z)]

def plot_polys(ax, verts, **kw):
    ax.add_collection3d(Poly3DCollection(verts, **kw))

fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection="3d")
ax.axis("off")

Z_PROD = 0
Z_FAKE = Z_PROD + LAYER_GAP
Z_REAL = Z_PROD - LAYER_GAP

plot_polys(ax, [[(-PLANE_W / 2 + X_SHIFT, -PLANE_H / 2, Z_FAKE),
                 (PLANE_W / 2 + X_SHIFT, -PLANE_H / 2, Z_FAKE),
                 (PLANE_W / 2 + X_SHIFT, PLANE_H / 2, Z_FAKE),
                 (-PLANE_W / 2 + X_SHIFT, PLANE_H / 2, Z_FAKE)]],
           facecolors="lightgray", alpha=0.4)
ax.text2D(TEXT_X, TEXT_Y["Fake"], "Fake User Layer", transform=ax.transAxes, weight="bold", fontsize=17, ha="left")

fv = [circle_poly(uv_fake[u], 0.3, Z_FAKE) for u in keep_fake]
fc = [rating_color[user_avg[u]] for u in keep_fake]
plot_polys(ax, fv, facecolors=fc, edgecolors="none", alpha=1.0)

lf, cf = [], []
for _, r in df[~df["is_real"]].iterrows():
    u0, v0 = uv_fake[r["Reviewer_id"]]
    u1, v1 = uv_prod[r["Product_id"]]
    lf.append([(u0, v0, Z_FAKE), (u1, v1, Z_PROD)])
    cf.append(rating_color[int(r["Rating"])])
ax.add_collection3d(Line3DCollection(lf, colors=cf, linewidths=0.6, alpha=0.9))

plot_polys(ax, [[(-PLANE_W / 2 + X_SHIFT, -PLANE_H / 2, Z_PROD),
                 (PLANE_W / 2 + X_SHIFT, -PLANE_H / 2, Z_PROD),
                 (PLANE_W / 2 + X_SHIFT, PLANE_H / 2, Z_PROD),
                 (-PLANE_W / 2 + X_SHIFT, PLANE_H / 2, Z_PROD)]],
           facecolors="lightgray", alpha=0.4)
ax.text2D(TEXT_X, TEXT_Y["Product"], "Product Layer", transform=ax.transAxes, weight="bold", fontsize=17, ha="left")

pv = [square_poly(uv_prod[p], 0.1, Z_PROD) for p in prod_ids]
plot_polys(ax, pv, facecolors="white", edgecolors="black", alpha=1.0, linewidths=0.8)

lr, cr = [], []
for _, r in df[df["is_real"]].iterrows():
    u0, v0 = uv_real[r["Reviewer_id"]]
    u1, v1 = uv_prod[r["Product_id"]]
    lr.append([(u0, v0, Z_REAL), (u1, v1, Z_PROD)])
    cr.append(rating_color[int(r["Rating"])])
ax.add_collection3d(Line3DCollection(lr, colors=cr, linewidths=0.01, alpha=0.5))

plot_polys(ax, [[(-PLANE_W / 2 + X_SHIFT, -PLANE_H / 2, Z_REAL),
                 (PLANE_W / 2 + X_SHIFT, -PLANE_H / 2, Z_REAL),
                 (PLANE_W / 2 + X_SHIFT, PLANE_H / 2, Z_REAL),
                 (-PLANE_W / 2 + X_SHIFT, PLANE_H / 2, Z_REAL)]],
           facecolors="lightgray", alpha=0.4)
ax.text2D(TEXT_X, TEXT_Y["Real"], "Real User Layer", transform=ax.transAxes, weight="bold", fontsize=17, ha="left")

rv = [circle_poly(uv_real[u], 0.1, Z_REAL) for u in keep_real]
rc = [rating_color[user_avg[u]] for u in keep_real]
plot_polys(ax, rv, facecolors=rc, edgecolors="none", alpha=1.0)

ax.set_box_aspect((PLANE_W, PLANE_H, 2 * LAYER_GAP))
ax.view_init(elev=25, azim=35)

from matplotlib.lines import Line2D
legend = [Line2D([0], [0], marker="s", linestyle="", markersize=8,
                 markerfacecolor="white", markeredgecolor="black", label="Product")] + \
         [Line2D([0], [0], marker="o", linestyle="", markersize=8,
                 markerfacecolor=c, markeredgecolor="none", label=f"Rating {r}")
          for r, c in rating_color.items()]
ax.legend(handles=legend, loc="lower right", bbox_to_anchor=(1.05, 0.05),
          frameon=False, fontsize=15)

plt.tight_layout()

outfile_pdf = OUT_DIR / "3d_user_product_layers.pdf"
outfile_png = OUT_DIR / "3d_user_product_layers.png"

fig.savefig(outfile_pdf, bbox_inches="tight")
fig.savefig(outfile_png, bbox_inches="tight", dpi=600)
plt.close(fig)
print(f"Saved: {outfile_pdf} and {outfile_png}")