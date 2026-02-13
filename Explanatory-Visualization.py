import os
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import warnings, logging, random
from time import perf_counter
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore", category=Warning)

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra as cs_dijkstra, shortest_path as cs_shortest_path

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

try:
    import umap
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False

try:
    import networkit as nk
    _HAS_NETWORKIT = True
except Exception:
    _HAS_NETWORKIT = False


DATA_FILE  = "data.xlsx"
SHEET_NAME = 0
OUT_DIR    = "nature_style_figures"
DPI_SAVE   = 600

SEED       = 42
SAMPLE_FRAC = 1.0
MAX_USERS_PER_PRODUCT = 200

NODE_FEATURES = [
    "clustering", "closeness", "betweenness", "pagerank", "eigenvector",
    "katz", "kcore", "harmonic", "two_hop_wsum", "strength"
]
LANDMARK_K = 256
SP_CHUNK   = 16

MAX_POINTS_PER_CLASS = 6000
MAX_TSNE_PER_CLASS   = 2000

SCHEM_CORE_N = 120
SCHEM_MID_N  = 120
SCHEM_PERI_N = 120

MAKE_FIG1_EMB_2x3 = True
MAKE_FIG2_CCDF_2x5 = True
MAKE_FIG7_COREPERI_SCHEM = True


def setup_style():
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": DPI_SAVE,
        "font.size": 9.5,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


COLORS = {
    "real":   "#CC79A7",
    "fake":   "#56B4E9",
    "center": "#D55E00",
    "product": "#9E9E9E",
    "edge":   "#4D4D4D",
    "line_w": "#CC79A7",
    "line_u": "#56B4E9",
}


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TopoOnlyNetworkFigs")


def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def save_show(fig, filename: str):
    _ensure_dir(OUT_DIR)
    base, _ = os.path.splitext(filename)
    pdf_name = base + ".pdf"
    path = os.path.join(OUT_DIR, pdf_name)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)


def add_real_fake_legend(ax, loc="best", fontsize=8.5):
    handles = [
        Patch(facecolor=COLORS["real"], edgecolor="none", label="Real"),
        Patch(facecolor=COLORS["fake"], edgecolor="none", label="Fake"),
    ]
    ax.legend(handles=handles, loc=loc, fontsize=fontsize, frameon=False)


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Reviewer_id"] = df["Reviewer_id"].astype(str)
    df["Product_id"]  = df["Product_id"].astype(str)
    df["Date"] = pd.to_datetime(df["Date"])
    df["rating_norm"] = (df["Rating"] - 1) / 4.0
    df["coupling"] = (1 - np.abs(df["Sentiment"] - df["rating_norm"])).clip(0, 1)
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce").fillna(0).astype(int)
    df["Label_user"] = pd.to_numeric(df["Label_user"], errors="coerce").fillna(0).astype(int)
    return df


def user_labels_from_df(df: pd.DataFrame) -> pd.Series:
    s = df.groupby("Reviewer_id")["Label_user"].first()
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def build_bipartite_multigraph(df: pd.DataFrame) -> Tuple[nx.MultiGraph, np.ndarray, np.ndarray]:
    users = df["Reviewer_id"].unique()
    prods = df["Product_id"].unique()
    B = nx.MultiGraph()
    B.add_nodes_from(users, bipartite="user")
    B.add_nodes_from(prods, bipartite="product")
    for _, r in df.iterrows():
        B.add_edge(r["Reviewer_id"], r["Product_id"], weight=float(r["coupling"]))
    return B, users, prods


def collapse_multiedges_to_simple_edges(B: nx.MultiGraph) -> List[Tuple[str, str, float]]:
    esum = {}
    for u, v, d in B.edges(data=True):
        key = tuple(sorted((u, v)))
        esum[key] = esum.get(key, 0.0) + float(d.get("weight", 1.0))
    return [(u, v, w) for (u, v), w in esum.items()]


def _build_csr_from_edges(edges_merged, weighted: bool):
    nodes = []
    for u, v, _ in edges_merged:
        nodes.append(u)
        nodes.append(v)
    nodes = list(dict.fromkeys(nodes))
    n2i = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)

    m = len(edges_merged)
    rows = np.empty(m * 2, dtype=np.int64)
    cols = np.empty(m * 2, dtype=np.int64)
    data = np.empty(m * 2, dtype=np.float64)

    for k, (u, v, w) in enumerate(edges_merged):
        i = n2i[u]
        j = n2i[v]
        val = float(w) if weighted else 1.0
        rows[2 * k] = i
        cols[2 * k] = j
        data[2 * k] = val
        rows[2 * k + 1] = j
        cols[2 * k + 1] = i
        data[2 * k + 1] = val

    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    A.sum_duplicates()
    A.eliminate_zeros()
    return A, nodes, n2i


def _pagerank_power(A: sp.csr_matrix, max_iter=50, tol=1e-6, damping=0.85):
    N = A.shape[0]
    out = np.asarray(A.sum(axis=1)).ravel()
    out[out == 0] = 1.0
    P = sp.diags(1.0 / out) @ A
    x = np.full(N, 1.0 / N, dtype=np.float64)
    teleport = (1.0 - damping) / N
    for _ in range(max_iter):
        x_new = damping * (x @ P) + teleport
        if np.linalg.norm(x_new - x, 1) < tol:
            x = x_new
            break
        x = x_new
    return x


def _eigenvector_power(A: sp.csr_matrix, max_iter=80, tol=1e-6):
    N = A.shape[0]
    rng = np.random.RandomState(42)
    x = rng.rand(N).astype(np.float64)
    x /= (np.linalg.norm(x) + 1e-12)
    for _ in range(max_iter):
        x_new = A @ x
        nrm = np.linalg.norm(x_new) + 1e-12
        x_new /= nrm
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    return x


def _katz_iter(A: sp.csr_matrix, alpha=0.005, beta=1.0, max_iter=80, tol=1e-6):
    N = A.shape[0]
    x = np.zeros(N, dtype=np.float64)
    b = np.full(N, beta, dtype=np.float64)
    for _ in range(max_iter):
        x_new = alpha * (A @ x) + b
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    return x


def closeness_harmonic_landmarks_csgraph(
    A_dist: sp.csr_matrix,
    landmarks: np.ndarray,
    weighted: bool,
    chunk_size: int = 16,
    eps: float = 1e-12,
):
    N = A_dist.shape[0]
    sumd = np.zeros(N, dtype=np.float64)
    cntd = np.zeros(N, dtype=np.float64)
    sum_inv = np.zeros(N, dtype=np.float64)
    cnt_inv = np.zeros(N, dtype=np.float64)

    for i in range(0, len(landmarks), int(max(1, chunk_size))):
        src = landmarks[i : i + int(max(1, chunk_size))]
        if weighted:
            D = cs_dijkstra(A_dist, directed=False, indices=src)
        else:
            D = cs_shortest_path(A_dist, directed=False, unweighted=True, indices=src)

        finite = np.isfinite(D)
        if finite.any():
            sumd += np.where(finite, D, 0.0).sum(axis=0)
            cntd += finite.sum(axis=0).astype(np.float64)
            pos = finite & (D > 0)
            if pos.any():
                sum_inv += np.where(pos, 1.0 / (D + eps), 0.0).sum(axis=0)
                cnt_inv += pos.sum(axis=0).astype(np.float64)
        del D

    closeness = np.where(cntd > 1.0, (cntd - 1.0) / (sumd + eps), 0.0)
    harmonic = np.where(cnt_inv > 0.0, sum_inv / (cnt_inv + eps), 0.0)
    return closeness, harmonic


def weighted_strength_core_number(G: nx.Graph, weight: str = "weight") -> Dict:
    import heapq

    if G.number_of_nodes() == 0:
        return {}
    cur = {u: float(G.degree(u, weight=weight)) for u in G.nodes()}
    heap = [(cur[u], u) for u in G.nodes()]
    heapq.heapify(heap)
    removed = set()
    core = {}
    while heap:
        s, u = heapq.heappop(heap)
        if u in removed:
            continue
        if abs(s - cur[u]) > 1e-12:
            continue
        removed.add(u)
        core[u] = float(s)
        for v in G.neighbors(u):
            if v in removed:
                continue
            w = float(G[u][v].get(weight, 1.0))
            cur[v] = cur[v] - w
            heapq.heappush(heap, (cur[v], v))
    return core


def compute_node_features_from_edges(edges_merged, weighted: bool):
    t0 = perf_counter()
    A_w, nodes, n2i = _build_csr_from_edges(edges_merged, weighted=weighted)
    N = A_w.shape[0]

    A_bin = A_w.copy()
    A_bin.data = np.ones_like(A_bin.data)

    strength = np.asarray(A_w.sum(axis=1)).ravel()
    pagerank = _pagerank_power(A_w)
    eigenvec = _eigenvector_power(A_w)
    katz = _katz_iter(A_w)

    if weighted:
        A_dist = A_w.copy()
        A_dist.data = 1.0 / (A_w.data + 1e-12)
    else:
        A_dist = A_bin.copy()

    rng = np.random.RandomState(42)
    L = min(int(LANDMARK_K), N)
    landmarks = rng.choice(N, size=L, replace=False)
    closeness, harmonic = closeness_harmonic_landmarks_csgraph(
        A_dist, landmarks, weighted=weighted, chunk_size=SP_CHUNK
    )

    A2 = (A_w @ A_w).tocsr()
    A2.sum_duplicates()
    A2.eliminate_zeros()

    A2_excl = A2.copy()
    A2_excl.setdiag(0.0)
    A2_excl.eliminate_zeros()
    A2_excl = A2_excl - A2_excl.multiply(A_bin)
    A2_excl.eliminate_zeros()

    two_hop_wsum = np.asarray(A2_excl.sum(axis=1)).ravel()

    coo = A2_excl.tocoo()
    ii = coo.row
    jj = coo.col
    val2 = coo.data.astype(np.float64, copy=False)
    clustering = np.zeros(N, dtype=np.float64)
    cnt = np.zeros(N, dtype=np.float64)
    if val2.size > 0:
        if not weighted:
            deg = np.asarray(A_bin.sum(axis=1)).ravel()
            common = val2
            union = deg[ii] + deg[jj] - common
            sim = np.where(union > 0, common / union, 0.0)
        else:
            dot = val2
            nrm = np.sqrt(np.asarray(A_w.multiply(A_w).sum(axis=1)).ravel() + 1e-12)
            sim = dot / (nrm[ii] * nrm[jj] + 1e-12)
        np.add.at(clustering, ii, sim)
        np.add.at(cnt, ii, 1.0)
        clustering = np.where(cnt > 0, clustering / cnt, 0.0)

    edges_for_graph = edges_merged if weighted else [(u, v, 1.0) for (u, v, _w) in edges_merged]
    Gs = nx.Graph()
    Gs.add_weighted_edges_from(edges_for_graph, weight="weight")
    try:
        if not weighted:
            core_num = nx.core_number(Gs)
        else:
            core_num = weighted_strength_core_number(Gs, weight="weight")
        kcore = np.zeros(N, dtype=np.float64)
        for node, k in core_num.items():
            if node in n2i:
                kcore[n2i[node]] = float(k)
    except Exception:
        kcore = np.zeros(N, dtype=np.float64)

    betweenness = np.zeros(N, dtype=np.float64)
    if _HAS_NETWORKIT:
        node_list = list(Gs.nodes())
        n2j = {n: i for i, n in enumerate(node_list)}
        Gnk = nk.Graph(len(node_list), weighted=bool(weighted), directed=False)
        for u, v, d in Gs.edges(data=True):
            w = float(d.get("weight", 1.0))
            if weighted:
                Gnk.addEdge(n2j[u], n2j[v], 1.0 / (w + 1e-12))
            else:
                Gnk.addEdge(n2j[u], n2j[v])
        if weighted:
            bc = nk.centrality.Betweenness(Gnk, normalized=False, computeEdgeCentrality=False)
        else:
            bc = nk.centrality.KadabraBetweenness(Gnk, err=0.01, delta=0.1, deterministic=False, k=0)
        bc.run()
        scores = np.asarray(bc.scores(), dtype=np.float64)
        for i, n in enumerate(node_list):
            betweenness[n2i[n]] = float(scores[i])

    feats = {
        "clustering": clustering,
        "closeness": closeness,
        "betweenness": betweenness,
        "pagerank": pagerank,
        "eigenvector": eigenvec,
        "katz": katz,
        "kcore": kcore,
        "harmonic": harmonic,
        "two_hop_wsum": two_hop_wsum,
        "strength": strength,
    }

    feat_mat = np.vstack(
        [np.array([feats[f][n2i[n]] for f in NODE_FEATURES], dtype=np.float64) for n in nodes]
    )
    logger.info(f"Topo features done (weighted={weighted}) N={N} time={perf_counter()-t0:.2f}s")
    return nodes, feat_mat


def scatter_embedding(ax, Z2: np.ndarray, y: np.ndarray, title: str, show_legend: bool = False):
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    if Z2.size == 0:
        ax.text(0.5, 0.5, "Empty", ha="center", va="center")
        return
    m0 = (y == 0)
    m1 = (y == 1)
    if m0.any():
        ax.scatter(Z2[m0, 0], Z2[m0, 1], s=3.2, alpha=0.80, color=COLORS["real"], label="Real")
    if m1.any():
        ax.scatter(Z2[m1, 0], Z2[m1, 1], s=3.2, alpha=0.80, color=COLORS["fake"], label="Fake")
    if show_legend:
        ax.legend(loc="best")


def reduce_2d(method: str, X: np.ndarray) -> np.ndarray:
    if method == "PCA":
        return PCA(n_components=2, random_state=SEED).fit_transform(X)
    if method == "UMAP":
        if _HAS_UMAP:
            reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=20, min_dist=0.1)
            return reducer.fit_transform(X)
        return PCA(n_components=2, random_state=SEED).fit_transform(X)
    if method == "TSNE":
        tsne = TSNE(n_components=2, random_state=SEED, init="pca", learning_rate="auto")
        return tsne.fit_transform(X)
    raise ValueError(method)


def downsample_indices_by_class(y: np.ndarray, max_per_class: int):
    rng = np.random.RandomState(SEED)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    if len(idx0) > max_per_class:
        idx0 = rng.choice(idx0, size=max_per_class, replace=False)
    if len(idx1) > max_per_class:
        idx1 = rng.choice(idx1, size=max_per_class, replace=False)
    return np.concatenate([idx0, idx1])


def make_fig1_2x3(X_unw, y_unw, X_w, y_w):
    idx_u = downsample_indices_by_class(y_unw, MAX_POINTS_PER_CLASS)
    idx_w = downsample_indices_by_class(y_w, MAX_POINTS_PER_CLASS)
    Xu, yu = X_unw[idx_u], y_unw[idx_u]
    Xw, yw = X_w[idx_w], y_w[idx_w]

    idxu_ts = downsample_indices_by_class(y_unw, MAX_TSNE_PER_CLASS)
    idxw_ts = downsample_indices_by_class(y_w, MAX_TSNE_PER_CLASS)

    fig, axes = plt.subplots(2, 3, figsize=(10.0, 6.1), constrained_layout=True)

    Z = reduce_2d("PCA", Xu)
    scatter_embedding(axes[0, 0], Z, yu, "PCA - Unweighted", show_legend=True)
    Z = reduce_2d("UMAP", Xu)
    scatter_embedding(axes[0, 1], Z, yu, ("UMAP" if _HAS_UMAP else "UMAP (PCA fallback)") + " - Unweighted")
    Z = reduce_2d("TSNE", X_unw[idxu_ts])
    scatter_embedding(axes[0, 2], Z, y_unw[idxu_ts], "t-SNE - Unweighted")

    Z = reduce_2d("PCA", Xw)
    scatter_embedding(axes[1, 0], Z, yw, "PCA - Weighted")
    Z = reduce_2d("UMAP", Xw)
    scatter_embedding(axes[1, 1], Z, yw, ("UMAP" if _HAS_UMAP else "UMAP (PCA fallback)") + " - Weighted")
    Z = reduce_2d("TSNE", X_w[idxw_ts])
    scatter_embedding(axes[1, 2], Z, y_w[idxw_ts], "t-SNE - Weighted")

    save_show(fig, "FIG1_TopoFeatures_PCA_UMAP_tSNE_2x3.pdf")


def ccdf_xy(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    if x.size < 12:
        return None, None
    xs = np.sort(x)
    n = xs.size
    y = 1.0 - (np.arange(1, n + 1) / (n + 1.0))
    return xs, y


def _set_exactly_three_log_ticks(ax):
    for which in ["x", "y"]:
        lim = ax.get_xlim() if which == "x" else ax.get_ylim()
        lo = max(float(lim[0]), 1e-300)
        hi = max(float(lim[1]), lo * 1.000001)
        e0 = int(np.floor(np.log10(lo)))
        e1 = int(np.ceil(np.log10(hi)))
        if e0 == e1:
            exps = [e0 - 1, e0, e0 + 1]
        else:
            em = int(np.round((e0 + e1) / 2.0))
            exps = [e0, em, e1]
        ticks = [10.0 ** e for e in exps]
        if which == "x":
            ax.set_xticks(ticks)
            ax.xaxis.set_minor_locator(mticker.NullLocator())
        else:
            ax.set_yticks(ticks)
            ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.grid(alpha=0.15)


def plot_fig2_ccdf_2x5(X_unw_raw, y_unw, X_w_raw, y_w):
    fig, axes = plt.subplots(2, 5, figsize=(10.0, 4.6), constrained_layout=True)
    for i, f in enumerate(NODE_FEATURES):
        ax = axes[i // 5, i % 5]
        ax.set_title(f, pad=2)
        ax.set_box_aspect(1)

        for (Xraw, y, ls, tag) in [
            (X_unw_raw, y_unw, "-", "Unw"),
            (X_w_raw, y_w, "--", "W"),
        ]:
            xr = Xraw[y == 0, i]
            xf = Xraw[y == 1, i]
            xs, yy = ccdf_xy(xr)
            if xs is not None:
                ax.plot(xs, yy, linestyle=ls, linewidth=1.3, color=COLORS["real"], label=f"Real-{tag}")
            xs, yy = ccdf_xy(xf)
            if xs is not None:
                ax.plot(xs, yy, linestyle=ls, linewidth=1.3, color=COLORS["fake"], label=f"Fake-{tag}")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.15)

        if f.lower() == "katz":
            _set_exactly_three_log_ticks(ax)

        if i == 0:
            ax.legend(loc="best", fontsize=8)

        if i // 5 == 1:
            ax.set_xlabel("x")
        if i % 5 == 0:
            ax.set_ylabel("CCDF")

    save_show(fig, "FIG2_TopoFeature_CCDF_PowerLaw_2x5.pdf")


def build_uu_graph(df: pd.DataFrame, weighted: bool) -> nx.Graph:
    rng = np.random.RandomState(SEED)
    prod2reviews: Dict[str, List[Tuple[str, float]]] = {}
    for _, r in df.iterrows():
        u = str(r["Reviewer_id"])
        p = str(r["Product_id"])
        prod2reviews.setdefault(p, []).append((u, float(r["coupling"])))

    G = nx.Graph()
    for _, lst in prod2reviews.items():
        if len(lst) < 2:
            continue
        if len(lst) > MAX_USERS_PER_PRODUCT:
            idx = rng.choice(len(lst), size=MAX_USERS_PER_PRODUCT, replace=False)
            lst = [lst[i] for i in idx]

        n = len(lst)
        for i in range(n):
            ui, wi = lst[i]
            for j in range(i + 1, n):
                uj, wj = lst[j]
                if ui == uj:
                    continue
                w = (wi * wj) if weighted else 1.0
                if G.has_edge(ui, uj):
                    G[ui][uj]["weight"] += w
                else:
                    G.add_edge(ui, uj, weight=w)
    return G


def core_levels_unweighted(G: nx.Graph) -> Dict[str, int]:
    if G.number_of_edges() == 0:
        return {n: 0 for n in G.nodes()}
    return nx.core_number(G)


def core_levels_weighted_quantiles(G: nx.Graph, n_shells: int = 12) -> Dict[str, int]:
    core_w = weighted_strength_core_number(G, weight="weight")
    if not core_w:
        return {n: 0 for n in G.nodes()}
    nodes = list(core_w.keys())
    vals = np.array([core_w[n] for n in nodes], dtype=float)
    qs = np.quantile(vals, np.linspace(0, 1, n_shells + 1))
    qs = np.unique(qs)
    lvl = {}
    for n, v in zip(nodes, vals):
        k = int(np.searchsorted(qs, v, side="right") - 1)
        k = max(0, min(k, len(qs) - 2))
        lvl[n] = k
    for n in G.nodes():
        lvl.setdefault(n, 0)
    return lvl


def sample_core_mid_peri(G: nx.Graph, levels: Dict[str, int], n_core: int, n_mid: int, n_peri: int) -> List[str]:
    rng = np.random.RandomState(SEED)
    nodes = list(G.nodes())
    if not nodes:
        return []
    lv = np.array([levels.get(n, 0) for n in nodes], dtype=int)
    K = int(lv.max()) if lv.size else 0
    if K == 0:
        take = min(len(nodes), n_core + n_mid + n_peri)
        return list(rng.choice(nodes, size=take, replace=False))

    core_th = max(1, int(np.quantile(lv, 0.85)))
    peri_th = int(np.quantile(lv, 0.20))

    core = [n for n, k in zip(nodes, lv) if k >= core_th]
    peri = [n for n, k in zip(nodes, lv) if k <= peri_th]
    mid = [n for n in nodes if n not in set(core) and n not in set(peri)]

    def pick(lst, k):
        if len(lst) <= k:
            return lst
        return list(rng.choice(lst, size=k, replace=False))

    return pick(core, n_core) + pick(mid, n_mid) + pick(peri, n_peri)


def draw_coreperi_schematic(
    ax,
    G: nx.Graph,
    user_labels: pd.Series,
    title: str,
    weighted: bool,
    add_legend: bool = False,
):
    ax.set_title(title)
    ax.axis("off")
    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "Empty", ha="center", va="center")
        return

    levels = core_levels_unweighted(G) if not weighted else core_levels_weighted_quantiles(G, n_shells=12)

    keep_nodes = sample_core_mid_peri(G, levels, SCHEM_CORE_N, SCHEM_MID_N, SCHEM_PERI_N)
    H = G.subgraph(keep_nodes).copy()

    rng = np.random.RandomState(SEED)
    nodes = list(H.nodes())
    lv = np.array([levels.get(n, 0) for n in nodes], dtype=int)
    K = max(1, int(lv.max()))

    pos = {}
    for n in nodes:
        k = levels.get(n, 0)
        t = 1.0 - (k / K)
        if t < 0.33:
            r = 0.25
        elif t < 0.66:
            r = 0.55
        else:
            r = 0.85
        ang = 2 * np.pi * rng.rand()
        rr = r * (0.85 + 0.3 * rng.rand())
        pos[n] = np.array([rr * np.cos(ang), rr * np.sin(ang)])

    nx.draw_networkx_edges(H, pos, ax=ax, width=0.35, alpha=0.20, edge_color=COLORS["edge"])

    labs = np.array([int(user_labels.get(str(n), 0)) for n in nodes], dtype=int)
    real_nodes = [n for n, f in zip(nodes, labs) if f == 0]
    fake_nodes = [n for n, f in zip(nodes, labs) if f == 1]
    nx.draw_networkx_nodes(H, pos, nodelist=real_nodes, node_size=18, node_color=COLORS["real"], ax=ax, linewidths=0)
    nx.draw_networkx_nodes(H, pos, nodelist=fake_nodes, node_size=18, node_color=COLORS["fake"], ax=ax, linewidths=0)

    if add_legend:
        add_real_fake_legend(ax, loc="lower left")


def plot_fig7_coreperi(Guu_unw: nx.Graph, Guu_w: nx.Graph, user_labels: pd.Series):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), constrained_layout=True)
    draw_coreperi_schematic(axes[0], Guu_unw, user_labels, "Core–Periphery schematic - Unweighted", weighted=False, add_legend=True)
    draw_coreperi_schematic(axes[1], Guu_w, user_labels, "Core–Periphery schematic - Weighted", weighted=True, add_legend=True)
    save_show(fig, "FIG7_CorePeriphery_Schematic_1x2.pdf")


def main():
    setup_style()
    set_seed(SEED)
    _ensure_dir(OUT_DIR)

    logger.info(f"Reading Excel: {DATA_FILE} sheet={SHEET_NAME}")
    df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
    df = preprocess_df(df)
    if 0 < float(SAMPLE_FRAC) < 1.0:
        df = df.sample(frac=float(SAMPLE_FRAC), random_state=SEED).reset_index(drop=True)

    user_labels = user_labels_from_df(df)

    B, users, prods = build_bipartite_multigraph(df)
    edges_merged = collapse_multiedges_to_simple_edges(B)

    nodes_unw, feat_unw = compute_node_features_from_edges(edges_merged, weighted=False)
    nodes_w, feat_w = compute_node_features_from_edges(edges_merged, weighted=True)

    user_set = set(map(str, users))

    def extract_user_Xy(nodes, feat_mat):
        idx = [i for i, n in enumerate(nodes) if str(n) in user_set]
        user_nodes = [nodes[i] for i in idx]
        Xraw = feat_mat[idx, :]
        y = np.array([int(user_labels.get(str(n), 0)) for n in user_nodes], dtype=int)
        return user_nodes, Xraw, y

    _, X_unw_raw, y_unw = extract_user_Xy(nodes_unw, feat_unw)
    _, X_w_raw, y_w = extract_user_Xy(nodes_w, feat_w)

    X_unw = StandardScaler().fit_transform(X_unw_raw).astype(np.float32)
    X_w = StandardScaler().fit_transform(X_w_raw).astype(np.float32)

    Guu_unw = build_uu_graph(df, weighted=False)
    Guu_w = build_uu_graph(df, weighted=True)

    logger.info(f"UU(unw): nodes={Guu_unw.number_of_nodes()} edges={Guu_unw.number_of_edges()}")
    logger.info(f"UU(w):   nodes={Guu_w.number_of_nodes()} edges={Guu_w.number_of_edges()}")

    if MAKE_FIG1_EMB_2x3:
        make_fig1_2x3(X_unw, y_unw, X_w, y_w)

    if MAKE_FIG2_CCDF_2x5:
        plot_fig2_ccdf_2x5(X_unw_raw, y_unw, X_w_raw, y_w)

    if MAKE_FIG7_COREPERI_SCHEM:
        plot_fig7_coreperi(Guu_unw, Guu_w, user_labels)

    logger.info(f"DONE. PDFs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
