# -*- coding: utf-8 -*-
import os
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import warnings, logging, math, random, re
from typing import Dict, Tuple, List, Optional
from contextlib import nullcontext
from time import perf_counter

warnings.filterwarnings("ignore", category=Warning, module=r"torch_geometric\.typing")
warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
logging.getLogger("numexpr").setLevel(logging.WARNING)

from tqdm.auto import tqdm

# ——— Logging  ———
class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

handler = TqdmLoggingHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
root = logging.getLogger()
root.setLevel(logging.INFO)
root.handlers = [handler]
root.propagate = False
logger = logging.getLogger(__name__)

# ——— ANSI Colors ———
ANSI_BLUE   = "\033[34m"
ANSI_GREEN  = "\033[32m"
ANSI_RED    = "\033[31m"
ANSI_ORANGE = "\033[33m"
ANSI_RESET  = "\033[0m"

import numpy as np
import pandas as pd
import networkx as nx
import networkit as nk

try:
    import scipy.sparse as sp
    from scipy.sparse.csgraph import dijkstra as cs_dijkstra, shortest_path as cs_shortest_path
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

import multiprocessing as mp
from multiprocessing import shared_memory
from collections import deque
import heapq

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score

from torch_geometric.utils import softmax
from torch_geometric.nn import MessagePassing, GATConv, GCNConv

# ============================================================
# torch_sparse
# ============================================================
try:
    from torch_sparse import coalesce as ts_coalesce
    from torch_sparse import spspmm as ts_spspmm
    from torch_sparse import SparseTensor
except Exception as e:
    raise ImportError(
        "need install torch_sparse (SparseTensor + spspmm/coalesce + GPU neighbor sampling)"
    ) from e

# ============================================================
# torch_sparse CUDA probe
# ============================================================
TS_CUDA_OK = False          # for coalesce/spspmm
TS_SAMPLE_CUDA_OK = False   # for SparseTensor.sample_adj

def probe_torch_sparse_cuda():
    """Run torch_sparse CUDA kernel probes ONCE in main process."""
    global TS_CUDA_OK, TS_SAMPLE_CUDA_OK

    TS_CUDA_OK = False
    TS_SAMPLE_CUDA_OK = False

    if not torch.cuda.is_available():
        logger.info("CUDA not available; torch_sparse will run on CPU.")
        return

    # ---- probe coalesce/spspmm ----
    try:
        dev = torch.device("cuda")
        idxA = torch.tensor([[0, 1], [1, 0]], device=dev, dtype=torch.long)
        valA = torch.tensor([1.0, 1.0], device=dev, dtype=torch.float32)
        idxB = torch.tensor([[0, 1], [1, 0]], device=dev, dtype=torch.long)
        valB = torch.tensor([1.0, 1.0], device=dev, dtype=torch.float32)

        _i, _v = ts_coalesce(idxA, valA, m=2, n=2)
        _i2, _v2 = ts_spspmm(_i, _v, idxB, valB, 2, 2, 2)
        TS_CUDA_OK = True
        logger.info("torch_sparse CUDA kernels (coalesce/spspmm): OK")
    except Exception as e:
        TS_CUDA_OK = False
        logger.warning(f"torch_sparse CUDA kernels (coalesce/spspmm): FAILED -> fallback to CPU. Reason: {e}")

    # ---- probe sample_adj ----
    try:
        dev = torch.device("cuda")
        row = torch.tensor([0, 1], device=dev, dtype=torch.long)
        col = torch.tensor([1, 0], device=dev, dtype=torch.long)
        val = torch.tensor([1.0, 1.0], device=dev, dtype=torch.float32)
        st = SparseTensor(row=row, col=col, value=val, sparse_sizes=(2, 2)).coalesce()
        seed = torch.tensor([0, 1], device=dev, dtype=torch.long)
        _sa, _nid = st.sample_adj(seed, 1, replace=False)
        TS_SAMPLE_CUDA_OK = True
        logger.info("torch_sparse CUDA kernels (SparseTensor.sample_adj): OK")
    except Exception as e:
        TS_SAMPLE_CUDA_OK = False
        logger.warning(f"torch_sparse CUDA kernels (SparseTensor.sample_adj): FAILED -> will sample on CPU. Reason: {e}")

def _sparse_tensor_device(adj: SparseTensor) -> torch.device:
    try:
        return adj.device()
    except Exception:
        try:
            return adj.storage.row().device  # best effort
        except Exception:
            return torch.device("cpu")

# ============================================================
# dtype-safe index_add helpers (BF16/FP32)
# ============================================================
def index_add_dtype(out: torch.Tensor, dim: int, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    if src.dtype != out.dtype:
        src = src.to(out.dtype)
    return out.index_add(dim, index, src)

def index_add__dtype(out: torch.Tensor, dim: int, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    if src.dtype != out.dtype:
        src = src.to(out.dtype)
    out.index_add_(dim, index, src)
    return out

def autocast_ctx(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=dtype, enabled=True)
    return nullcontext()

# ============================================================
# ========== Adjustable parameters ==========
# ============================================================
SEED_START, SEED_STOP = 42, 51
SEEDS = list(range(SEED_START, SEED_STOP + 1))

SAMPLE_FRAC = 0.01

# ===== CPU parallelism for feature computation =====
CPU_FEATURE_WORKERS = 6
CPU_FEATURE_CHUNK   = 2048

# Train/VAL/TEST:
TRAIN_FRAC = 0.8
VAL_FRAC   = 0.1
TEST_FRAC  = 0.1
assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-9

MAX_EPOCHS_GNN = 100
PATIENCE = 10
MIN_DELTA = 1e-4

LR = 0.01
WEIGHT_DECAY = 0.01

HIDDEN = 128
HEADS = 4
DROPOUT = 0.005

MODEL_ORDER = [
    'RGCN', 'HAN', 'RSHN', 'GTN', 'HetGNN', 'HGT', 'MAGNN', 'HetSANN', 'SimpleHGN', 'HGOT', 'SE-HTGNN'
]

MODE_LABELS = ['MODE-A', 'MODE-B', 'MODE-C']
MODE_ANSI = {'MODE-A': "", 'MODE-B': ANSI_BLUE, 'MODE-C': ANSI_ORANGE}

# ============================================================
# TF32 + Disable deterministic (A100 acceleration)
# ============================================================
def enable_a100_fast_math():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# ============================================================
# Data preprocess
# ============================================================
def stats_block(df, tag):
    n_users = df['Reviewer_id'].nunique()
    user_labels = df.groupby('Reviewer_id')['Label_user'].first()
    n_fake_u = (pd.to_numeric(user_labels, errors='coerce').fillna(0).astype(int) == 1).sum()
    n_reviews = len(df)
    n_fake_r = (pd.to_numeric(df['Label'], errors='coerce').fillna(0).astype(int) == 1).sum()
    n_edges = n_reviews
    n_products = df['Product_id'].nunique()
    logger.info(
        f"{tag}: Users={n_users}, Fake users={n_fake_u} ({(n_fake_u / max(n_users, 1)):.2%}); "
        f"Reviews={n_reviews}, Fake reviews={n_fake_r} ({(n_fake_r / max(n_reviews, 1)):.2%}); "
        f"Edges={n_edges}; Products={n_products}"
    )

def preprocess_df(df):
    df = df.copy()
    df['Reviewer_id'] = df['Reviewer_id'].astype(str)
    df['Product_id'] = df['Product_id'].astype(str)
    df['Date'] = pd.to_datetime(df['Date'])
    df['rating_norm'] = (df['Rating'] - 1) / 4.0
    df['coupling'] = (1 - np.abs(df['Sentiment'] - df['rating_norm'])).clip(0, 1)
    df['Label'] = pd.to_numeric(df['Label'], errors='coerce').fillna(0).astype(int)
    df['Label_user'] = pd.to_numeric(df['Label_user'], errors='coerce').fillna(0).astype(int)
    return df

def balance_by_day(df, seed=42):
    parts = []
    for _, grp in df.groupby(df['Date'].dt.date):
        real = grp[grp['Label'] == 0]
        fake = grp[grp['Label'] == 1]
        m = min(len(real), len(fake))
        if m > 0:
            parts.append(real.sample(m, random_state=seed))
            parts.append(fake.sample(m, random_state=seed))
    if not parts:
        raise ValueError("No available data, please check the label/date.")
    return pd.concat(parts, ignore_index=True)

# ============================================================
# Graph structural node features (CPU)
# ============================================================
NODE_FEATURES = [
    "clustering", "closeness", "betweenness", "pagerank", "eigenvector",
    "katz", "kcore", "harmonic", "two_hop_wsum", "strength"
]

def feature_complexity_strings_symbolic(weighted: bool) -> Dict[str, str]:
    logN = "logN"
    sp_all = "O(N*(E+N))" if not weighted else f"O(N*(E {logN}))"
    bt_exact = "O(N*E)" if not weighted else f"O(N*E {logN})"
    power_iter = "O(I*E)"
    deg2 = "O(Σ deg(v)^2)"
    return {
        "strength": "O(E)",
        "closeness": sp_all,
        "harmonic":  sp_all,
        "pagerank":  power_iter,
        "eigenvector": power_iter,
        "katz": power_iter,
        "betweenness": bt_exact,
        "kcore": "O(E)",
        "two_hop_wsum": deg2,
        "clustering": deg2,
    }

# ============================================================
# CPU multi-core parallel node feature computation
# ============================================================
LANDMARK_K = 256
BETW_K = 128
SP_CHUNK = 16
BETW_CHUNK = 8

def collapse_multiedges_to_simple_edges(B: nx.MultiGraph):
    """Merge MultiGraph edges into weighted simple edges list (u,v,w)."""
    esum = {}
    for u, v, d in B.edges(data=True):
        key = tuple(sorted((u, v)))
        esum[key] = esum.get(key, 0.0) + float(d.get("weight", 1.0))
    return [(u, v, w) for (u, v), w in esum.items()]

def _build_csr_from_edges(edges_merged, weighted: bool):
    nodes = []
    for u, v, _ in edges_merged:
        nodes.append(u); nodes.append(v)
    nodes = list(dict.fromkeys(nodes))  # stable unique
    n2i = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)

    m = len(edges_merged)
    rows = np.empty(m * 2, dtype=np.int64)
    cols = np.empty(m * 2, dtype=np.int64)
    data = np.empty(m * 2, dtype=np.float64)

    for k, (u, v, w) in enumerate(edges_merged):
        i = n2i[u]; j = n2i[v]
        val = float(w) if weighted else 1.0
        rows[2*k]   = i; cols[2*k]   = j; data[2*k]   = val
        rows[2*k+1] = j; cols[2*k+1] = i; data[2*k+1] = val

    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    A.sum_duplicates()
    A.eliminate_zeros()
    return A, nodes, n2i

# ----------------------------
# Sparse power iterations
# ----------------------------
def _pagerank_power(A: sp.csr_matrix, max_iter=50, tol=1e-6, damping=0.85):
    N = A.shape[0]
    out = np.asarray(A.sum(axis=1)).ravel()
    out[out == 0] = 1.0
    P = sp.diags(1.0 / out) @ A  # row-stochastic
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

# ----------------------------
# Shared CSR for workers
# ----------------------------
def _shm_pack(arr: np.ndarray):
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    view[:] = arr
    meta = (shm.name, arr.shape, arr.dtype.str)
    return shm, meta

def _shm_unpack(meta):
    name, shape, dtype_str = meta
    shm = shared_memory.SharedMemory(name=name)
    arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
    return shm, arr

_W_indptr = None
_W_indices = None
_W_data = None
_W_N = 0
_W_weighted = False
_W_shms = None

def _init_worker_csr(indptr_meta, indices_meta, data_meta, N: int, weighted: bool):
    global _W_indptr, _W_indices, _W_data, _W_N, _W_weighted, _W_shms
    _W_shms = []
    shm1, indptr = _shm_unpack(indptr_meta);  _W_shms.append(shm1)
    shm2, indices = _shm_unpack(indices_meta); _W_shms.append(shm2)
    shm3, data = _shm_unpack(data_meta); _W_shms.append(shm3)
    _W_indptr, _W_indices, _W_data = indptr, indices, data
    _W_N = int(N)
    _W_weighted = bool(weighted)

def _single_source_sp_stats_reuse(s: int, dist_buf: np.ndarray):
    N = _W_N
    indptr, indices, data = _W_indptr, _W_indices, _W_data
    weighted = _W_weighted

    if not weighted:
        dist_buf.fill(-1)
        dist_buf[s] = 0
        Q = deque([s])
        while Q:
            v = Q.popleft()
            rs, re = indptr[v], indptr[v + 1]
            for w in indices[rs:re]:
                w = int(w)
                if dist_buf[w] < 0:
                    dist_buf[w] = dist_buf[v] + 1
                    Q.append(w)

        mask = dist_buf >= 0
        d = dist_buf.astype(np.float64, copy=False)

    else:
        dist_buf.fill(np.inf)
        dist_buf[s] = 0.0
        Q = [(0.0, s)]
        while Q:
            dv, v = heapq.heappop(Q)
            if dv > dist_buf[v] + 1e-12:
                continue
            rs, re = indptr[v], indptr[v + 1]
            neigh = indices[rs:re]
            wts = data[rs:re]
            for w, wlen in zip(neigh, wts):
                w = int(w)
                nd = dv + float(wlen)
                if nd < dist_buf[w] - 1e-12:
                    dist_buf[w] = nd
                    heapq.heappush(Q, (nd, w))

        mask = np.isfinite(dist_buf)
        d = dist_buf

    sumd = np.where(mask, d, 0.0)
    cntd = mask.astype(np.float64)
    inv = np.zeros_like(d, dtype=np.float64)
    np.divide(1.0, d, out=inv, where=(mask & (d > 0)))
    cnt_inv = (mask & (d > 0)).astype(np.float64)

    return sumd, cntd, inv, cnt_inv

def _single_source_sp_stats(s: int):
    if _W_weighted:
        buf = np.empty(_W_N, dtype=np.float64)
    else:
        buf = np.empty(_W_N, dtype=np.int32)
    return _single_source_sp_stats_reuse(int(s), buf)

def _worker_landmark_batch(src_list):
    N = _W_N
    sumd = np.zeros(N, dtype=np.float64)
    cntd = np.zeros(N, dtype=np.float64)
    sum_inv = np.zeros(N, dtype=np.float64)
    cnt_inv = np.zeros(N, dtype=np.float64)
    dist_buf = np.empty(N, dtype=(np.float64 if _W_weighted else np.int32))

    for s in src_list:
        a, b, c, d_ = _single_source_sp_stats_reuse(int(s), dist_buf)
        sumd += a
        cntd += b
        sum_inv += c
        cnt_inv += d_

    return sumd, cntd, sum_inv, cnt_inv

def _brandes_one_source(s: int):
    N = _W_N
    indptr, indices, data = _W_indptr, _W_indices, _W_data
    weighted = _W_weighted
    eps = 1e-12

    S = []
    P = [[] for _ in range(N)]
    sigma = np.zeros(N, dtype=np.float64)
    sigma[s] = 1.0

    if not weighted:
        dist = np.full(N, -1, dtype=np.int32)
        dist[s] = 0
        Q = deque([s])

        while Q:
            v = Q.popleft()
            S.append(v)
            rs, re = indptr[v], indptr[v + 1]
            for w in indices[rs:re]:
                w = int(w)
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    Q.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)
    else:
        dist = np.full(N, np.inf, dtype=np.float64)
        dist[s] = 0.0
        Q = [(0.0, s)]

        while Q:
            dv, v = heapq.heappop(Q)
            if dv > dist[v] + eps:
                continue
            S.append(v)
            rs, re = indptr[v], indptr[v + 1]
            neigh = indices[rs:re]
            wts = data[rs:re]
            for w, wlen in zip(neigh, wts):
                w = int(w)
                nd = dv + float(wlen)
                if nd < dist[w] - eps:
                    dist[w] = nd
                    heapq.heappush(Q, (nd, w))
                    sigma[w] = sigma[v]
                    P[w] = [v]
                elif abs(nd - dist[w]) <= eps:
                    sigma[w] += sigma[v]
                    P[w].append(v)

    delta = np.zeros(N, dtype=np.float64)
    betw = np.zeros(N, dtype=np.float64)
    while S:
        w = S.pop()
        for v in P[w]:
            if sigma[w] > 0:
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
        if w != s:
            betw[w] += delta[w]
    return betw

def _worker_betw_batch(src_list):
    N = _W_N
    acc = np.zeros(N, dtype=np.float64)
    for s in src_list:
        acc += _brandes_one_source(int(s))
    return acc

def _chunk_list(arr, chunk):
    arr = list(arr)
    for i in range(0, len(arr), chunk):
        yield arr[i:i + chunk]

def compute_node_features_fast_parallel_from_edges(
    edges_merged,
    weighted: bool,
    n_jobs: int,
    sp_chunk: int,
    betw_chunk: int,
    landmark_k: int = LANDMARK_K,
    betw_k: int = BETW_K,
):

    if not _HAS_SCIPY:
        raise RuntimeError("SciPy not available.")

    t0 = perf_counter()
    timings = {}

    # ---- build adjacency (CSR) ----
    t = perf_counter()
    A_w, nodes, n2i = _build_csr_from_edges(edges_merged, weighted=weighted)
    N = A_w.shape[0]
    timings["_build_A"] = perf_counter() - t

    # binary mask (exclude 1-hop)
    A_bin = A_w.copy()
    A_bin.data = np.ones_like(A_bin.data)

    # ---- strength ----
    t = perf_counter()
    strength = np.asarray(A_w.sum(axis=1)).ravel()
    timings["strength"] = perf_counter() - t

    # ---- pagerank/eigen/katz (single-process sparse) ----
    t = perf_counter()
    pagerank = _pagerank_power(A_w, max_iter=50, tol=1e-6, damping=0.85)
    timings["pagerank"] = perf_counter() - t

    t = perf_counter()
    eigenvec = _eigenvector_power(A_w, max_iter=80, tol=1e-6)
    timings["eigenvector"] = perf_counter() - t

    t = perf_counter()
    katz = _katz_iter(A_w, alpha=0.005, beta=1.0, max_iter=80, tol=1e-6)
    timings["katz"] = perf_counter() - t

    # ---- build distance CSR for SP/betweenness ----
    # weighted: distance = 1/weight ; unweighted: all ones
    t = perf_counter()
    if weighted:
        A_dist = A_w.copy()
        A_dist.data = 1.0 / (A_w.data + 1e-12)
    else:
        A_dist = A_bin.copy()

    indptr = A_dist.indptr.astype(np.int64, copy=False)
    indices = A_dist.indices.astype(np.int32, copy=False)
    data = A_dist.data.astype(np.float64, copy=False)
    timings["_build_dist"] = perf_counter() - t

    # ---- prepare shm + pool ----
    t = perf_counter()
    shm_indptr, indptr_meta = _shm_pack(indptr)
    shm_indices, indices_meta = _shm_pack(indices)
    shm_data, data_meta = _shm_pack(data)
    timings["_shm_pack"] = perf_counter() - t

    ctx = mp.get_context("spawn")
    pool = None

    # - weighted=False: use weight=1.0 everywhere (unweighted topology)
    # - weighted=True : keep merged weights
    edges_for_graph = edges_merged if weighted else [(u, v, 1.0) for (u, v, _w) in edges_merged]

    try:
        pool = ctx.Pool(
            processes=int(max(1, n_jobs)),
            initializer=_init_worker_csr,
            initargs=(indptr_meta, indices_meta, data_meta, int(N), bool(weighted)),
        )

        t = perf_counter()
        rng = np.random.RandomState(42)
        L = min(int(landmark_k), N)
        landmarks = rng.choice(N, size=L, replace=False)

        closeness, harmonic = closeness_harmonic_landmarks_csgraph(
            A_dist=A_dist,
            landmarks=landmarks,
            weighted=weighted,
            chunk_size=int(max(1, sp_chunk)),
            eps=1e-12,
        )
        timings["closeness"] = perf_counter() - t
        timings["harmonic"] = 0.0

        t = perf_counter()
        A2 = (A_w @ A_w).tocsr()
        A2.sum_duplicates()
        A2.eliminate_zeros()
        timings["_A2"] = perf_counter() - t

        t = perf_counter()
        A2_excl = A2.copy()
        A2_excl.setdiag(0.0)
        A2_excl.eliminate_zeros()
        A2_excl = A2_excl - A2_excl.multiply(A_bin)
        A2_excl.eliminate_zeros()
        timings["_A2_excl"] = perf_counter() - t

        t = perf_counter()
        two_hop_wsum = np.asarray(A2_excl.sum(axis=1)).ravel()
        timings["two_hop_wsum"] = perf_counter() - t

        t = perf_counter()
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
        timings["clustering"] = perf_counter() - t

        t = perf_counter()

        G_nk = build_nk_graph_from_edges_merged(
            edges_merged=edges_for_graph,
            n2i=n2i,
            N=N,
            weighted_distance=bool(weighted),
            eps=1e-12
        )

        bet_method = "kadabra" if not weighted else "exact"
        betweenness = betweenness_networkit(
            G_nk,
            method=bet_method,
            nSamples=int(betw_k),
            err=0.01,
            delta=0.1,
            normalized=False,
            threads=CPU_FEATURE_WORKERS,
            seed=42,
        )

        timings["betweenness"] = perf_counter() - t

    finally:
        if pool is not None:
            pool.close()
            pool.join()

        for shm in (shm_indptr, shm_indices, shm_data):
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass

    Gs = nx.Graph()
    Gs.add_weighted_edges_from(edges_for_graph, weight="weight")

    t = perf_counter()
    try:
        if not weighted:
            core_num = nx.core_number(Gs)  # unweighted k-core
        else:
            core_num = weighted_strength_core_number(Gs, weight="weight")  # weighted strength-core

        kcore = np.zeros(N, dtype=np.float64)
        for _node, _k in core_num.items():
            if _node in n2i:
                kcore[n2i[_node]] = float(_k)
    except Exception:
        kcore = np.zeros(N, dtype=np.float64)
    timings["kcore"] = perf_counter() - t

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

    for f in NODE_FEATURES:
        timings.setdefault(f, 0.0)

    feat_dict = {n: [float(feats[f][n2i[n]]) for f in NODE_FEATURES] for n in nodes}
    timings["_total"] = perf_counter() - t0
    return Gs, feat_dict, timings

def compute_node_features_parallel_from_edges(edges_merged, weighted: bool, n_jobs=None, chunk_size=None):
    if n_jobs is None:
        n_jobs = int(CPU_FEATURE_WORKERS) if "CPU_FEATURE_WORKERS" in globals() else 6

    if chunk_size is None:
        sp_chunk = SP_CHUNK
        betw_chunk = BETW_CHUNK
    else:
        c = int(max(1, min(chunk_size, 128)))
        sp_chunk = max(4, c // 2)
        betw_chunk = max(2, c // 4)

    return compute_node_features_fast_parallel_from_edges(
        edges_merged,
        weighted=weighted,
        n_jobs=int(n_jobs),
        sp_chunk=int(sp_chunk),
        betw_chunk=int(betw_chunk),
        landmark_k=LANDMARK_K,
        betw_k=BETW_K,
    )

def build_bipartite_graph(df):
    users = df['Reviewer_id'].unique()
    prods = df['Product_id'].unique()
    umap = {u: i for i, u in enumerate(users)}
    pmap = {p: i for i, p in enumerate(prods)}
    B = nx.MultiGraph()
    B.add_nodes_from(users, bipartite='user')
    B.add_nodes_from(prods, bipartite='product')

    rows, cols, w = [], [], []
    for _, r in df.iterrows():
        B.add_edge(r['Reviewer_id'], r['Product_id'], weight=float(r['coupling']))
        rows.append(umap[r['Reviewer_id']])
        cols.append(pmap[r['Product_id']])
        w.append(float(r['coupling']))
    return B, users, prods, rows, cols, w

def _features_to_tensor(feat_dict, nodes):
    arr = np.vstack([feat_dict[n] for n in nodes])
    arr = StandardScaler().fit_transform(arr)
    return torch.tensor(arr, dtype=torch.float32)

def closeness_harmonic_landmarks_csgraph(
    A_dist: "sp.csr_matrix",
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
        src = landmarks[i:i + int(max(1, chunk_size))]

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

def build_nk_graph_from_edges_merged(edges_merged, n2i, N: int, weighted_distance: bool, eps: float = 1e-12):
    G = nk.Graph(N, weighted=bool(weighted_distance), directed=False)
    if weighted_distance:
        for u, v, w in edges_merged:
            iu = n2i[u]; iv = n2i[v]
            dist = 1.0 / (float(w) + eps)
            G.addEdge(iu, iv, dist)
    else:
        for u, v, _w in edges_merged:
            iu = n2i[u]; iv = n2i[v]
            G.addEdge(iu, iv)
    return G

def betweenness_networkit(
    G: "nk.Graph",
    method: str = "kadabra",   # "kadabra" / "estimate" / "approx" / "exact"
    nSamples: int = 128,       # estimate
    err: float = 0.01,         # kadabra/approx
    delta: float = 0.1,        # kadabra/approx
    normalized: bool = False,  # exact/estimate
    threads: int = 0,
    seed: int = 42,
):

    if threads and threads > 0:
        nk.engineering.setNumberOfThreads(int(threads))
    nk.engineering.setSeed(int(seed), True)

    method = method.lower()
    if method == "estimate":
        bc = nk.centrality.EstimateBetweenness(G, int(nSamples), normalized=bool(normalized), parallel_flag=True)
    elif method == "approx":
        bc = nk.centrality.ApproxBetweenness(G, epsilon=float(err), delta=float(delta), universalConstant=1.0)
    elif method == "kadabra":
        bc = nk.centrality.KadabraBetweenness(G, err=float(err), delta=float(delta), deterministic=False, k=0)
    elif method == "exact":
        bc = nk.centrality.Betweenness(G, normalized=bool(normalized), computeEdgeCentrality=False)
    else:
        raise ValueError(f"Unknown method={method}")

    bc.run()
    scores = np.asarray(bc.scores(), dtype=np.float64)
    return scores

def weighted_strength_core_number(G: nx.Graph, weight: str = "weight") -> Dict:
    if G.number_of_nodes() == 0:
        return {}
    cur = {u: 0.0 for u in G.nodes()}
    for u in G.nodes():
        cur[u] = float(G.degree(u, weight=weight))

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

# ============================================================
# Pretty table utils
# ============================================================
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
def strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)

try:
    from wcwidth import wcswidth as _wcswidth
    _HAS_WCWIDTH = True
except Exception:
    _HAS_WCWIDTH = False
    _wcswidth = None

def vis_width(s: str) -> int:
    ss = strip_ansi(s)
    if _HAS_WCWIDTH:
        w = _wcswidth(ss)
        return w if w >= 0 else len(ss)
    return len(ss)

def pad_left(s: str, width: int) -> str:
    pad = width - vis_width(s)
    if pad <= 0:
        return s
    return s + (" " * pad)

def console_feature_profile(times_unw: Dict[str, float], times_w: Dict[str, float],
                            comp_unw: Dict[str, str], comp_w: Dict[str, str]):
    headers = ["Feature", "Complexity(unw)", "Time_unw(s)", "Complexity(w)", "Time_w(s)"]
    rows = []
    for f in NODE_FEATURES:
        cu = comp_unw.get(f, "-")
        cw = comp_w.get(f, "-")
        tu = float(times_unw.get(f, 0.0))
        tw = float(times_w.get(f, 0.0))
        rows.append([f, cu, f"{tu:.4f}", cw, f"{tw:.4f}"])

    ncol = len(headers)
    col_widths = []
    for i in range(ncol):
        maxw = vis_width(headers[i])
        for r in rows:
            maxw = max(maxw, vis_width(str(r[i])))
        col_widths.append(maxw)

    print("\n===== Node Feature Build Profile =====")
    head_line = '|' + '|'.join(f" {pad_left(headers[i], col_widths[i])} " for i in range(ncol)) + '|'
    sep_line  = '|' + '|'.join('-' * (col_widths[i] + 2) for i in range(ncol)) + '|'
    print(head_line)
    print(sep_line)
    for r in rows:
        line = '|' + '|'.join(f" {pad_left(str(r[i]), col_widths[i])} " for i in range(ncol)) + '|'
        print(line)

def precompute_feature_cache(B, users, prods):
    edges_merged = collapse_multiedges_to_simple_edges(B)
    _, feat_unw, t_unw = compute_node_features_parallel_from_edges(edges_merged, weighted=False)
    _, feat_w, t_w = compute_node_features_parallel_from_edges(edges_merged, weighted=True)
    comp_unw = feature_complexity_strings_symbolic(False)
    comp_w = feature_complexity_strings_symbolic(True)
    console_feature_profile(t_unw, t_w, comp_unw, comp_w)

    return {
        'unweighted': {
            'X_u': _features_to_tensor(feat_unw, users),
            'X_p': _features_to_tensor(feat_unw, prods)
        },
        'weighted': {
            'X_u': _features_to_tensor(feat_w, users),
            'X_p': _features_to_tensor(feat_w, prods)
        }
    }

# ============================================================
# Sparse helpers (torch_sparse) with CPU fallback
# ============================================================
def _ts_maybe_cpu(*tensors: torch.Tensor):
    if not tensors:
        return tensors, (lambda x: x), torch.device("cpu")

    dev = tensors[0].device
    need_cpu = (dev.type == "cuda") and (not TS_CUDA_OK)
    if not need_cpu:
        return tensors, (lambda x: x), dev

    tensors_cpu = tuple(t.detach().cpu() for t in tensors)
    def move_back(x):
        if isinstance(x, torch.Tensor):
            return x.to(dev, non_blocking=True)
        return x
    return tensors_cpu, move_back, dev

def coalesce_idx_val(idx: torch.Tensor, val: torch.Tensor, m: int, n: int):
    if idx.numel() == 0:
        return idx, val

    (idx2, val2), back, _ = _ts_maybe_cpu(idx, val)
    idx_out, val_out = ts_coalesce(idx2, val2, m=m, n=n)
    return back(idx_out), back(val_out)

def add_self_loops_with_weight(idx: torch.Tensor, val: torch.Tensor, N: int, fill: float = 1.0):
    dev = idx.device
    self_idx = torch.arange(N, device=dev, dtype=torch.long)
    sl = torch.stack([self_idx, self_idx], dim=0)
    sv = torch.full((N,), float(fill), device=dev, dtype=val.dtype)
    idx2 = torch.cat([idx, sl], dim=1) if idx.numel() else sl
    val2 = torch.cat([val, sv], dim=0) if val.numel() else sv
    return coalesce_idx_val(idx2, val2, N, N)

def sparse_mm(idxA, valA, idxB, valB, m, k, n):
    if idxA.numel() == 0 or idxB.numel() == 0:
        dev = idxA.device
        return (torch.empty((2, 0), dtype=torch.long, device=dev),
                torch.empty((0,), dtype=valA.dtype, device=dev))

    (idxA2, valA2, idxB2, valB2), back, _ = _ts_maybe_cpu(idxA, valA, idxB, valB)
    idxC, valC = ts_spspmm(idxA2, valA2, idxB2, valB2, m, k, n)
    idxC, valC = ts_coalesce(idxC, valC, m=m, n=n)
    return back(idxC), back(valC)

def prune_topk_global(idx: torch.Tensor, val: torch.Tensor, k: Optional[int]):
    if k is None or val.numel() <= k:
        return idx, val
    topk = torch.topk(val, k=k, largest=True, sorted=False)
    keep = topk.indices
    return idx[:, keep], val[keep]

# ============================================================
# Meta-path UPU / UU4（GPU + torch_sparse / CPU fallback）
# ============================================================
def build_upu_metapath_edges_gpu(num_users, num_prods, rows_t, cols_t, w_t):
    i_up = torch.stack([rows_t, cols_t], dim=0)
    v_up = w_t
    i_up, v_up = coalesce_idx_val(i_up, v_up, num_users, num_prods)
    i_pu = torch.stack([i_up[1], i_up[0]], dim=0)
    v_pu = v_up
    i_pu, v_pu = coalesce_idx_val(i_pu, v_pu, num_prods, num_users)
    i_uu, v_uu = sparse_mm(i_up, v_up, i_pu, v_pu, num_users, num_prods, num_users)
    mask = i_uu[0] != i_uu[1]
    return i_uu[:, mask], v_uu[mask]

def build_uu_power_gpu(idx_uu: torch.Tensor, val_uu: torch.Tensor, U: int, power: int,
                      prune_k: Optional[int] = None):
    assert power % 2 == 0 and power >= 2
    if power == 2:
        return idx_uu, val_uu
    steps = power // 2 - 1
    idxC, valC = idx_uu, val_uu
    for _ in range(steps):
        idxC, valC = sparse_mm(idxC, valC, idx_uu, val_uu, U, U, U)
        mask = idxC[0] != idxC[1]
        idxC, valC = idxC[:, mask], valC[mask]
        idxC, valC = prune_topk_global(idxC, valC, prune_k)
    return idxC, valC

# ============================================================
# Metrics
# ============================================================
def safe_roc_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return 0.5

def safe_pr_auc(y_true, y_score):
    try:
        return average_precision_score(y_true, y_score)
    except Exception:
        return 0.5

def _scores_from_full(prob_full, labels, idx):
    y_true = labels[idx]
    y_score = prob_full[idx]
    y_pred = (y_score >= 0.5).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1], zero_division=0)
    roc = safe_roc_auc(y_true, y_score)
    pr = safe_pr_auc(y_true, y_score)
    return {'precision': p[0], 'recall': r[0], 'f1': f[0], 'roc_auc': roc, 'pr_auc': pr}

# ============================================================
# Pretty results
# ============================================================
_SUB = str.maketrans({
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉", ".": "․", "-": "₋", "+": "₊",
})
USE_UNICODE_SUBSCRIPT = (os.name != "nt")

def fmt_mean_std(mean_val, std_val):
    m = f"{mean_val:.4f}"
    s = f"{std_val:.4f}"
    if USE_UNICODE_SUBSCRIPT and _HAS_WCWIDTH:
        s_sub = s.translate(_SUB)
        return f"{m}₍{s_sub}₎"
    return f"{m}({s})"

def console_reports(all_results_by_mode_seed, mode_labels, models=MODEL_ORDER, excel_path="overall_results.xlsx"):
    records = []
    for ml in mode_labels:
        for seed, res in all_results_by_mode_seed[ml].items():
            for m in models:
                if m in res:
                    r = res[m]
                    records.append({
                        'Seed': seed, 'Mode': ml, 'Model': m,
                        'precision': r['precision'],
                        'recall': r['recall'],
                        'f1': r['f1'],
                        'pr_auc': r['pr_auc'],
                        'roc_auc': r['roc_auc'],
                        'runtime_sec': r.get('runtime_sec', np.nan),
                        'best_epoch': r.get('best_epoch', np.nan),
                    })
    df = pd.DataFrame(records)
    if df.empty:
        print("No results to report.")
        return

    metrics = [
        ('precision', 'P+'),
        ('recall', 'R+'),
        ('f1', 'F+'),
        ('pr_auc', 'PR-AUC+'),
        ('roc_auc', 'ROC-AUC+'),
        ('runtime_sec', 'Time(s)'),
        ('best_epoch', 'BestEp'),
    ]
    agg = df.groupby(['Mode', 'Model']).agg({k: ['mean', 'std'] for k, _ in metrics})
    base_mode = 'MODE-A'

    base_mean = {}
    for m in models:
        if (base_mode, m) not in agg.index:
            continue
        for met_key, _ in metrics:
            base_mean[(m, met_key)] = float(agg.loc[(base_mode, m)][(met_key, 'mean')])

    def fmt_delta(d: float) -> str:
        if not np.isfinite(d):
            d = 0.0
        return f"{d:+.4f}"

    def fmt_cell(mean_val: float, std_val: float, delta_val: float, mode: str) -> str:
        main = fmt_mean_std(mean_val, std_val)
        if mode == base_mode:
            return main
        s = fmt_delta(delta_val)
        if s in ("+0.0000", "-0.0000"):
            return main
        row_color = MODE_ANSI.get(mode, "")
        if delta_val > 0:
            s = f"{ANSI_RED}{s}{ANSI_RESET}{row_color}"
        elif delta_val < 0:
            s = f"{ANSI_GREEN}{s}{ANSI_RESET}{row_color}"
        return f"{main} {s}"

    headers = ['Mode', 'Model'] + [name for _, name in metrics]
    rows_out = []
    for m in models:
        for ml in mode_labels:
            if (ml, m) not in agg.index:
                continue
            row_cells = [ml, m]
            for met_key, _disp in metrics:
                mean_val = float(agg.loc[(ml, m)][(met_key, 'mean')])
                std_val = float(agg.loc[(ml, m)][(met_key, 'std')])
                if not np.isfinite(std_val):
                    std_val = 0.0
                b = base_mean.get((m, met_key), mean_val)
                delta_val = float(mean_val - b)
                row_cells.append(fmt_cell(mean_val, std_val, delta_val, ml))
            rows_out.append(row_cells)

    ncol = len(headers)
    col_widths = []
    for i in range(ncol):
        maxw = vis_width(headers[i])
        for r in rows_out:
            maxw = max(maxw, vis_width(r[i]))
        col_widths.append(maxw)

    print("\n===== Overall Results (mean₍std₎; MODE-B/C show Δ vs MODE-A) =====")
    head_line = '|' + '|'.join(f" {pad_left(headers[i], col_widths[i])} " for i in range(ncol)) + '|'
    sep_line  = '|' + '|'.join('-' * (col_widths[i] + 2) for i in range(ncol)) + '|'
    print(head_line)
    print(sep_line)

    for r in rows_out:
        mode = r[0]
        color = MODE_ANSI.get(mode, "")
        line = '|' + '|'.join(f" {pad_left(r[i], col_widths[i])} " for i in range(ncol)) + '|'
        if color:
            print(color + line + ANSI_RESET)
        else:
            print(line)

    df_display = pd.DataFrame(rows_out, columns=headers)
    for c in df_display.columns:
        df_display[c] = df_display[c].astype(str).apply(strip_ansi)

    num_records = []
    for m in models:
        for ml in mode_labels:
            if (ml, m) not in agg.index:
                continue
            rec = {'Mode': ml, 'Model': m}
            for met_key, disp in metrics:
                mean_val = float(agg.loc[(ml, m)][(met_key, 'mean')])
                std_val  = float(agg.loc[(ml, m)][(met_key, 'std')])
                if not np.isfinite(std_val):
                    std_val = 0.0
                b = base_mean.get((m, met_key), mean_val)
                delta_val = float(mean_val - b)
                rec[f'{disp}_mean'] = mean_val
                rec[f'{disp}_std'] = std_val
                rec[f'{disp}_delta_vs_MODE-A'] = delta_val
            num_records.append(rec)

    df_numeric = pd.DataFrame(num_records)
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_display.to_excel(writer, index=False, sheet_name='display')
        df_numeric.to_excel(writer, index=False, sheet_name='numeric')

    logger.info(f"Excel saved: {excel_path}")

# ============================================================
# Standard FFN
# ============================================================
class FFN(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * hidden_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * hidden_mult, dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

# ============================================================
# 1) HGT (dtype-safe index_add)
# ============================================================
class WeightedHGTConv(MessagePassing):
    def __init__(self, in_dims, out_dim, metadata, heads=4, dropout=0.0):
        super().__init__(aggr='add', node_dim=0)
        node_types, edge_types = metadata
        self.node_types = node_types
        self.edge_types = edge_types
        self.heads = heads
        self.out_dim = out_dim
        self.dk = math.ceil(out_dim / heads)
        self.proj_dim = self.dk * heads
        self.k_lin = nn.ModuleDict({nt: nn.Linear(in_dims[nt], self.proj_dim) for nt in node_types})
        self.q_lin = nn.ModuleDict({nt: nn.Linear(in_dims[nt], self.proj_dim) for nt in node_types})
        self.v_lin = nn.ModuleDict({nt: nn.Linear(in_dims[nt], self.proj_dim) for nt in node_types})
        self.rel_pri = nn.ParameterDict()
        self.rel_k = nn.ParameterDict()
        self.rel_v = nn.ParameterDict()

        for (src, rel, dst) in edge_types:
            key = f"{src}__{rel}__{dst}"
            self.rel_pri[key] = nn.Parameter(torch.ones(heads))
            self.rel_k[key] = nn.Parameter(torch.Tensor(heads, self.dk, self.dk))
            self.rel_v[key] = nn.Parameter(torch.Tensor(heads, self.dk, self.dk))

        self.out_lin = nn.ModuleDict({nt: nn.Linear(self.proj_dim, out_dim) for nt in node_types})
        self.skip = nn.ParameterDict({nt: nn.Parameter(torch.ones(1)) for nt in node_types})
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for m in list(self.k_lin.values()) + list(self.q_lin.values()) + list(self.v_lin.values()) + list(self.out_lin.values()):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        for k in self.rel_k.values():
            nn.init.xavier_uniform_(k.view(self.heads, -1))
        for v in self.rel_v.values():
            nn.init.xavier_uniform_(v.view(self.heads, -1))
        for p in self.rel_pri.values():
            nn.init.ones_(p)
        for p in self.skip.values():
            nn.init.ones_(p)

    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None):
        K = {nt: self.k_lin[nt](x).view(-1, self.heads, self.dk) for nt, x in x_dict.items()}
        Q = {nt: self.q_lin[nt](x).view(-1, self.heads, self.dk) for nt, x in x_dict.items()}
        V = {nt: self.v_lin[nt](x).view(-1, self.heads, self.dk) for nt, x in x_dict.items()}
        out = {nt: x.new_zeros(x.size(0), self.heads, self.dk) for nt, x in x_dict.items()}

        for (src, rel, dst), eidx in edge_index_dict.items():
            row, col = eidx
            key = f"{src}__{rel}__{dst}"
            k = K[src][row]
            q = Q[dst][col]
            v = V[src][row]
            rk = self.rel_k[key]
            rv = self.rel_v[key]
            k = torch.einsum("ehd,hdf->ehf", k, rk)
            v = torch.einsum("ehd,hdf->ehf", v, rv)
            att = (q * k).sum(-1) / math.sqrt(self.dk)
            if edge_weight_dict is not None:
                ew = edge_weight_dict[(src, rel, dst)].view(-1, 1).clamp(min=1e-12)
                att = att + torch.log(ew).to(att.dtype)
            att = F.leaky_relu(att, 0.2)
            num_dst = x_dict[dst].size(0)
            alpha = torch.stack(
                [softmax(att[:, h], index=col, num_nodes=num_dst) for h in range(self.heads)],
                dim=1
            )
            alpha = self.dropout(alpha).to(v.dtype)
            msg = v * alpha.unsqueeze(-1)
            msg = msg * self.rel_pri[key].to(msg.dtype).view(1, -1, 1)
            msg = self.dropout(msg)
            out[dst] = index_add_dtype(out[dst], 0, col, msg)

        out2 = {}
        for nt in out:
            h = self.out_lin[nt](out[nt].reshape(out[nt].size(0), -1))
            out2[nt] = torch.sigmoid(self.skip[nt]) * h + (1 - torch.sigmoid(self.skip[nt])) * x_dict[nt]
        return out2

class HGTBlock(nn.Module):
    def __init__(self, metadata, in_dims, dim, heads=4, dropout=0.0):
        super().__init__()
        self.conv = WeightedHGTConv(in_dims, dim, metadata, heads=heads, dropout=dropout)
        self.norm1 = nn.ModuleDict({nt: nn.LayerNorm(dim) for nt in metadata[0]})
        self.norm2 = nn.ModuleDict({nt: nn.LayerNorm(dim) for nt in metadata[0]})
        self.ffn   = nn.ModuleDict({nt: FFN(dim, hidden_mult=4, dropout=dropout) for nt in metadata[0]})
        self.drop  = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict, edge_weight_dict):
        h = self.conv(x_dict, edge_index_dict, edge_weight_dict)
        out = {}
        for nt, x in x_dict.items():
            y = self.drop(F.gelu(h[nt]))
            x1 = self.norm1[nt](x + y)
            y2 = self.ffn[nt](x1)
            out[nt] = self.norm2[nt](x1 + y2)
        return out

class HGTNet(nn.Module):
    def __init__(self, metadata, in_dims, hid, out_dim, heads=4, layers=2, dropout=0.0):
        super().__init__()
        node_types, _ = metadata
        self.in_proj = nn.ModuleDict({nt: nn.Linear(in_dims[nt], hid) for nt in node_types})
        self.blocks = nn.ModuleList([
            HGTBlock(metadata, {nt: hid for nt in node_types}, hid, heads=heads, dropout=dropout)
            for _ in range(layers)
        ])
        self.cls = nn.Linear(hid, out_dim)

    def forward(self, x_dict, edge_index_dict, edge_weight_dict, cache):
        h = {nt: F.gelu(self.in_proj[nt](x)) for nt, x in x_dict.items()}
        for blk in self.blocks:
            h = blk(h, edge_index_dict, edge_weight_dict)
        return self.cls(h['user'])

# ============================================================
# 2) RGCN (dtype-safe index_add)
# ============================================================
class WeightedRGCNConvBasis(nn.Module):
    def __init__(self, in_dim, out_dim, num_relations, num_bases=4, bias=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_rel = num_relations
        self.num_bases = min(num_bases, num_relations)
        self.bases = nn.Parameter(torch.Tensor(self.num_bases, in_dim, out_dim))
        self.comp = nn.Parameter(torch.Tensor(num_relations, self.num_bases))
        self.self_loop = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.bases.view(self.num_bases, -1))
        nn.init.xavier_uniform_(self.comp)
        nn.init.xavier_uniform_(self.self_loop)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_type, edge_weight=None):
        row, col = edge_index
        coef = self.comp[edge_type]
        Wr = torch.einsum("eb, bio -> eio", coef, self.bases)
        msg = torch.bmm(x[row].unsqueeze(1), Wr).squeeze(1)

        if edge_weight is None:
            ew = torch.ones(row.size(0), device=x.device, dtype=x.dtype)
        else:
            ew = edge_weight.to(x.dtype).clamp(min=1e-12)

        deg = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        index_add__dtype(deg, 0, col, ew)
        deg = deg.clamp(min=1e-12)
        norm = (ew / deg[col]).view(-1, 1)
        msg = msg * norm
        out = x.new_zeros(x.size(0), self.out_dim)
        out = index_add_dtype(out, 0, col, msg)
        out = out + x @ self.self_loop
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out

class RGCN_Model(nn.Module):
    def __init__(self, in_dim, hid, out_dim, num_rel=2, dropout=0.0, num_bases=4):
        super().__init__()
        self.type_emb = nn.Embedding(2, 8)
        self.lin_in = nn.Linear(in_dim + 8, hid)
        self.conv1 = WeightedRGCNConvBasis(hid, hid, num_relations=num_rel, num_bases=num_bases)
        self.conv2 = WeightedRGCNConvBasis(hid, hid, num_relations=num_rel, num_bases=num_bases)
        self.norm1 = nn.LayerNorm(hid)
        self.norm2 = nn.LayerNorm(hid)
        self.ffn = FFN(hid, hidden_mult=4, dropout=dropout)
        self.norm3 = nn.LayerNorm(hid)
        self.cls = nn.Linear(hid, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, ei_dict, ew_dict, cache):
        homo = cache['homo']
        x = homo['x']
        node_type = homo['node_type']
        edge_index = homo['edge_index']
        edge_type = homo['edge_type']
        edge_weight = homo['edge_weight']
        user_idx = homo['user_idx']

        h = torch.cat([x, self.type_emb(node_type)], dim=1)
        h = self.dropout(F.gelu(self.lin_in(h)))
        y = self.dropout(F.gelu(self.conv1(h, edge_index, edge_type, edge_weight=edge_weight)))
        h = self.norm1(h + y)
        y = self.dropout(F.gelu(self.conv2(h, edge_index, edge_type, edge_weight=edge_weight)))
        h = self.norm2(h + y)
        h = self.norm3(h + self.ffn(h))
        out = self.cls(h)
        return out[user_idx]

# ============================================================
# 3) GTN (torch_sparse spspmm)
# ============================================================
class GTLayer(nn.Module):
    """A_c = sum_r softmax(alpha_c)[r] * A_r"""
    def __init__(self, num_rel: int, num_channels: int):
        super().__init__()
        self.num_rel = num_rel
        self.C = num_channels
        self.alpha = nn.Parameter(torch.zeros(num_channels, num_rel))

    def forward(self, rel_adjs: List[Tuple[torch.Tensor, torch.Tensor]], N: int):
        A = []
        for c in range(self.C):
            a = F.softmax(self.alpha[c], dim=0)
            all_i, all_v = [], []
            for r, (idx_r, val_r) in enumerate(rel_adjs):
                if idx_r.numel() == 0:
                    continue
                all_i.append(idx_r)
                all_v.append(val_r * a[r])
            if len(all_i) == 0:
                idx = torch.empty((2, 0), dtype=torch.long, device=self.alpha.device)
                val = torch.empty((0,), dtype=torch.float32, device=self.alpha.device)
            else:
                idx = torch.cat(all_i, dim=1)
                val = torch.cat(all_v, dim=0)
                idx, val = coalesce_idx_val(idx, val, N, N)
            A.append((idx, val))
        return A

class GTN_Model(nn.Module):
    def __init__(
        self,
        in_dim_homo: int,
        hid: int,
        out_dim: int,
        num_rel: int = 2,
        num_channels: int = 3,
        num_layers: int = 2,
        dropout: float = 0.0,
        prune_max_edges: Optional[int] = 2_000_000,
    ):
        super().__init__()
        self.num_rel = num_rel
        self.C = num_channels
        self.L = num_layers
        self.prune_max_edges = prune_max_edges
        self.type_emb = nn.Embedding(2, 8)
        self.lin_in = nn.Linear(in_dim_homo + 8, hid)
        self.gt_layers = nn.ModuleList([GTLayer(num_rel, num_channels) for _ in range(num_layers)])
        self.gcn1 = nn.ModuleList([GCNConv(hid, hid, add_self_loops=True, normalize=True) for _ in range(self.C)])
        self.gcn2 = nn.ModuleList([GCNConv(hid, hid, add_self_loops=True, normalize=True) for _ in range(self.C)])
        self.fuse = nn.Linear(self.C * hid, hid)
        self.norm = nn.LayerNorm(hid)
        self.ffn = FFN(hid, hidden_mult=4, dropout=dropout)
        self.norm2 = nn.LayerNorm(hid)
        self.cls = nn.Linear(hid, out_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _build_rel_adj(edge_index, edge_type, rel_id, edge_weight, N):
        mask = (edge_type == rel_id)
        idx = edge_index[:, mask]
        if idx.numel() == 0:
            return (torch.empty((2, 0), dtype=torch.long, device=edge_index.device),
                    torch.empty((0,), dtype=torch.float32, device=edge_index.device))
        if edge_weight is None:
            val = torch.ones(idx.size(1), device=idx.device, dtype=torch.float32)
        else:
            val = edge_weight[mask].to(torch.float32).clamp(min=1e-12)
        idx, val = coalesce_idx_val(idx, val, N, N)
        return idx, val

    def forward(self, x_dict, ei_dict, ew_dict, cache):
        homo = cache['homo']
        x = homo['x']
        node_type = homo['node_type']
        edge_index = homo['edge_index']
        edge_type = homo['edge_type']
        edge_weight = homo['edge_weight']
        user_idx = homo['user_idx']
        N = x.size(0)
        h0 = torch.cat([x, self.type_emb(node_type)], dim=1)
        h0 = self.dropout(F.gelu(self.lin_in(h0)))
        rel_adjs = [self._build_rel_adj(edge_index, edge_type, r, edge_weight, N) for r in range(self.num_rel)]
        dev = x.device
        eye = torch.arange(N, device=dev, dtype=torch.long)
        Ai = torch.stack([eye, eye], dim=0)
        Av = torch.ones(N, device=dev, dtype=torch.float32)
        A_channels = [(Ai, Av) for _ in range(self.C)]

        for layer in self.gt_layers:
            mixed = layer(rel_adjs, N)
            new_channels = []
            for c in range(self.C):
                idxA, valA = A_channels[c]
                idxB, valB = mixed[c]
                if idxB.numel() == 0:
                    new_channels.append((idxA, valA))
                    continue
                idxC, valC = sparse_mm(idxA, valA, idxB, valB, N, N, N)
                idxC, valC = prune_topk_global(idxC, valC, self.prune_max_edges)
                new_channels.append((idxC, valC))
            A_channels = new_channels

        hs = []
        for c in range(self.C):
            idxA, valA = A_channels[c]
            idxA, valA = add_self_loops_with_weight(idxA, valA, N, fill=1.0)
            hc = self.dropout(F.gelu(self.gcn1[c](h0, idxA, edge_weight=valA)))
            hc = self.dropout(F.gelu(self.gcn2[c](hc, idxA, edge_weight=valA)))
            hs.append(hc)

        Hcat = torch.cat(hs, dim=1)
        h = self.dropout(F.gelu(self.fuse(Hcat)))
        h = self.norm(h0 + h)
        h = self.norm2(h + self.ffn(h))
        out = self.cls(h)
        return out[user_idx]

# ============================================================
# 4) HAN
# ============================================================
class SemanticAttention(nn.Module):
    def __init__(self, in_dim, hidden: Optional[int] = None):
        super().__init__()
        h = hidden or in_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.Tanh(),
            nn.Linear(h, 1, bias=False)
        )
    def forward(self, H):
        w = self.proj(H).softmax(dim=1)
        return (w * H).sum(dim=1)

class HAN_Model(nn.Module):
    def __init__(self, in_dim_user, hid, out_dim, heads=4, dropout=0.0, num_metapaths=2):
        super().__init__()
        self.num_metapaths = num_metapaths
        self.proj = nn.Linear(in_dim_user, hid)
        self.gats = nn.ModuleList([
            GATConv(hid, hid // heads, heads=heads, dropout=dropout, edge_dim=1, add_self_loops=False)
            for _ in range(num_metapaths)
        ])
        self.sem_attn = SemanticAttention(hid)
        self.norm = nn.LayerNorm(hid)
        self.ffn = FFN(hid, hidden_mult=4, dropout=dropout)
        self.norm2 = nn.LayerNorm(hid)
        self.cls = nn.Linear(hid, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, ei_dict, ew_dict, cache):
        meta_list = cache['han_meta_w'] if ew_dict is not None else cache['han_meta_u']
        x_u = x_dict['user']
        h0 = self.dropout(F.gelu(self.proj(x_u)))

        outs = []
        for k, (idx, val) in enumerate(meta_list[:self.num_metapaths]):
            idx2, val2 = add_self_loops_with_weight(idx, val, h0.size(0), fill=1.0)
            edge_attr = torch.log(val2.clamp(min=1e-12)).view(-1, 1).to(h0.dtype)
            hk = self.gats[k](h0, idx2, edge_attr=edge_attr)
            outs.append(self.dropout(F.gelu(hk)))
        H = torch.stack(outs, dim=1)
        h = self.sem_attn(H)
        h = self.norm(h0 + h)
        h = self.norm2(h + self.ffn(h))
        return self.cls(h)

# ============================================================
# 5) SimpleHGN (dtype-safe index_add)
# ============================================================
class SimpleHGNConv(MessagePassing):
    def __init__(self, in_dim, out_dim, num_rel=2, heads=4, dropout=0.0):
        super().__init__(aggr='add', node_dim=0)
        self.heads = heads
        assert out_dim % heads == 0
        self.dh = out_dim // heads
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.rel_emb = nn.Embedding(num_rel, out_dim)
        self.att = nn.Parameter(torch.Tensor(heads, 3 * self.dh))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, x, edge_index, edge_type, edge_weight=None):
        h = self.lin(x).view(-1, self.heads, self.dh)
        row, col = edge_index
        r = self.rel_emb(edge_type).view(-1, self.heads, self.dh)
        cat = torch.cat([h[row], h[col], r], dim=-1)
        e = (cat * self.att.to(cat.dtype)).sum(-1)
        if edge_weight is not None:
            e = e + torch.log(edge_weight.clamp(min=1e-12)).to(e.dtype).view(-1, 1)
        e = self.leaky_relu(e)
        alpha = torch.stack(
            [softmax(e[:, k], index=col, num_nodes=h.size(0)) for k in range(self.heads)],
            dim=1
        )
        alpha = self.dropout(alpha).to(h.dtype)

        msg = (h[row] + r) * alpha.unsqueeze(-1)
        out = x.new_zeros(h.size(0), self.heads, self.dh)
        out = index_add_dtype(out, 0, col, msg)  # dtype-safe
        out = out.reshape(h.size(0), -1)
        return out

class SimpleHGN_Model(nn.Module):
    def __init__(self, in_dim, hid, out_dim, num_rel=2, heads=4, dropout=0.0):
        super().__init__()
        self.type_emb = nn.Embedding(2, 8)
        self.lin_in = nn.Linear(in_dim + 8, hid)
        self.conv1 = SimpleHGNConv(hid, hid, num_rel=num_rel, heads=heads, dropout=dropout)
        self.conv2 = SimpleHGNConv(hid, hid, num_rel=num_rel, heads=heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hid)
        self.norm2 = nn.LayerNorm(hid)
        self.ffn = FFN(hid, hidden_mult=4, dropout=dropout)
        self.norm3 = nn.LayerNorm(hid)
        self.cls = nn.Linear(hid, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, ei_dict, ew_dict, cache):
        homo = cache['homo']
        x = homo['x']
        node_type = homo['node_type']
        edge_index = homo['edge_index']
        edge_type = homo['edge_type']
        edge_weight = homo['edge_weight']
        user_idx = homo['user_idx']
        h = torch.cat([x, self.type_emb(node_type)], dim=1)
        h = self.dropout(F.gelu(self.lin_in(h)))
        y = self.dropout(F.gelu(self.conv1(h, edge_index, edge_type, edge_weight=edge_weight)))
        h = self.norm1(h + y)
        y = self.dropout(F.gelu(self.conv2(h, edge_index, edge_type, edge_weight=edge_weight)))
        h = self.norm2(h + y)
        h = self.norm3(h + self.ffn(h))
        out = self.cls(h)
        return out[user_idx]

# ============================================================
# 6) HetSANN (dtype-safe index_add)
# ============================================================
class HetSANNConv(MessagePassing):
    def __init__(self, in_dim, out_dim, num_rel=2, heads=4, dropout=0.0):
        super().__init__(aggr='add', node_dim=0)
        self.heads = heads
        assert out_dim % heads == 0
        self.dh = out_dim // heads
        self.Wq = nn.Linear(in_dim, out_dim, bias=False)
        self.Wk = nn.Linear(in_dim, out_dim, bias=False)
        self.Wv = nn.Linear(in_dim, out_dim, bias=False)
        self.rel_k = nn.Embedding(num_rel, out_dim)
        self.rel_v = nn.Embedding(num_rel, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type, edge_weight=None):
        Q = self.Wq(x).view(-1, self.heads, self.dh)
        K = self.Wk(x).view(-1, self.heads, self.dh)
        V = self.Wv(x).view(-1, self.heads, self.dh)
        rk = self.rel_k(edge_type).view(-1, self.heads, self.dh)
        rv = self.rel_v(edge_type).view(-1, self.heads, self.dh)
        row, col = edge_index
        q = Q[col]
        k = K[row] + rk
        v = V[row] + rv
        att = (q * k).sum(-1) / math.sqrt(self.dh)
        if edge_weight is not None:
            att = att + torch.log(edge_weight.clamp(min=1e-12)).to(att.dtype).view(-1, 1)
        att = F.leaky_relu(att, 0.2)
        alpha = torch.stack(
            [softmax(att[:, h], index=col, num_nodes=x.size(0)) for h in range(self.heads)],
            dim=1
        )
        alpha = self.dropout(alpha).to(v.dtype)
        msg = v * alpha.unsqueeze(-1)
        out = x.new_zeros(x.size(0), self.heads, self.dh)
        out = index_add_dtype(out, 0, col, msg)  # dtype-safe
        out = out.reshape(x.size(0), -1)
        return out

class HetSANN_Model(nn.Module):
    def __init__(self, in_dim, hid, out_dim, num_rel=2, heads=4, dropout=0.0):
        super().__init__()
        self.type_emb = nn.Embedding(2, 8)
        self.lin_in = nn.Linear(in_dim + 8, hid)
        self.conv1 = HetSANNConv(hid, hid, num_rel=num_rel, heads=heads, dropout=dropout)
        self.conv2 = HetSANNConv(hid, hid, num_rel=num_rel, heads=heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hid)
        self.norm2 = nn.LayerNorm(hid)
        self.ffn = FFN(hid, hidden_mult=4, dropout=dropout)
        self.norm3 = nn.LayerNorm(hid)
        self.cls = nn.Linear(hid, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, ei_dict, ew_dict, cache):
        homo = cache['homo']
        x = homo['x']
        node_type = homo['node_type']
        edge_index = homo['edge_index']
        edge_type = homo['edge_type']
        edge_weight = homo['edge_weight']
        user_idx = homo['user_idx']
        h = torch.cat([x, self.type_emb(node_type)], dim=1)
        h = self.dropout(F.gelu(self.lin_in(h)))
        y = self.dropout(F.gelu(self.conv1(h, edge_index, edge_type, edge_weight=edge_weight)))
        h = self.norm1(h + y)
        y = self.dropout(F.gelu(self.conv2(h, edge_index, edge_type, edge_weight=edge_weight)))
        h = self.norm2(h + y)
        h = self.norm3(h + self.ffn(h))
        out = self.cls(h)
        return out[user_idx]

# ============================================================
# 7) RSHN (dtype-safe index_add)
# ============================================================
class RSHN_Model(nn.Module):
    def __init__(self, in_dim, hid, out_dim, num_rel=2, dropout=0.0):
        super().__init__()
        self.type_emb = nn.Embedding(2, 8)
        self.lin_in = nn.Linear(in_dim + 8, hid)
        self.rel_emb = nn.Embedding(num_rel, hid)
        self.W_rel = nn.Parameter(torch.Tensor(num_rel, hid, hid))
        self.gate = nn.Sequential(
            nn.Linear(3 * hid, hid),
            nn.ReLU(),
            nn.Linear(hid, 1)
        )
        self.norm1 = nn.LayerNorm(hid)
        self.ffn = FFN(hid, hidden_mult=4, dropout=dropout)
        self.norm3 = nn.LayerNorm(hid)
        self.cls = nn.Linear(hid, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_rel.view(self.W_rel.size(0), -1))
        nn.init.xavier_uniform_(self.lin_in.weight)
        nn.init.zeros_(self.lin_in.bias)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, x_dict, ei_dict, ew_dict, cache):
        homo = cache['homo']
        x = homo['x']
        node_type = homo['node_type']
        edge_index = homo['edge_index']
        edge_type = homo['edge_type']
        edge_weight = homo['edge_weight']
        user_idx = homo['user_idx']
        h = torch.cat([x, self.type_emb(node_type)], dim=1)
        h = self.dropout(F.gelu(self.lin_in(h)))
        row, col = edge_index
        Wr = self.W_rel[edge_type]
        hr = torch.bmm(h[row].unsqueeze(1), Wr).squeeze(1)
        rr = self.rel_emb(edge_type)
        gate_inp = torch.cat([h[row], h[col], rr], dim=1)
        g = torch.sigmoid(self.gate(gate_inp)).view(-1, 1)
        msg = (hr + rr) * g

        if edge_weight is None:
            ew = torch.ones(row.size(0), device=h.device, dtype=h.dtype)
        else:
            ew = edge_weight.to(h.dtype).clamp(min=1e-12)
        msg = msg * ew.view(-1, 1)
        deg = torch.zeros(h.size(0), device=h.device, dtype=h.dtype)
        index_add__dtype(deg, 0, col, ew)
        deg = deg.clamp(min=1e-12)
        norm = (ew / deg[col]).view(-1, 1)
        msg = msg * norm
        agg = h.new_zeros(h.size(0), h.size(1))
        agg = index_add_dtype(agg, 0, col, msg)  # dtype-safe
        h1 = self.norm1(h + self.dropout(F.gelu(agg)))
        h2 = self.norm3(h1 + self.ffn(h1))
        out = self.cls(h2)
        return out[user_idx]

# ============================================================
# 8) HetGNN (SparseTensor.sample_adj) with CPU fallback and move back to GPU
# ============================================================
def _csr_to_dense_neighbors(rowptr: torch.Tensor,
                            col: torch.Tensor,
                            val: Optional[torch.Tensor],
                            num_rows: int,
                            K: int,
                            n_id: torch.Tensor):
    device = rowptr.device
    counts = (rowptr[1:] - rowptr[:-1]).clamp(min=0)
    E = int(col.numel())

    if E == 0:
        neigh = torch.full((num_rows, K), -1, device=device, dtype=torch.long)
        w = torch.zeros((num_rows, K), device=device, dtype=torch.float32)
        return neigh, w

    rows = torch.repeat_interleave(torch.arange(num_rows, device=device), counts)
    pos = torch.arange(E, device=device) - rowptr[rows]
    neigh = torch.full((num_rows, K), -1, device=device, dtype=torch.long)
    w = torch.zeros((num_rows, K), device=device, dtype=torch.float32)
    neigh_global = n_id[col]

    if val is None:
        val = torch.ones(E, device=device, dtype=torch.float32)
    else:
        val = val.to(torch.float32)
    neigh[rows, pos] = neigh_global
    w[rows, pos] = val
    return neigh, w

def _sample_dense_from_sparse_tensor(adj: SparseTensor,
                                     seed_nodes: torch.Tensor,
                                     K: int,
                                     replace: bool = False):

    global TS_SAMPLE_CUDA_OK
    out_dev = seed_nodes.device
    adj_dev = _sparse_tensor_device(adj)

    def _do_sample(_adj: SparseTensor, _seed: torch.Tensor):
        sampled_adj, n_id = _adj.sample_adj(_seed, K, replace=replace)
        rowptr, col, val = sampled_adj.csr()
        return rowptr, col, val, n_id

    seed_for_sample = seed_nodes
    if adj_dev.type == "cpu" and seed_for_sample.device.type == "cuda":
        seed_for_sample = seed_for_sample.detach().cpu()

    if adj_dev.type == "cuda" and (not TS_SAMPLE_CUDA_OK):
        adj = adj.cpu()
        adj_dev = torch.device("cpu")
        if seed_for_sample.device.type == "cuda":
            seed_for_sample = seed_nodes.detach().cpu()

    try:
        rowptr, col, val, n_id = _do_sample(adj, seed_for_sample)
    except RuntimeError as e:
        msg = str(e)
        if ("No CUDA version supported" in msg) or (adj_dev.type == "cuda"):
            TS_SAMPLE_CUDA_OK = False
            logger.warning(f"sample_adj failed on CUDA -> fallback CPU permanently. Reason: {e}")
            adj_cpu = adj.cpu()
            seed_cpu = seed_nodes.detach().cpu()
            rowptr, col, val, n_id = _do_sample(adj_cpu, seed_cpu)
        else:
            raise

    rowptr = rowptr.to(out_dev, non_blocking=True)
    col    = col.to(out_dev, non_blocking=True)
    n_id   = n_id.to(out_dev, non_blocking=True)
    if val is not None:
        val = val.to(out_dev, non_blocking=True)
    neigh, w = _csr_to_dense_neighbors(rowptr, col, val, seed_nodes.numel(), K, n_id)
    return neigh, w

def _gumbel_topk(scores: torch.Tensor, k: int):
    B, M = scores.shape
    if k >= M:
        return torch.arange(M, device=scores.device).view(1, M).expand(B, M)
    return torch.topk(scores, k=k, dim=1, largest=True, sorted=False).indices

class HetGNN_Model(nn.Module):
    def __init__(self, in_dim_user, in_dim_prod, hid, out_dim, dropout=0.0,
                 K1=20, Kp=10, K2=20, seed_base=42):
        super().__init__()
        self.K1 = int(K1)
        self.Kp = int(Kp)
        self.K2 = int(K2)
        self.seed_base = int(seed_base)
        self._epoch = 0
        self._seed_for_run = 0
        self.u_enc = nn.Linear(in_dim_user, hid)
        self.p_enc = nn.Linear(in_dim_prod, hid)
        self.gru_v1 = nn.GRU(hid, hid // 2, batch_first=True, bidirectional=True)
        self.gru_v2 = nn.GRU(hid, hid // 2, batch_first=True, bidirectional=True)
        self.att_v1 = nn.Linear(hid, 1, bias=False)
        self.att_v2 = nn.Linear(hid, 1, bias=False)
        self.type_attn = nn.Sequential(
            nn.Linear(hid, hid),
            nn.Tanh(),
            nn.Linear(hid, 1, bias=False)
        )
        self.update = nn.GRUCell(hid, hid)
        self.norm = nn.LayerNorm(hid)
        self.ffn = FFN(hid, hidden_mult=4, dropout=dropout)
        self.norm2 = nn.LayerNorm(hid)
        self.cls = nn.Linear(hid, out_dim)
        self.drop = nn.Dropout(dropout)

    def set_epoch(self, epoch: int, seed_for_run: int = 0):
        self._epoch = int(epoch)
        self._seed_for_run = int(seed_for_run)

    def _aggregate_view_sequence(self, seq: torch.Tensor, gru: nn.GRU, att: nn.Linear):
        out, _ = gru(seq)
        score = att(out).squeeze(-1)
        alpha = torch.softmax(score, dim=1)
        return (out * alpha.unsqueeze(-1)).sum(dim=1)

    def forward(self, x_dict, ei_dict, ew_dict, cache):
        device = x_dict['user'].device
        weighted = (ew_dict is not None)
        up_adj: SparseTensor = cache['up_adj']  # U x P (may be on CPU)
        pu_adj: SparseTensor = cache['pu_adj']  # P x U (may be on CPU)
        hu0 = self.drop(F.gelu(self.u_enc(x_dict['user'])))      # [U,H]
        hp0 = self.drop(F.gelu(self.p_enc(x_dict['product'])))   # [P,H]
        U = hu0.size(0)
        maxK = max(self.K1, self.Kp)
        seed_u = torch.arange(U, device=device, dtype=torch.long)
        p_nei_max, w_up_max = _sample_dense_from_sparse_tensor(up_adj, seed_u, maxK, replace=False)

        # view1
        p_nei_1 = p_nei_max[:, :self.K1]
        w1 = w_up_max[:, :self.K1]
        p_safe = p_nei_1.clamp(min=0)
        emb1 = hp0[p_safe]  # [U,K1,H]
        mask1 = (p_nei_1 >= 0).to(emb1.dtype).unsqueeze(-1)
        emb1 = emb1 * mask1
        if weighted:
            emb1 = emb1 * w1.unsqueeze(-1).to(emb1.dtype)

        # view2
        p_mid = p_nei_max[:, :self.Kp]
        w_mid = w_up_max[:, :self.Kp]
        p_mid_flat = p_mid.reshape(-1)              # [U*Kp]
        valid_p = (p_mid_flat >= 0)
        p_seed = p_mid_flat.clamp(min=0)
        u2_flat, w_pu_flat = _sample_dense_from_sparse_tensor(pu_adj, p_seed, 1, replace=False)
        u2_flat = u2_flat.squeeze(1)
        w_pu_flat = w_pu_flat.squeeze(1)
        u2_flat = torch.where(valid_p, u2_flat, torch.full_like(u2_flat, -1))
        w_pu_flat = torch.where(valid_p, w_pu_flat, torch.zeros_like(w_pu_flat))
        u2 = u2_flat.view(U, self.Kp)
        w_pu = w_pu_flat.view(U, self.Kp)
        self_ids = torch.arange(U, device=device).view(U, 1)
        is_self = (u2 == self_ids)
        u2 = torch.where(is_self, torch.full_like(u2, -1), u2)
        w_pu = torch.where(is_self, torch.zeros_like(w_pu), w_pu)
        path_w = (w_mid * w_pu).clamp(min=0.0)
        eps = 1e-12
        g = -torch.log(-torch.log(torch.rand((U, self.Kp), device=device).clamp(min=eps)).clamp(min=eps))
        valid_u2 = (u2 >= 0)
        base = torch.log(path_w + eps) if weighted else torch.zeros_like(path_w)
        score = torch.where(valid_u2, base + g, torch.full_like(base, -1e30))
        sel = _gumbel_topk(score, self.K2)
        u2_sel = u2.gather(1, sel)
        w2_sel = path_w.gather(1, sel)
        u2_safe = u2_sel.clamp(min=0)
        emb2 = hu0[u2_safe]  # [U,K2,H]
        mask2 = (u2_sel >= 0).to(emb2.dtype).unsqueeze(-1)
        emb2 = emb2 * mask2
        if weighted:
            emb2 = emb2 * w2_sel.unsqueeze(-1).to(emb2.dtype)

        v1 = self._aggregate_view_sequence(emb1, self.gru_v1, self.att_v1)
        v2 = self._aggregate_view_sequence(emb2, self.gru_v2, self.att_v2)
        H2 = torch.stack([v1, v2], dim=1)
        beta = torch.softmax(self.type_attn(H2), dim=1)
        agg = (beta * H2).sum(dim=1)
        hu = self.update(agg, hu0)
        hu = self.norm(hu0 + self.drop(F.gelu(hu)))
        hu = self.norm2(hu + self.ffn(hu))
        return self.cls(hu)

# ============================================================
# 9) MAGNN (dtype-safe index_add)
# ============================================================
def build_magnn_upu_instances(
    U: int,
    rows: List[int],
    cols: List[int],
    w: List[float],
    Kp: int = 20,
    Ku: int = 200,
    seed: int = 42,
):
    rng = np.random.RandomState(seed)
    by_p: Dict[int, List[Tuple[int, float]]] = {}
    for u, p, ww in zip(rows, cols, w):
        by_p.setdefault(int(p), []).append((int(u), float(ww)))

    for p in list(by_p.keys()):
        lst = by_p[p]
        if len(lst) > Kp:
            idx = rng.choice(len(lst), size=Kp, replace=False)
            by_p[p] = [lst[i] for i in idx]

    inst_u, inst_p, inst_u2, inst_w = [], [], [], []
    for p, lst in by_p.items():
        if len(lst) <= 1:
            continue
        us = [u for (u, _) in lst]
        ws = [ww for (_, ww) in lst]
        n = len(us)
        for i in range(n):
            u_t = us[i]
            w_t = ws[i]
            for j in range(n):
                if j == i:
                    continue
                u_s = us[j]
                w_s = ws[j]
                inst_u.append(u_t)
                inst_p.append(p)
                inst_u2.append(u_s)
                inst_w.append(w_t * w_s)

    if len(inst_u) == 0:
        return (
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
        )

    inst_u  = np.asarray(inst_u, dtype=np.int64)
    inst_p  = np.asarray(inst_p, dtype=np.int64)
    inst_u2 = np.asarray(inst_u2, dtype=np.int64)
    inst_w  = np.asarray(inst_w, dtype=np.float32)

    keep_mask = np.zeros(len(inst_u), dtype=bool)
    from collections import defaultdict
    grp = defaultdict(list)
    for idx, u_t in enumerate(inst_u):
        grp[int(u_t)].append(idx)

    for u_t, idxs in grp.items():
        if len(idxs) <= Ku:
            keep_mask[idxs] = True
        else:
            sel = rng.choice(idxs, size=Ku, replace=False)
            keep_mask[sel] = True

    inst_u  = inst_u[keep_mask]
    inst_p  = inst_p[keep_mask]
    inst_u2 = inst_u2[keep_mask]
    inst_w  = inst_w[keep_mask]

    return (
        torch.tensor(inst_u, dtype=torch.long),
        torch.tensor(inst_p, dtype=torch.long),
        torch.tensor(inst_u2, dtype=torch.long),
        torch.tensor(inst_w, dtype=torch.float32),
    )

class MAGNN_Model(nn.Module):
    def __init__(self, in_dim_user, in_dim_prod, hid, out_dim, dropout=0.0):
        super().__init__()
        self.u_proj = nn.Linear(in_dim_user, hid)
        self.p_proj = nn.Linear(in_dim_prod, hid)
        self.inst_mlp = nn.Sequential(
            nn.Linear(3 * hid, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, hid),
        )
        self.att = nn.Linear(hid, 1, bias=False)
        self.norm1 = nn.LayerNorm(hid)
        self.ffn = FFN(hid, hidden_mult=4, dropout=dropout)
        self.norm2 = nn.LayerNorm(hid)
        self.cls = nn.Linear(hid, out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_dict, ei_dict, ew_dict, cache):
        hu = self.drop(F.gelu(self.u_proj(x_dict['user'])))
        hp = self.drop(F.gelu(self.p_proj(x_dict['product'])))
        inst_u  = cache['magnn_upu_inst_u']
        inst_p  = cache['magnn_upu_inst_p']
        inst_u2 = cache['magnn_upu_inst_u2']
        z = self.inst_mlp(torch.cat([hu[inst_u], hp[inst_p], hu[inst_u2]], dim=1))
        score = self.att(z).view(-1)

        if ew_dict is not None:
            iw = cache['magnn_upu_inst_w'].clamp(min=1e-12)
            score = score + torch.log(iw).to(score.dtype)

        alpha = softmax(score, index=inst_u, num_nodes=hu.size(0)).to(z.dtype)
        agg = hu.new_zeros(hu.size(0), hu.size(1))
        agg = index_add_dtype(agg, 0, inst_u, z * alpha.view(-1, 1))
        h = self.norm1(hu + self.drop(F.gelu(agg)))
        h = self.norm2(h + self.ffn(h))
        return self.cls(h)

# ============================================================
# HGOT
# ============================================================
class HGOT_Model(nn.Module):
    def __init__(
        self,
        in_dim_user: int,
        hid: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.0,
        num_metapaths: int = 2,
        ot_reg: float = 0.05,
        ot_eps: float = 0.05,
        ot_iters: int = 10,
        ot_cosine: bool = True,
        ot_max_nodes: int = 512,
    ):
        super().__init__()
        self.num_metapaths = num_metapaths
        self.ot_reg = float(ot_reg)
        self.ot_eps = float(ot_eps)
        self.ot_iters = int(ot_iters)
        self.ot_cosine = bool(ot_cosine)
        self.ot_max_nodes = int(ot_max_nodes)
        self.proj = nn.Linear(in_dim_user, hid)
        self.gats = nn.ModuleList([
            GATConv(hid, hid // heads, heads=heads, dropout=dropout, edge_dim=1, add_self_loops=False)
            for _ in range(num_metapaths)
        ])
        self.sem_attn = SemanticAttention(hid)
        self.norm = nn.LayerNorm(hid)
        self.ffn = FFN(hid, hidden_mult=4, dropout=dropout)
        self.norm2 = nn.LayerNorm(hid)
        self.cls = nn.Linear(hid, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.extra_loss = None

    def _sinkhorn(self, C: torch.Tensor):
        N = C.size(0)
        device = C.device
        K = torch.exp(-C / max(self.ot_eps, 1e-9)).clamp(min=1e-12)
        a = torch.full((N,), 1.0 / N, device=device)
        b = torch.full((N,), 1.0 / N, device=device)
        u = torch.ones_like(a)
        v = torch.ones_like(b)

        for _ in range(self.ot_iters):
            u = a / (K @ v).clamp(min=1e-12)
            v = b / (K.t() @ u).clamp(min=1e-12)
        pi = torch.diag(u) @ K @ torch.diag(v)
        return pi

    def _pair_cost(self, Zp: torch.Tensor, Zc: torch.Tensor):
        if self.ot_cosine:
            Zp_n = F.normalize(Zp, dim=1)
            Zc_n = F.normalize(Zc, dim=1)
            sim = Zp_n @ Zc_n.t()
            C = 1.0 - sim
        else:
            C = torch.cdist(Zp, Zc, p=2).pow(2)
        return C

    def forward(self, x_dict, ei_dict, ew_dict, cache):
        device = x_dict['user'].device
        meta_list = cache['han_meta_w'] if ew_dict is not None else cache['han_meta_u']
        meta_list = meta_list[:self.num_metapaths]
        x_u = x_dict['user']
        h0 = self.dropout(F.gelu(self.proj(x_u)))
        branch_outs = []
        for k, (idx, val) in enumerate(meta_list):
            if ew_dict is None:
                val_eff = torch.ones_like(val)
            else:
                val_eff = val.clamp(min=1e-12)

            idx2, val2 = add_self_loops_with_weight(idx, val_eff, h0.size(0), fill=1.0)
            edge_attr = torch.log(val2.clamp(min=1e-12)).view(-1, 1).to(h0.dtype)
            hk = self.gats[k](h0, idx2, edge_attr=edge_attr)
            branch_outs.append(self.dropout(F.gelu(hk)))

        H = torch.stack(branch_outs, dim=1)
        z_c = self.sem_attn(H)
        z_c = self.norm(h0 + z_c)
        z_c = self.norm2(z_c + self.ffn(z_c))
        self.extra_loss = None
        if self.training and self.ot_reg > 0:
            U = z_c.size(0)
            if U > self.ot_max_nodes:
                idxs = torch.randperm(U, device=device)[:self.ot_max_nodes]
            else:
                idxs = torch.arange(U, device=device)
            zc_s = z_c[idxs]
            ot_loss = 0.0

            for zp in branch_outs:
                zp_s = zp[idxs]
                C = self._pair_cost(zp_s, zc_s).detach()
                pi = self._sinkhorn(C)
                z_trans = pi @ zc_s
                ot_loss = ot_loss + F.mse_loss(z_trans, zp_s)

            self.extra_loss = self.ot_reg * ot_loss

        return self.cls(z_c)

# ============================================================
# SE-HTGNN (dtype-safe scatter via index_add_)
# ============================================================
def _scatter_add_1d(src_1d: torch.Tensor, index_1d: torch.Tensor, dim_size: int):
    out = src_1d.new_zeros(dim_size)
    return index_add__dtype(out, 0, index_1d, src_1d)

def _scatter_add_2d(src_2d: torch.Tensor, index_1d: torch.Tensor, dim_size: int):
    out = src_2d.new_zeros(dim_size, src_2d.size(1))
    return index_add__dtype(out, 0, index_1d, src_2d)

def _gcn_norm(edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int):
    src, dst = edge_index
    deg = _scatter_add_1d(edge_weight, dst, dim_size=num_nodes).clamp(min=1e-12)
    norm = edge_weight / (deg[src].sqrt() * deg[dst].sqrt())
    return norm

def _gcn_aggregate(edge_index: torch.Tensor, edge_weight: torch.Tensor, x: torch.Tensor):
    num_nodes = x.size(0)
    src, dst = edge_index
    norm = _gcn_norm(edge_index, edge_weight, num_nodes).unsqueeze(-1).to(x.dtype)
    msg = x[src] * norm
    out = _scatter_add_2d(msg, dst, dim_size=num_nodes)
    return out

class SE_HTGNN_Model(nn.Module):
    def __init__(
        self,
        in_dim_user: int,
        hid: int,
        out_dim: int,
        num_relations: int = 2,
        refine_steps: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_relations = num_relations
        self.refine_steps = refine_steps
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(in_dim_user, hid)
        self.grus = nn.ModuleList([nn.GRUCell(hid, hid) for _ in range(num_relations)])
        self.score = nn.ModuleList([nn.Linear(hid, 1, bias=False) for _ in range(num_relations)])
        self.e0 = nn.Parameter(torch.zeros(num_relations, hid))
        self.ffn = nn.Sequential(
            nn.Linear(hid, 4 * hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hid, hid),
        )
        self.norm1 = nn.LayerNorm(hid)
        self.norm2 = nn.LayerNorm(hid)
        self.cls = nn.Linear(hid, out_dim)
        self.extra_loss = None

    def forward(self, x_dict, ei_dict, ew_dict, cache):
        meta_list = cache['han_meta_w'] if ew_dict is not None else cache['han_meta_u']
        meta_list = meta_list[:self.num_relations]

        x_u = x_dict['user']
        h = self.dropout(F.gelu(self.proj(x_u)))
        U = h.size(0)

        e_states = []
        for r in range(self.num_relations):
            e_init = self.e0[r].to(h.device).to(h.dtype).unsqueeze(0).expand(U, -1).contiguous()
            e_states.append(e_init)

        for _ in range(self.refine_steps):
            rel_outs = []
            rel_scores = []

            for r, (idx, val) in enumerate(meta_list):
                if ew_dict is None:
                    w = torch.ones_like(val)
                else:
                    w = val.clamp(min=1e-12)
                w = w.to(h.dtype)
                h_r = F.elu(_gcn_aggregate(idx, w, h))
                h_r = self.dropout(h_r)
                rel_outs.append(h_r)
                e_states[r] = self.grus[r](h_r, e_states[r])
                e_mean = e_states[r].mean(dim=0, keepdim=True)
                s_r = self.score[r](e_mean).squeeze(0)
                rel_scores.append(s_r)

            scores = torch.stack(rel_scores, dim=0).squeeze(-1)
            alpha = torch.softmax(scores, dim=0)

            h_fused = 0.0
            for r in range(self.num_relations):
                h_fused = h_fused + alpha[r].to(h.dtype) * rel_outs[r]
            h = self.norm1(h + h_fused)
            h = self.norm2(h + self.ffn(h))

        self.extra_loss = None
        return self.cls(h)

# ============================================================
# Training: BF16 autocast + fused AdamW + Early stopping
# ============================================================
def make_fused_adamw(params, lr, weight_decay):
    try:
        return optim.AdamW(params, lr=LR, weight_decay=weight_decay, fused=True)
    except TypeError:
        return optim.AdamW(params, lr=LR, weight_decay=weight_decay)

@torch.no_grad()
def _infer_probs(model, x_dict, ei_dict, ew_dict, cache, device, amp_dtype):
    model.eval()
    with autocast_ctx(device, amp_dtype):
        logits_u = model(x_dict, ei_dict, ew_dict, cache)
        prob = torch.softmax(logits_u.float(), dim=1)[:, 1]
    return prob.detach().cpu().numpy()

def train_model_early_stop(
    model,
    x_dict, ei_dict, ew_dict,
    labels_np: np.ndarray,
    idx_tr: np.ndarray, idx_va: np.ndarray, idx_te: np.ndarray,
    device,
    cache,
    seed_for_run: int = 0,
    amp_dtype=torch.bfloat16,
):
    model = model.to(device)

    opt = make_fused_adamw(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    tr_idx = torch.tensor(idx_tr, dtype=torch.long, device=device)
    va_idx = torch.tensor(idx_va, dtype=torch.long, device=device)
    te_idx = torch.tensor(idx_te, dtype=torch.long, device=device)
    y = torch.tensor(labels_np, dtype=torch.long, device=device)

    best_val = -1e9
    best_epoch = -1
    best_state = None
    bad = 0

    for ep in range(MAX_EPOCHS_GNN):
        model.train()
        if hasattr(model, "set_epoch"):
            try:
                model.set_epoch(ep, seed_for_run=seed_for_run)
            except TypeError:
                model.set_epoch(ep)

        opt.zero_grad(set_to_none=True)

        with autocast_ctx(device, amp_dtype):
            logits_u = model(x_dict, ei_dict, ew_dict, cache)
            loss = loss_fn(logits_u[tr_idx].float(), y[tr_idx])
            if hasattr(model, "extra_loss") and model.extra_loss is not None:
                loss = loss + model.extra_loss.float()

        loss.backward()
        opt.step()

        prob_val = _infer_probs(model, x_dict, ei_dict, ew_dict, cache, device, amp_dtype)
        val_scores = _scores_from_full(prob_val, labels_np, idx_va)
        val_metric = float(val_scores["pr_auc"])

        if val_metric > best_val + MIN_DELTA:
            best_val = val_metric
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    prob_full = _infer_probs(model, x_dict, ei_dict, ew_dict, cache, device, amp_dtype)
    scores = _scores_from_full(prob_full, labels_np, idx_te)
    scores["best_epoch"] = best_epoch
    return scores

# ============================================================
# Registry / runner
# ============================================================
def build_model_registry(cache, in_dims):
    metadata = (['user', 'product'], list(cache['ei_dict'].keys()))
    return {
        'HGT': (lambda: HGTNet(metadata, {'user': in_dims['user'], 'product': in_dims['product']},
                              HIDDEN, 2, heads=HEADS, layers=2, dropout=DROPOUT)),
        'RGCN': (lambda: RGCN_Model(in_dims['homo_in'], HIDDEN, 2, num_rel=2, dropout=DROPOUT, num_bases=4)),
        'GTN':  (lambda: GTN_Model(in_dims['homo_in'], HIDDEN, 2, num_rel=2, num_channels=3, num_layers=2,
                                   dropout=DROPOUT, prune_max_edges=2_000_000)),
        'HAN':  (lambda: HAN_Model(in_dims['user'], HIDDEN, 2, heads=HEADS, dropout=DROPOUT, num_metapaths=2)),
        'MAGNN':(lambda: MAGNN_Model(in_dims['user'], in_dims['product'], HIDDEN, 2, dropout=DROPOUT)),
        'HetGNN':(lambda: HetGNN_Model(in_dims['user'], in_dims['product'], HIDDEN, 2,
                                       dropout=DROPOUT, K1=20, Kp=10, K2=20, seed_base=42)),
        'RSHN': (lambda: RSHN_Model(in_dims['homo_in'], HIDDEN, 2, num_rel=2, dropout=DROPOUT)),
        'SimpleHGN': (lambda: SimpleHGN_Model(in_dims['homo_in'], HIDDEN, 2, num_rel=2, heads=HEADS, dropout=DROPOUT)),
        'HetSANN': (lambda: HetSANN_Model(in_dims['homo_in'], HIDDEN, 2, num_rel=2, heads=HEADS, dropout=DROPOUT)),
        'HGOT': (lambda: HGOT_Model(
            in_dim_user=in_dims['user'],
            hid=HIDDEN,
            out_dim=2,
            heads=HEADS,
            dropout=DROPOUT,
            num_metapaths=2,
            ot_reg=0.05,
            ot_eps=0.05,
            ot_iters=10,
            ot_max_nodes=512,
        )),
        'SE-HTGNN': (lambda: SE_HTGNN_Model(
            in_dim_user=in_dims['user'],
            hid=HIDDEN,
            out_dim=2,
            num_relations=2,
            refine_steps=2,
            dropout=DROPOUT,
        )),
    }

def run_one(seed, mp_weights, feat_weights, device, base_cache, on_step=None, amp_dtype=torch.bfloat16):
    set_seed(seed)

    x_dict = base_cache['x_dict_w'] if feat_weights else base_cache['x_dict_unw']
    ew_dict = base_cache['ew_dict'] if mp_weights else None
    homo = base_cache['homo_w'] if feat_weights else base_cache['homo_unw']
    homo_run = {
        'x': homo['x'],
        'node_type': homo['node_type'],
        'edge_index': homo['edge_index'],
        'edge_type': homo['edge_type'],
        'edge_weight': base_cache['homo_edge_weight'] if mp_weights else None,
        'user_idx': homo['user_idx']
    }

    cache_run = dict(base_cache)
    cache_run['homo'] = homo_run
    labels = base_cache['labels_np']
    tr, va, te = base_cache['splits'][seed]

    in_dims = {
        'user': x_dict['user'].size(1),
        'product': x_dict['product'].size(1),
        'homo_in': homo_run['x'].size(1)
    }

    registry = build_model_registry(cache_run, in_dims)
    results = {}

    for mname in MODEL_ORDER:
        model = registry[mname]()
        t0 = perf_counter()
        scores = train_model_early_stop(
            model, x_dict, base_cache['ei_dict'], ew_dict,
            labels, tr, va, te, device, cache_run, seed_for_run=seed, amp_dtype=amp_dtype
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
        scores['runtime_sec'] = perf_counter() - t0
        results[mname] = scores
        if on_step is not None:
            on_step()
    return results

# ============================================================
# Main
# ============================================================
def main():
    enable_a100_fast_math()
    probe_torch_sparse_cuda()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    logger.info(f"TF32 matmul: {getattr(torch.backends.cuda.matmul, 'allow_tf32', False)}")
    logger.info(f"Seeds: {SEEDS[0]}..{SEEDS[-1]} (total {len(SEEDS)})")

    if device.type == "cuda":
        logger.info(f"torch_sparse CUDA OK (coalesce/spspmm): {TS_CUDA_OK}")
        logger.info(f"torch_sparse CUDA OK (sample_adj): {TS_SAMPLE_CUDA_OK}")
    logger.info(f"CPU feature workers: {CPU_FEATURE_WORKERS}, chunk: {CPU_FEATURE_CHUNK}")

    df_raw = pd.read_excel('data.xlsx')
    df_raw = preprocess_df(df_raw)
    stats_block(df_raw, "Before balancing")
    df_bal = balance_by_day(df_raw, seed=42)
    stats_block(df_bal, "After balancing (before sampling)")
    df_final = df_bal.sample(frac=SAMPLE_FRAC, random_state=42).reset_index(drop=True)
    stats_block(df_final, f"After sampling (ratio={SAMPLE_FRAC:.0%})")

    MODES = [
        (MODE_LABELS[0], False, False),
        (MODE_LABELS[1], True,  False),
        (MODE_LABELS[2], True,  True),
    ]

    # ===== Graph build on CPU + feature compute on CPU (parallel) =====
    B, users, prods, rows, cols, w = build_bipartite_graph(df_final)
    feature_cache = precompute_feature_cache(B, users, prods)

    labels_series = pd.to_numeric(
        df_final.groupby('Reviewer_id')['Label_user'].first(),
        errors='coerce'
    ).fillna(0).astype(int)
    labels_cache = labels_series.reindex(users)

    assert labels_cache.notna().all(), "Some users do not have a corresponding Label_user."
    labels_np = labels_cache.to_numpy()
    U = len(users)
    P = len(prods)

    # ===== Fixed tensor GPU =====
    x_dict_w = {
        'user': feature_cache['weighted']['X_u'].to(device, non_blocking=True),
        'product': feature_cache['weighted']['X_p'].to(device, non_blocking=True)
    }
    x_dict_unw = {
        'user': feature_cache['unweighted']['X_u'].to(device, non_blocking=True),
        'product': feature_cache['unweighted']['X_p'].to(device, non_blocking=True)
    }

    rows_t = torch.tensor(rows, dtype=torch.long, device=device)
    cols_t = torch.tensor(cols, dtype=torch.long, device=device)
    w_t    = torch.tensor(w, dtype=torch.float32, device=device)

    ei_dict = {
        ('user', 'reviews', 'product'): torch.stack([rows_t, cols_t], dim=0),
        ('product', 'rev_by', 'user'):  torch.stack([cols_t, rows_t], dim=0),
    }
    ew_dict = {
        ('user', 'reviews', 'product'): w_t,
        ('product', 'rev_by', 'user'):  w_t,
    }

    # ===== UPU/UU4 GPU + torch_sparse (fallback inside coalesce/spspmm wrappers) =====
    upu_w_i, upu_w_v = build_upu_metapath_edges_gpu(U, P, rows_t, cols_t, w_t)
    ones_w = torch.ones_like(w_t)
    upu_u_i, upu_u_v = build_upu_metapath_edges_gpu(U, P, rows_t, cols_t, ones_w)
    PRUNE_META = 2_000_000
    uu4_w_i, uu4_w_v = build_uu_power_gpu(upu_w_i, upu_w_v, U, power=4, prune_k=PRUNE_META)
    uu4_u_i, uu4_u_v = build_uu_power_gpu(upu_u_i, upu_u_v, U, power=4, prune_k=PRUNE_META)
    han_meta_w = [(upu_w_i, upu_w_v), (uu4_w_i, uu4_w_v)]
    han_meta_u = [(upu_u_i, upu_u_v), (uu4_u_i, uu4_u_v)]

    # ===== Homo graph on GPU =====
    def build_homo_x(xu, xp):
        return torch.cat([xu, xp], dim=0)

    src_up = rows_t
    dst_up = cols_t + U
    src_pu = cols_t + U
    dst_pu = rows_t

    edge_index = torch.stack([torch.cat([src_up, src_pu]), torch.cat([dst_up, dst_pu])], dim=0)
    edge_type = torch.cat([torch.zeros(rows_t.numel(), dtype=torch.long, device=device),
                           torch.ones(cols_t.numel(), dtype=torch.long, device=device)], dim=0)
    edge_weight = torch.cat([w_t, w_t], dim=0)

    node_type = torch.cat([torch.zeros(U, dtype=torch.long, device=device),
                           torch.ones(P, dtype=torch.long, device=device)], dim=0)
    user_idx = torch.arange(U, dtype=torch.long, device=device)

    homo_unw = {
        'x': build_homo_x(x_dict_unw['user'], x_dict_unw['product']),
        'node_type': node_type,
        'edge_index': edge_index,
        'edge_type': edge_type,
        'edge_weight': None,
        'user_idx': user_idx
    }
    homo_w = {
        'x': build_homo_x(x_dict_w['user'], x_dict_w['product']),
        'node_type': node_type,
        'edge_index': edge_index,
        'edge_type': edge_type,
        'edge_weight': None,
        'user_idx': user_idx
    }

    # ===== Early stopping split =====
    all_splits = {}
    idx_all = np.arange(U)
    for seed in SEEDS:
        tr, tmp = train_test_split(
            idx_all, test_size=(1.0 - TRAIN_FRAC), random_state=seed, stratify=labels_np
        )
        va, te = train_test_split(
            tmp, test_size=0.5, random_state=seed + 999, stratify=labels_np[tmp]
        )
        all_splits[seed] = (tr, va, te)

    # ===== MAGNN instances (GPU) =====
    inst_u, inst_p, inst_u2, inst_w = build_magnn_upu_instances(
        U, rows, cols, w, Kp=20, Ku=200, seed=42
    )

    inst_u  = inst_u.to(device, non_blocking=True)
    inst_p  = inst_p.to(device, non_blocking=True)
    inst_u2 = inst_u2.to(device, non_blocking=True)
    inst_w  = inst_w.to(device, non_blocking=True)

    # ===== HetGNN adjacency =====
    # If sample_adj has no CUDA support, keep SparseTensor on CPU.
    if (device.type == "cuda") and (not TS_SAMPLE_CUDA_OK):
        logger.warning("Building SparseTensor on CPU due to torch_sparse sample_adj CUDA failure.")
        up_adj = SparseTensor(row=rows_t.detach().cpu(), col=cols_t.detach().cpu(),
                              value=w_t.detach().cpu(), sparse_sizes=(U, P)).coalesce()
        pu_adj = SparseTensor(row=cols_t.detach().cpu(), col=rows_t.detach().cpu(),
                              value=w_t.detach().cpu(), sparse_sizes=(P, U)).coalesce()
    else:
        up_adj = SparseTensor(row=rows_t, col=cols_t, value=w_t, sparse_sizes=(U, P)).coalesce()
        pu_adj = SparseTensor(row=cols_t, col=rows_t, value=w_t, sparse_sizes=(P, U)).coalesce()

    base_cache = {
        'labels_np': labels_np,
        'x_dict_w': x_dict_w,
        'x_dict_unw': x_dict_unw,
        'ei_dict': ei_dict,
        'ew_dict': ew_dict,
        'splits': all_splits,
        'han_meta_w': han_meta_w,
        'han_meta_u': han_meta_u,
        'homo_unw': homo_unw,
        'homo_w': homo_w,
        'homo_edge_weight': edge_weight,
        'magnn_upu_inst_u': inst_u,
        'magnn_upu_inst_p': inst_p,
        'magnn_upu_inst_u2': inst_u2,
        'magnn_upu_inst_w': inst_w,
        'up_adj': up_adj,
        'pu_adj': pu_adj,
    }

    all_results_by_mode_seed = {ml: {} for (ml, _, _) in MODES}
    total_tasks = len(SEEDS) * len(MODES) * len(MODEL_ORDER)
    amp_dtype = torch.bfloat16

    with tqdm(total=total_tasks, desc="Progress", dynamic_ncols=True, leave=True) as pbar:
        def step():
            pbar.update(1)
        for seed in SEEDS:
            for (mode_label, mp_w, feat_w) in MODES:
                res = run_one(seed, mp_w, feat_w, device, base_cache, on_step=step, amp_dtype=amp_dtype)
                all_results_by_mode_seed[mode_label][seed] = res
    console_reports(all_results_by_mode_seed, mode_labels=MODE_LABELS, models=MODEL_ORDER)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()