"""FlyWire 커넥톰에서 뉴런 묶음(회로)을 잘라내 신경망 층의 '배선'으로 쓰는 Circuit"""
from __future__ import annotations

from pathlib import Path

from . import _check as _C
import numpy as np
import pandas as pd

# 이름 → (주석 열, 값 또는 값 목록). 주석 열: super_class, cell_class, cell_sub_class, cell_type
# 버섯체 기본 회로: 투사 뉴런 → Kenyon 세포 → MBON, APL이 전체를 억제
MUSHROOM_BODY = {
    "PN":   ("cell_class", "ALPN"),
    "KC":   ("cell_class", "Kenyon_Cell"),
    "APL":  ("cell_type", "APL"),
    "MBON": ("cell_class", "MBON"),
}


def _repair(pre: np.ndarray, post: np.ndarray, N: int, rng, max_iter: int = 200) -> np.ndarray:
    """섞은 뒤 생긴 중복 연결(같은 pre→post 두 번)과 자기 연결(pre == post)을 없앰.
    문제 연결의 post를 무작위 다른 연결과 맞바꿈 → 뉴런별 연결 수는 그대로 유지.
    (중복을 그냥 두면 합쳐져서 무작위 회로의 뉴런이 실제보다 적은 입력을 받게 됨)"""
    post = post.copy()
    for _ in range(max_iter):
        bad = pd.Series(pre * N + post).duplicated().values | (pre == post)
        if not bad.any():
            return post
        for i, j in zip(np.nonzero(bad)[0], rng.integers(0, len(post), bad.sum())):
            post[i], post[j] = post[j], post[i]          # 하나씩 (한꺼번에 하면 같은 j가 겹칠 때 값이 사라짐)
    raise RuntimeError(f"중복 연결을 {max_iter}번 안에 없애지 못함 ({bad.sum()}개 남음)")


def _json_value(v):
    """주석 값 하나를 JSON으로: 참거짓·정수·실수는 그대로, 빈 값은 None, 나머지는 문자열"""
    if v is None or (not isinstance(v, (str, bytes)) and pd.isna(v)):
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v)
    return str(v)


def _scipy_sparse():
    """scipy.sparse (from_scipy·to_scipy에서만 - 없으면 설치 안내)"""
    try:
        import scipy.sparse as sps
    except ImportError as e:
        raise ImportError("scipy가 필요한 기능 - pip install scipy") from e
    return sps


def require_disjoint_groups(circuit, who: str):
    """그룹끼리 뉴런이 겹치면 오류 - 층·Neuropil이 그룹마다 뉴런을 한 자리에 두는 계산 (입력 자리, 연결 종류 번호)이
    겹친 뉴런에서 조용히 틀어지므로 (예전: 겹친 입력 뉴런이 두 번 들어가 한쪽 값만 쓰이거나 기울기가 두 번)"""
    if not circuit.groups:
        return
    ids = np.concatenate([np.asarray(v, np.int64).reshape(-1) for v in circuit.groups.values()])
    if len(np.unique(ids)) != len(ids):
        u, n = np.unique(ids, return_counts=True)
        raise ValueError(f"{who}: 회로의 그룹끼리 뉴런이 겹침 ({int((n > 1).sum())}개, 예: {u[n > 1][:5].tolist()}) - "
                         f"그룹마다 다른 뉴런이어야 함: circuit.regroup({{이름: 번호}})로 다시 나눌 것")


class Circuit:
    """뉴런 N개와 부호 있는 시냅스 목록 (pre → post, weight = ±시냅스 수)

    groups: 이름 → 이 회로 안에서의 뉴런 번호 배열 (예: groups["KC"])
    meta:   뉴런별 주석 (cell_type, cell_sub_class 등), 행 순서 = 회로 번호
    pos:    뉴런별 위치 (N, 3) nm (FlyWire 주석의 대표 점), 없으면 None
    """

    def __init__(self, root_ids, groups: dict[str, np.ndarray], pre, post, weight, name="circuit",
                 meta: pd.DataFrame | None = None, pos=None):
        self.root_ids = np.asarray(root_ids, dtype=np.int64)
        self.groups = {k: np.asarray(v, dtype=np.int64) for k, v in groups.items()}
        self.pre = np.asarray(pre, dtype=np.int64)
        self.post = np.asarray(post, dtype=np.int64)
        self.weight = np.asarray(weight, dtype=np.float32)
        self.name = name
        self.meta = meta.reset_index(drop=True) if meta is not None else None
        self.pos = np.asarray(pos, dtype=np.float32) if pos is not None else None

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        from .ganglion.hints import CIRCUIT, missing
        raise missing(type(self).__name__, name, CIRCUIT, dir(type(self)) + list(self.__dict__))

    @property
    def N(self) -> int:
        return len(self.root_ids)

    @property
    def n_edges(self) -> int:
        return len(self.pre)

    def group_of(self) -> np.ndarray:
        """뉴런별 그룹 이름 (그룹이 겹치지 않는다고 가정). 어느 그룹에도 없는 뉴런은 "?" (ConnectomeLayer와 같음 -
        None이면 shuffled 등에서 문자열을 잇다가 실패하고 summary에서 조용히 빠짐)"""
        g = np.full(self.N, "?", dtype=object)
        for k, v in self.groups.items():
            g[v] = k
        return g

    @classmethod
    def from_flywire(cls, groups: dict = MUSHROOM_BODY, side: str | None = "right",
                     data_dir: str | Path | None = None, group_by: str | None = None, annotations: str = "flywire_annotations.tsv",
                     connectivity: str = "Connectivity_783.parquet",
                     completeness: str = "Completeness_783.csv") -> "Circuit":
        """주석으로 고른 뉴런들 사이의 연결만 남긴 회로 (induced subgraph)
        group_by: 주석 열 이름 (예: "cell_type")이면 각 그룹을 그 값마다 다시 나눔 → 그룹 이름 = 값
                  (값이 없는 뉴런은 "<그룹 이름>?"). 무작위 대조군(shuffled)도 이 단위로 섞임
        data_dir: None이면 flydnet.data_dir("flywire") (환경변수 → ~/.flydnet/config.json → ~/.flydnet/data)"""
        from .data import read_connectivity, require
        d = require("flywire", data_dir)
        all_ids = pd.read_csv(d / completeness, index_col=0).index.values.astype(np.int64)
        ann = pd.read_csv(d / annotations, sep="\t", low_memory=False,
                          usecols=["root_id", "super_class", "cell_class", "cell_sub_class", "cell_type", "side",
                                   "pos_x", "pos_y", "pos_z"])
        ann = ann[ann.root_id.isin(all_ids)].drop_duplicates("root_id")    # whole_brain과 같게 (주석 파일 버전에 따라 같은
        if side:                                                          # 뉴런이 두 번이면 meta·pos 행이 뉴런 수와 어긋남)
            sides = sorted(ann.side.dropna().astype(str).unique())
            if side not in sides:                                       # 예전: side="both" 등이 그룹이 모두 비었다는 오류로 보였음
                raise ValueError(f"side는 {sides} 중 하나 또는 None (양쪽): {side!r}")
            ann = ann[ann.side == side]

        picked, gidx = [], {}
        for name, (col, val) in groups.items():
            vals = [val] if isinstance(val, str) else list(val)
            sel = ann[ann[col].isin(vals)]
            parts = sel.groupby(sel[group_by].fillna(f"{name}?"), sort=True) if group_by else [(name, sel)]
            for sub, s in parts:
                if sub in gidx:
                    raise ValueError(f"그룹 이름이 겹침: {sub}")
                gidx[sub] = np.arange(len(picked), len(picked) + len(s))
                picked.extend(s.root_id.values)
        ids = np.array(picked, dtype=np.int64)
        empty = [g for g in groups if not group_by and len(gidx.get(g, [])) == 0]
        if empty:
            import difflib
            hints = []
            for g in empty:
                col, val = groups[g]
                vals = [val] if isinstance(val, str) else list(val)
                pool = ann[col].dropna().astype(str).unique().tolist()
                near = sorted({m for v in vals for m in difflib.get_close_matches(str(v), pool, n=3, cutoff=0.6)})
                hints.append(f"{g} ({col}={vals}{', 비슷한 값: ' + str(near) if near else ''})")
            raise ValueError(f"뉴런이 하나도 없는 그룹: {'; '.join(hints)} (side={side!r}도 확인)")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("그룹끼리 뉴런이 겹침")

        # 전체 뇌 번호 → 회로 번호
        glob = pd.Series(np.arange(len(all_ids)), index=all_ids)[ids].values
        local = np.full(len(all_ids), -1, np.int64); local[glob] = np.arange(len(ids))

        P, Q, W = read_connectivity(d, connectivity)                  # 바꿔 둔 npz (pyarrow 필요 없음) 또는 parquet
        pre, post = local[P], local[Q]
        keep = (pre >= 0) & (post >= 0)
        w = W[keep]
        a = ann.set_index("root_id").loc[ids]
        meta = a[["super_class", "cell_class", "cell_sub_class", "cell_type", "side"]].reset_index()
        return cls(ids, gidx, pre[keep], post[keep], w, name=f"FlyWire {'/'.join(groups)} ({side or 'both'})",
                   meta=meta, pos=a[["pos_x", "pos_y", "pos_z"]].values)

    @classmethod
    def whole_brain(cls, data_dir: str | Path | None = None, group_by: str = "super_class",
                    annotations: str = "flywire_annotations.tsv", connectivity: str = "Connectivity_783.parquet",
                    completeness: str = "Completeness_783.csv") -> "Circuit":
        """전체 뇌 (FlyWire v783 138,639개 뉴런, Shiu et al. 2024 모델과 같은 뉴런·순서). 그룹 = 주석 group_by 값
        (주석이 없는 뉴런은 "unannotated"). fd.genetics.driver로 주석 조건이나 뉴런 ID로 집단을 고름"""
        from .data import read_connectivity, require
        d = require("flywire", data_dir)
        ids = pd.read_csv(d / completeness, index_col=0).index.values.astype(np.int64)
        cols = ["super_class", "cell_class", "cell_sub_class", "cell_type", "side"]
        ann = pd.read_csv(d / annotations, sep="\t", low_memory=False, usecols=["root_id", *cols, "pos_x", "pos_y", "pos_z"])
        a = ann.drop_duplicates("root_id").set_index("root_id").reindex(ids)
        key = a[group_by].astype("string").fillna("unannotated").to_numpy()
        names, inv = np.unique(key, return_inverse=True)
        groups = {str(n): np.nonzero(inv == i)[0] for i, n in enumerate(names)}
        P, Q, W = read_connectivity(d, connectivity)
        w = W.astype(np.float32)
        meta = a[cols].reset_index().rename(columns={"index": "root_id"})
        return cls(ids, groups, P, Q, w,
                   name="FlyWire 전체 뇌", meta=meta, pos=a[["pos_x", "pos_y", "pos_z"]].to_numpy(np.float32))

    # ─────────────── 어떤 그래프든 ───────────────
    @classmethod
    def from_edges(cls, pre, post, weight=None, groups=None, names=None, meta=None, pos=None,
                   name: str = "graph", rest: str = "rest", n: int | None = None) -> "Circuit":
        """연결 목록으로 회로 만들기 (커넥톰이 아니어도 됨)

        pre, post: 주는·받는 노드. 정수(0 ~ N-1) 또는 이름(문자열 등)
        weight:    연결 세기 (부호 = 흥분 +/억제 -). None이면 모두 1. ConnectomeLayer에서는 시냅스 수처럼
                   w_syn(기본 0.275 mV)을 곱해 쓰므로, 세기가 1 근처면 gains·params={"w_syn": ...}로 키울 것
        groups:    {그룹 이름: 노드 번호 또는 이름 목록} 또는 노드마다 그룹 이름 (길이 N). 어느 그룹에도 없는 노드는 rest
        names:     노드 이름 순서 (이름으로 줄 때 번호를 정함). None이면 정수는 0..max, 이름은 정렬 순서
        n:         노드 수 (정수로 줄 때, 연결이 없는 노드까지). None이면 meta 행 수 또는 가장 큰 번호 + 1
        meta:      노드별 주석 표 (pandas, 행 = 노드 순서) → genetics.driver·explain에서 열 이름으로 고름.
                   이름으로 만들면 노드 이름이 meta["node"] (driver(c, node=[...])로 고름)
        pos:       노드 위치 (N, 2 또는 3)"""
        pre, post = np.asarray(pre), np.asarray(post)
        if pre.shape != post.shape or pre.ndim != 1:
            raise ValueError(f"pre·post는 길이가 같은 1차원: {pre.shape}, {post.shape}")
        labeled = names is not None or pre.dtype.kind not in "iu" or post.dtype.kind not in "iu"
        if labeled:
            order = list(names) if names is not None else sorted(set(pre.tolist()) | set(post.tolist()), key=str)
            if len(set(order)) != len(order):
                raise ValueError("names에 같은 이름이 있음")
            index = {k: i for i, k in enumerate(order)}
            try:
                pre_i = np.array([index[k] for k in pre.tolist()], np.int64)
                post_i = np.array([index[k] for k in post.tolist()], np.int64)
            except KeyError as e:
                raise KeyError(f"names에 없는 노드: {e}") from None
            N = len(order)
            if n is not None and n != N:                                # 예전: 조용히 무시 (연결 없는 노드가 빠진 회로)
                raise ValueError(f"이름으로 만들면 노드 수는 이름 수({N}): n={n} - 연결 없는 노드까지 넣으려면 "
                                 "names=[모든 노드 이름]으로")
        else:
            pre_i, post_i = pre.astype(np.int64), post.astype(np.int64)
            N = n if n is not None else len(meta) if meta is not None else                 int(max(pre_i.max(initial=-1), post_i.max(initial=-1))) + 1
            order = None
        w = np.ones(len(pre_i), np.float32) if weight is None else np.asarray(weight, np.float32)
        lookup = (lambda v: index[v]) if labeled else int
        if groups is None:
            gidx = {}
        elif isinstance(groups, dict):
            try:
                gidx = {str(k): np.array([lookup(v) for v in vs], np.int64) for k, vs in groups.items()}
            except KeyError as e:
                raise KeyError(f"그룹에 없는 노드: {e}") from None
        else:
            lab = np.asarray(groups, dtype=object)
            if len(lab) != N:
                raise ValueError(f"노드마다 그룹 이름이면 길이 {N}: {len(lab)}")
            gidx = {str(k): np.nonzero(lab == k)[0] for k in dict.fromkeys(lab.tolist())}
        covered = np.zeros(N, bool)
        for v in gidx.values():
            covered[v] = True
        if not covered.all():
            if rest in gidx:
                raise ValueError(f"그룹에 속하지 않은 노드를 넣을 그룹 이름 '{rest}'가 이미 있음")
            gidx[rest] = np.nonzero(~covered)[0]
        if meta is None and labeled:
            meta = pd.DataFrame({"node": [str(k) for k in order]})
        c = cls(np.arange(N), gidx, pre_i, post_i, w, name=name, meta=meta, pos=pos)
        c.check()
        return c

    @classmethod
    def from_scipy(cls, matrix, orientation: str = "pre_post", **kw) -> "Circuit":
        """연결 행렬 (scipy 희소 또는 numpy). orientation="pre_post"면 A[i, j] = i → j (networkx와 같음),
        "post_pre"면 A[j, i] = i → j. 0이 아닌 칸이 연결, 값이 세기"""
        sps = _scipy_sparse()
        if orientation not in ("pre_post", "post_pre"):
            raise ValueError("orientation은 'pre_post' 또는 'post_pre'")
        A = sps.coo_matrix(matrix)
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"정사각 행렬이어야 함: {A.shape}")
        A.sum_duplicates()
        keep = A.data != 0
        r, c, v = A.row[keep], A.col[keep], A.data[keep]
        pre, post = (r, c) if orientation == "pre_post" else (c, r)
        kw.setdefault("meta", None)
        out = cls.from_edges(pre, post, v, **{k: x for k, x in kw.items() if k != "meta"},
                             meta=kw["meta"] if kw["meta"] is not None else pd.DataFrame(index=range(A.shape[0])))
        return out

    @classmethod
    def from_networkx(cls, G, weight: str | None = "weight", group: str | None = "group", **kw) -> "Circuit":
        """networkx 그래프. 방향 없는 그래프는 양쪽 방향 연결로. 노드 속성(숫자·문자)은 meta 열,
        group 속성이 있으면 그룹. 노드 순서 = G.nodes 순서, 노드 이름은 meta["node"]"""
        nodes = list(G.nodes)
        index = {n: i for i, n in enumerate(nodes)}
        pre, post, w = [], [], []
        for a, b, d in G.edges(data=True):
            x = float(d.get(weight, 1.0)) if weight else 1.0
            pre.append(index[a]); post.append(index[b]); w.append(x)
            if not G.is_directed() and a != b:
                pre.append(index[b]); post.append(index[a]); w.append(x)
        attrs = {}
        for n in nodes:
            for k, v in G.nodes[n].items():
                if isinstance(v, (str, int, float, bool, np.integer, np.floating)):
                    attrs.setdefault(k, {})[n] = v
        meta = pd.DataFrame({"node": [str(n) for n in nodes], **{k: [d.get(n) for n in nodes] for k, d in attrs.items()}})
        groups = meta[group].astype(object).where(meta[group].notna(), None).to_numpy() if group in meta else None
        if groups is not None:
            groups = np.array(["rest" if g is None else str(g) for g in groups], dtype=object)
        return cls.from_edges(np.array(pre, np.int64), np.array(post, np.int64), np.array(w, np.float32),
                              groups=groups, meta=meta, name=kw.pop("name", "networkx"), **kw)

    def to_scipy(self, orientation: str = "pre_post"):
        """연결 행렬 (scipy CSR). 같은 쌍의 연결은 더함"""
        sps = _scipy_sparse()
        r, c = (self.pre, self.post) if orientation == "pre_post" else (self.post, self.pre)
        return sps.csr_matrix((self.weight, (r, c)), shape=(self.N, self.N))

    def to_networkx(self):
        """networkx.DiGraph (노드 = 회로 번호, 속성 group·meta 열, 연결 속성 weight)"""
        try:
            import networkx as nx
        except ImportError as e:
            raise ImportError('networkx가 필요한 기능 - pip install "flydnet[graph]" (또는 pip install networkx)') from e
        G = nx.DiGraph()
        g = self.group_of()
        for i in range(self.N):
            attrs = {"group": g[i]}
            if self.meta is not None:
                attrs.update({k: v for k, v in self.meta.iloc[i].items() if not (isinstance(v, float) and np.isnan(v))})
            G.add_node(i, **attrs)
        A = self.to_scipy().tocoo()
        G.add_weighted_edges_from(zip(A.row.tolist(), A.col.tolist(), A.data.tolist()))
        return G

    def regroup(self, groups: dict, rest: str = "rest") -> "Circuit":
        """그룹을 새로 정함 ({이름: 노드 번호}, 나머지는 rest). 연결·주석은 그대로"""
        for k, v in groups.items():
            a = np.asarray(v)
            if a.ndim != 1 or (a.size and (a.dtype.kind not in "iu" or a.min() < 0 or a.max() >= self.N)):
                raise ValueError(f"regroup: 그룹 {k!r}의 값은 노드 번호 배열 (0 ~ {self.N - 1}): {v!r}"[:200])
        return Circuit.from_edges(self.pre, self.post, self.weight, groups={k: np.asarray(v) for k, v in groups.items()},
                                  meta=self.meta if self.meta is not None else pd.DataFrame(index=range(self.N)),
                                  pos=self.pos, name=self.name, rest=rest)._with_ids(self.root_ids)

    def _with_ids(self, ids):
        self.root_ids = np.asarray(ids, np.int64)
        return self

    def check(self) -> "Circuit":
        """구조 확인: 번호 범위, 그룹 겹침, 세기 유한, 주석·위치 길이. 문제가 있으면 ValueError"""
        N = self.N
        if len(self.pre) != len(self.post) or len(self.pre) != len(self.weight):
            raise ValueError(f"pre·post·weight 길이가 다름: {len(self.pre)}, {len(self.post)}, {len(self.weight)}")
        if len(self.pre) and (min(self.pre.min(), self.post.min()) < 0 or max(self.pre.max(), self.post.max()) >= N):
            raise ValueError(f"연결의 노드 번호가 0 ~ {N - 1} 밖")
        if not np.isfinite(self.weight).all():
            raise ValueError("연결 세기에 NaN·무한대")
        seen = np.zeros(N, np.int64)
        for k, v in self.groups.items():
            if len(v) and (v.min() < 0 or v.max() >= N):
                raise ValueError(f"그룹 {k}의 번호가 0 ~ {N - 1} 밖")
            np.add.at(seen, v, 1)
        if (seen > 1).any():
            raise ValueError(f"여러 그룹에 속한 노드가 있음 (예: {np.nonzero(seen > 1)[0][:5].tolist()})")
        if self.meta is not None and len(self.meta) != N:
            raise ValueError(f"meta 행 수 {len(self.meta)} ≠ 노드 수 {N}")
        if self.pos is not None and len(self.pos) != N:
            raise ValueError(f"pos 행 수 {len(self.pos)} ≠ 노드 수 {N}")
        return self

    @classmethod
    def celegans(cls, synapses: str = "chemical", data_dir: str | Path | None = None) -> "Circuit":
        """예쁜꼬마선충 자웅동체 커넥톰 (Cook et al. 2019): 뉴런 300개 + 근육·기타 세포 148개

        그룹: neuron, body_muscle (체벽 근육), pharynx (인두 근육·주변 세포), other (자궁·음문 근육, 장 등)
        부호: GABA 뉴런 26개 (McIntire et al. 1993: DD1-6, VD1-13, RME 4개, AVL, DVB, RIS)가 주는 화학 시냅스는 억제
        synapses: "chemical" (기본) / "electrical" / "both". 전기 시냅스(간극 연결)는 이 모델에 따로 없어서
                  양방향 흥분 연결로 근사 (목록에 있는 방향 그대로)
        세기 = 시냅스 수 (전자현미경). 주석 열: node (뉴런 이름, 예 "ASEL"), cell_class, gaba
        뉴런 고르기: fd.genetics.driver(worm, node=["ASEL", "ASER"])"""
        import re
        from .data import require
        if synapses not in ("chemical", "electrical", "both"):
            raise ValueError("synapses는 'chemical', 'electrical', 'both'")
        d = pd.read_csv(require("worm", data_dir) / "herm_full_edgelist.csv")
        d.columns = [c.strip() for c in d.columns]
        for c in ("Source", "Target", "Type"):
            d[c] = d[c].astype(str).str.strip()
        names = sorted(set(d.Source) | set(d.Target))
        if synapses != "both":
            d = d[d.Type == synapses]
        gaba = set([f"DD0{i}" for i in range(1, 7)] + [f"VD{i:02d}" for i in range(1, 14)]
                   + ["RMEL", "RMER", "RMED", "RMEV", "AVL", "DVB", "RIS"])

        def kind(n):
            if not re.search(r"[a-z]", n):
                return "neuron"
            if "BWM" in n:
                return "body_muscle"
            if re.match(r"(pm|mc)\d", n):
                return "pharynx"
            return "other"
        cls_ = [kind(n) for n in names]
        sign = np.where((d.Type == "chemical") & d.Source.isin(gaba), -1.0, 1.0)
        meta = pd.DataFrame({"node": names, "cell_class": cls_, "gaba": [n in gaba for n in names]})
        return cls.from_edges(d.Source.to_numpy(), d.Target.to_numpy(), (d.Weight.to_numpy() * sign).astype(np.float32),
                              groups=np.array(cls_, dtype=object), names=names, meta=meta,
                              name=f"C. elegans (Cook 2019, {synapses})")

    def shuffled(self, seed: int = 0, pairs=None, exclude=None, local=None, merge=None) -> "Circuit":
        """무작위 배선 대조군: (보내는 그룹, 받는 그룹) 쌍마다 받는 뉴런을 섞음.
        그룹 간 연결 수·시냅스 수·뉴런별 입출력 개수는 그대로, '누가 누구에게'만 무작위.

        pairs:   섞을 쌍만 지정 (예: ["PN>KC"]). None이면 전부
        exclude: 섞지 않을 쌍 (예: ["PN>KC", "KC>KC"])
        local:   (xy, radius) 이면 받는 뉴런 위치 xy (N, 2)를 radius 크기 칸으로 나눠 같은 칸 안에서만 섞음
                 → 시야 위치 대응(retinotopy)은 유지, 그 안의 미세 배선(누가 정확히 누구에게)만 무작위.
                 위치가 NaN인 뉴런은 한 칸으로 묶음
        merge:   {그룹: 별칭} 섞을 때 같은 별칭 그룹들을 하나로 봄 (예: T4a~d → "T4").
                 받는 뉴런이 원래 다른 아형으로 가던 연결도 받게 됨 → 아형별 배선 차이(방향 구조)가 사라짐.
                 pairs/exclude는 별칭이 아닌 원래 그룹 이름 기준
        """
        _C.optional(_C.pos, 'local 반경', local[1] if local is not None else None)
        pairs = [pairs] if isinstance(pairs, str) else pairs              # 쌍 하나 (예전: 글자마다 쌍으로 봄 - '>')
        exclude = [exclude] if isinstance(exclude, str) else exclude
        rng = np.random.default_rng(seed)
        g = self.group_of()
        post = self.post.copy()
        key = pd.Series(g[self.pre] + ">" + g[self.post])
        if pairs is not None and (unknown := set(pairs) - set(key.unique())):
            raise ValueError(f"회로에 없는 연결 쌍: {sorted(unknown)}")
        pair_of = key.values
        if merge:
            ga = np.array([merge.get(x, x) for x in g], dtype=object)
            key = pd.Series(ga[self.pre] + ">" + ga[self.post])
        if local is not None:
            xy, radius = local
            b = np.floor(np.asarray(xy)[self.post] / radius)
            b = np.where(np.isnan(b).any(1, keepdims=True), np.inf, b)
            key = key + "|" + pd.Series(b[:, 0]).astype(str) + "," + pd.Series(b[:, 1]).astype(str)
        stuck = []
        for k, idx in key.groupby(key).groups.items():
            idx = np.asarray(idx)
            if pairs is not None or exclude is not None:
                pk = pair_of[idx]
                sel = np.ones(len(idx), bool)
                if pairs is not None:
                    sel &= np.isin(pk, list(pairs))
                if exclude is not None:
                    sel &= ~np.isin(pk, list(exclude))
                idx = idx[sel]
                if len(idx) == 0:
                    continue
            try:
                post[idx] = _repair(self.pre[idx], post[idx][rng.permutation(len(idx))], self.N, rng)
            except RuntimeError:                                 # 뉴런 몇 개뿐인 쌍은 중복 없이 섞을 수 없을 때가 있음
                stuck.append((k, len(idx)))
        if stuck:
            import warnings
            warnings.warn(f"중복 없이 섞을 수 없어 원래 배선으로 둔 연결 쌍 {len(stuck)}개 "
                          f"(연결 {sum(n for _, n in stuck):,}개 / 전체 {self.n_edges:,}개): "
                          f"{', '.join(k for k, _ in stuck[:5])}{' …' if len(stuck) > 5 else ''}")
        what = "" if pairs is None and exclude is None else \
            f" {'+'.join(pairs) if pairs is not None else 'all'}{' -' + '-'.join(exclude) if exclude else ''}"
        if local is not None:
            what += f" local r={local[1]:g}"
        if merge:
            what += f" merge {'+'.join(sorted(set(merge.values())))}"
        return Circuit(self.root_ids, self.groups, self.pre, post, self.weight,
                       name=f"{self.name} [shuffled{what}]", meta=self.meta, pos=self.pos)

    def _pair_index(self):
        """(보내는 그룹, 받는 그룹) 쌍마다 연결 번호들"""
        g = self.group_of()
        key = pd.Series(g[self.pre] + ">" + g[self.post])
        return {k: np.asarray(v) for k, v in key.groupby(key).groups.items()}

    def randomized(self, seed: int = 0) -> "Circuit":
        """더 강한 무작위 대조군: 그룹 쌍마다 연결 수와 시냅스 수 목록만 유지하고, 연결 상대를 그룹 안에서 균등하게 다시 뽑음.
        shuffled()와 달리 뉴런별 연결 수(차수) 분포도 무작위가 됨 (허브·차수 구조가 중요한지 묻는 대조군)"""
        rng = np.random.default_rng(seed)
        member = {}
        for gname, idx in self.groups.items():
            member[gname] = np.asarray(idx)
        member.setdefault("?", np.nonzero(self.group_of() == "?")[0])
        new_pre, new_post = self.pre.copy(), self.post.copy()
        for k, idx in self._pair_index().items():
            a, _, b = k.partition(">")
            src, dst = member[a], member[b]
            E, nd = len(idx), len(dst)
            total = len(src) * nd
            size = min(total, E)
            while True:                                                  # 겹치지 않게 뽑고 자기 연결은 빼기
                cand = rng.choice(total, size, replace=False)            # 무작위 순서 → 앞 E개가 균등 표본
                cand = cand[src[cand // nd] != dst[cand % nd]]
                if len(cand) >= E:
                    break
                if size == total:
                    raise ValueError(f"{k}: 연결 {E}개가 가능한 쌍보다 많음")
                size = min(total, size + 2 * (E - len(cand)) + 16)
            new_pre[idx] = src[cand[:E] // nd]
            new_post[idx] = dst[cand[:E] % nd]
        return Circuit(self.root_ids, self.groups, new_pre, new_post, self.weight,
                       name=f"{self.name} [randomized]", meta=self.meta, pos=self.pos)

    def shuffled_weights(self, seed: int = 0) -> "Circuit":
        """배선은 그대로, 그룹 쌍마다 시냅스 수(세기)만 연결끼리 섞음 (세기 분포가 중요한지 묻는 대조군)"""
        rng = np.random.default_rng(seed)
        w = self.weight.copy()
        for idx in self._pair_index().values():
            w[idx] = w[idx][rng.permutation(len(idx))]
        return Circuit(self.root_ids, self.groups, self.pre, self.post, w,
                       name=f"{self.name} [shuffled weights]", meta=self.meta, pos=self.pos)

    def subset(self, groups) -> "Circuit":
        """지정한 그룹들의 뉴런만 남긴 회로 (그 사이 연결만)"""
        groups = list(dict.fromkeys([groups] if isinstance(groups, str) else groups))   # 이름 하나·중복도
        unknown = [g for g in groups if g not in self.groups]
        if unknown or not groups:
            raise KeyError(f"회로에 없는 그룹: {unknown} (있는 것: {list(self.groups)[:20]})" if unknown else "그룹을 하나 이상")
        keep = np.concatenate([self.groups[g] for g in groups])
        new = np.full(self.N, -1, np.int64); new[keep] = np.arange(len(keep))
        m = (new[self.pre] >= 0) & (new[self.post] >= 0)
        gidx, o = {}, 0
        for g in groups:
            gidx[g] = np.arange(o, o + len(self.groups[g])); o += len(self.groups[g])
        return Circuit(self.root_ids[keep], gidx, new[self.pre[m]], new[self.post[m]], self.weight[m],
                       name=f"{self.name} [그룹 {len(groups)}개]",
                       meta=self.meta.iloc[keep] if self.meta is not None else None,
                       pos=self.pos[keep] if self.pos is not None else None)

    def normalized(self) -> "Circuit":
        """받는 뉴런마다 입력 시냅스 수 합(|weight|)이 1이 되도록 나눈 회로.
        입력 비율(누가 얼마나 주는지)은 그대로, 입력이 많은 뉴런과 적은 뉴런의 총입력 크기만 맞춤"""
        tot = np.bincount(self.post, weights=np.abs(self.weight), minlength=self.N)
        t = tot[self.post]
        w = np.where(t > 0, self.weight / np.where(t > 0, t, 1), 0).astype(np.float32)   # 입력이 모두 0인 뉴런은 0 (NaN 아님)
        return Circuit(self.root_ids, self.groups, self.pre, self.post, w, name=f"{self.name} [정규화]",
                       meta=self.meta, pos=self.pos)

    def with_sign(self, pre_groups, sign: int) -> "Circuit":
        """pre_groups 뉴런이 보내는 연결을 모두 흥분(+1) 또는 억제(-1)로 바꾼 회로.
        예: 광수용체의 히스타민은 받는 뉴런을 억제하지만 신경전달물질 예측에는 흥분으로 잡힘"""
        if sign not in (1, -1):
            raise ValueError("sign은 1 또는 -1")
        pre_groups = [pre_groups] if isinstance(pre_groups, str) else list(pre_groups)
        m = np.isin(self.pre, np.concatenate([self.groups[g] for g in pre_groups]))
        w = self.weight.copy(); w[m] = sign * np.abs(w[m])
        return Circuit(self.root_ids, self.groups, self.pre, self.post, w,
                       name=f"{self.name} [{'+'.join(pre_groups)} {'흥분' if sign > 0 else '억제'}]",
                       meta=self.meta, pos=self.pos)

    def summary(self) -> pd.DataFrame:
        """그룹 간 연결 요약 (시냅스 수, 흥분/억제)"""
        g = self.group_of()
        df = pd.DataFrame({"pre": g[self.pre], "post": g[self.post], "w": self.weight})
        return df.groupby(["pre", "post"]).agg(edges=("w", "size"), exc_syn=("w", lambda x: x[x > 0].sum()),
                                               inh_syn=("w", lambda x: (-x[x < 0]).sum()))

    def __repr__(self):
        gs = ", ".join(f"{k} {len(v)}" for k, v in self.groups.items())
        return f"<Circuit '{self.name}' | {self.N:,} neurons ({gs}) | {self.n_edges:,} edges>"

    # 저장 (torch 없이): numpy 배열과 JSON 문자열만 → np.savez(allow_pickle=False)로 안전하게 읽힘
    def to_arrays(self, prefix: str = "") -> dict:
        import json
        out = {prefix + "root_ids": self.root_ids, prefix + "pre": self.pre, prefix + "post": self.post,
               prefix + "weight": self.weight}
        info = dict(name=self.name, groups=list(self.groups))
        if self.meta is not None:
            info["meta"] = {str(c): [_json_value(v) for v in self.meta[c]] for c in self.meta.columns}
        out[prefix + "info"] = np.array(json.dumps(info, ensure_ascii=False))
        for i, g in enumerate(self.groups.values()):
            out[f"{prefix}group{i}"] = g
        if self.pos is not None:
            out[prefix + "pos"] = self.pos
        return out

    @classmethod
    def from_arrays(cls, d, prefix: str = "") -> "Circuit":
        import json
        info = json.loads(str(d[prefix + "info"]))
        groups = {g: np.asarray(d[f"{prefix}group{i}"]) for i, g in enumerate(info["groups"])}
        meta = pd.DataFrame(info["meta"]) if info.get("meta") is not None else None
        if meta is not None:                                       # JSON null → NaN (원래 주석과 같게 - pandas 1.x는 None으로 남았음)
            meta = meta.where(meta.notna(), np.nan)
        pos = np.asarray(d[prefix + "pos"]) if prefix + "pos" in d else None
        return cls(np.asarray(d[prefix + "root_ids"]), groups, np.asarray(d[prefix + "pre"]),
                   np.asarray(d[prefix + "post"]), np.asarray(d[prefix + "weight"]), name=info["name"],
                   meta=meta, pos=pos)
