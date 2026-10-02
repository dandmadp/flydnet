"""FlyWire 커넥톰에서 뉴런 묶음(회로)을 잘라내 신경망 층의 '배선'으로 쓰는 Circuit"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA = Path(os.environ.get("FLYDNET_DATA", r"D:\flybrain_local\Drosophila_brain_model"))

# 이름 → (주석 열, 값). 버섯체 기본 회로: 투사 뉴런 → Kenyon 세포 → MBON, APL이 전체를 억제
MUSHROOM_BODY = {
    "PN":   ("cell_class", "ALPN"),
    "KC":   ("cell_class", "Kenyon_Cell"),
    "APL":  ("cell_type", "APL"),
    "MBON": ("cell_class", "MBON"),
}


def _repair(pre: np.ndarray, post: np.ndarray, N: int, rng, max_iter: int = 1000) -> np.ndarray:
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


class Circuit:
    """뉴런 N개와 부호 있는 시냅스 목록 (pre → post, weight = ±시냅스 수)

    groups: 이름 → 이 회로 안에서의 뉴런 번호 배열 (예: groups["KC"])
    meta:   뉴런별 주석 (cell_type, cell_sub_class 등), 행 순서 = 회로 번호
    """

    def __init__(self, root_ids, groups: dict[str, np.ndarray], pre, post, weight, name="circuit",
                 meta: pd.DataFrame | None = None):
        self.root_ids = np.asarray(root_ids, dtype=np.int64)
        self.groups = {k: np.asarray(v, dtype=np.int64) for k, v in groups.items()}
        self.pre = np.asarray(pre, dtype=np.int64)
        self.post = np.asarray(post, dtype=np.int64)
        self.weight = np.asarray(weight, dtype=np.float32)
        self.name = name
        self.meta = meta.reset_index(drop=True) if meta is not None else None

    @property
    def N(self) -> int:
        return len(self.root_ids)

    @property
    def n_edges(self) -> int:
        return len(self.pre)

    def group_of(self) -> np.ndarray:
        """뉴런별 그룹 이름 (그룹이 겹치지 않는다고 가정)"""
        g = np.empty(self.N, dtype=object)
        for k, v in self.groups.items():
            g[v] = k
        return g

    @classmethod
    def from_flywire(cls, groups: dict = MUSHROOM_BODY, side: str | None = "right",
                     data_dir: str | Path = DEFAULT_DATA, annotations: str = "flywire_annotations.tsv",
                     connectivity: str = "Connectivity_783.parquet",
                     completeness: str = "Completeness_783.csv") -> "Circuit":
        """주석으로 고른 뉴런들 사이의 연결만 남긴 회로 (induced subgraph)"""
        d = Path(data_dir)
        all_ids = pd.read_csv(d / completeness, index_col=0).index.values.astype(np.int64)
        ann = pd.read_csv(d / annotations, sep="\t", low_memory=False,
                          usecols=["root_id", "cell_class", "cell_sub_class", "cell_type", "side"])
        ann = ann[ann.root_id.isin(all_ids)]
        if side:
            ann = ann[ann.side == side]

        picked, gidx = [], {}
        for name, (col, val) in groups.items():
            ids = ann.loc[ann[col] == val, "root_id"].values
            gidx[name] = np.arange(len(picked), len(picked) + len(ids))
            picked.extend(ids)
        ids = np.array(picked, dtype=np.int64)
        if len(np.unique(ids)) != len(ids):
            raise ValueError("그룹끼리 뉴런이 겹침")

        # 전체 뇌 번호 → 회로 번호
        glob = pd.Series(np.arange(len(all_ids)), index=all_ids)[ids].values
        local = np.full(len(all_ids), -1, np.int64); local[glob] = np.arange(len(ids))

        df = pd.read_parquet(d / connectivity, columns=["Presynaptic_Index", "Postsynaptic_Index",
                                                         "Connectivity", "Excitatory"])
        pre, post = local[df.Presynaptic_Index.values], local[df.Postsynaptic_Index.values]
        keep = (pre >= 0) & (post >= 0)
        w = (df.Connectivity.values * df.Excitatory.values)[keep]
        meta = ann.set_index("root_id").loc[ids, ["cell_class", "cell_sub_class", "cell_type"]].reset_index()
        return cls(ids, gidx, pre[keep], post[keep], w, name=f"FlyWire {'/'.join(groups)} ({side or 'both'})",
                   meta=meta)

    def shuffled(self, seed: int = 0, pairs=None, exclude=None) -> "Circuit":
        """무작위 배선 대조군: (보내는 그룹, 받는 그룹) 쌍마다 받는 뉴런을 섞음.
        그룹 간 연결 수·시냅스 수·뉴런별 입출력 개수는 그대로, '누가 누구에게'만 무작위.

        pairs:   섞을 쌍만 지정 (예: ["PN>KC"]). None이면 전부
        exclude: 섞지 않을 쌍 (예: ["PN>KC", "KC>KC"])
        """
        rng = np.random.default_rng(seed)
        g = self.group_of()
        post = self.post.copy()
        key = pd.Series(g[self.pre] + ">" + g[self.post])
        if pairs is not None and (unknown := set(pairs) - set(key.unique())):
            raise ValueError(f"회로에 없는 연결 쌍: {sorted(unknown)}")
        for k, idx in key.groupby(key).groups.items():
            if (pairs is not None and k not in pairs) or (exclude is not None and k in exclude):
                continue
            idx = np.asarray(idx)
            post[idx] = _repair(self.pre[idx], post[idx][rng.permutation(len(idx))], self.N, rng)
        what = "" if pairs is None and exclude is None else \
            f" {'+'.join(pairs) if pairs is not None else 'all'}{' -' + '-'.join(exclude) if exclude else ''}"
        return Circuit(self.root_ids, self.groups, self.pre, post, self.weight,
                       name=f"{self.name} [shuffled{what}]", meta=self.meta)

    def summary(self) -> pd.DataFrame:
        """그룹 간 연결 요약 (시냅스 수, 흥분/억제)"""
        g = self.group_of()
        df = pd.DataFrame({"pre": g[self.pre], "post": g[self.post], "w": self.weight})
        return df.groupby(["pre", "post"]).agg(edges=("w", "size"), exc_syn=("w", lambda x: x[x > 0].sum()),
                                               inh_syn=("w", lambda x: -x[x < 0].sum()))

    def __repr__(self):
        gs = ", ".join(f"{k} {len(v)}" for k, v in self.groups.items())
        return f"<Circuit '{self.name}' | {self.N:,} neurons ({gs}) | {self.n_edges:,} edges>"
