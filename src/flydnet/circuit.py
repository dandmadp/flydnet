"""FlyWire 커넥톰에서 뉴런 묶음(회로)을 잘라내 신경망 층의 '배선'으로 쓰는 Circuit"""
from __future__ import annotations

from pathlib import Path

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
                     data_dir: str | Path | None = None, group_by: str | None = None, annotations: str = "flywire_annotations.tsv",
                     connectivity: str = "Connectivity_783.parquet",
                     completeness: str = "Completeness_783.csv") -> "Circuit":
        """주석으로 고른 뉴런들 사이의 연결만 남긴 회로 (induced subgraph)
        group_by: 주석 열 이름 (예: "cell_type")이면 각 그룹을 그 값마다 다시 나눔 → 그룹 이름 = 값
                  (값이 없는 뉴런은 "<그룹 이름>?"). 무작위 대조군(shuffled)도 이 단위로 섞임
        data_dir: None이면 flydnet.data_dir("flywire") (환경변수 → ~/.flydnet/config.json → ~/.flydnet/data)"""
        from .data import require
        d = require("flywire", data_dir)
        all_ids = pd.read_csv(d / completeness, index_col=0).index.values.astype(np.int64)
        ann = pd.read_csv(d / annotations, sep="\t", low_memory=False,
                          usecols=["root_id", "super_class", "cell_class", "cell_sub_class", "cell_type", "side",
                                   "pos_x", "pos_y", "pos_z"])
        ann = ann[ann.root_id.isin(all_ids)]
        if side:
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
        a = ann.set_index("root_id").loc[ids]
        meta = a[["super_class", "cell_class", "cell_sub_class", "cell_type"]].reset_index()
        return cls(ids, gidx, pre[keep], post[keep], w, name=f"FlyWire {'/'.join(groups)} ({side or 'both'})",
                   meta=meta, pos=a[["pos_x", "pos_y", "pos_z"]].values)

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
        stuck = []
        for k, idx in key.groupby(key).groups.items():
            if (pairs is not None and k not in pairs) or (exclude is not None and k in exclude):
                continue
            idx = np.asarray(idx)
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
        return Circuit(self.root_ids, self.groups, self.pre, post, self.weight,
                       name=f"{self.name} [shuffled{what}]", meta=self.meta, pos=self.pos)

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
                                               inh_syn=("w", lambda x: -x[x < 0].sum()))

    def __repr__(self):
        gs = ", ".join(f"{k} {len(v)}" for k, v in self.groups.items())
        return f"<Circuit '{self.name}' | {self.N:,} neurons ({gs}) | {self.n_edges:,} edges>"

    # 저장: 텐서·문자열·리스트만 써서 torch.load(weights_only=True)로 안전하게 읽힘
    def to_dict(self) -> dict:
        import torch
        meta = None
        if self.meta is not None:
            meta = {c: [None if pd.isna(v) else str(v) for v in self.meta[c]] for c in self.meta.columns}
        return dict(root_ids=torch.from_numpy(self.root_ids), pre=torch.from_numpy(self.pre),
                    post=torch.from_numpy(self.post), weight=torch.from_numpy(self.weight),
                    groups={k: torch.from_numpy(v) for k, v in self.groups.items()}, name=self.name, meta=meta,
                    pos=torch.from_numpy(self.pos) if self.pos is not None else None)

    @classmethod
    def from_dict(cls, d: dict) -> "Circuit":
        meta = pd.DataFrame(d["meta"]) if d.get("meta") is not None else None
        return cls(d["root_ids"].numpy(), {k: v.numpy() for k, v in d["groups"].items()}, d["pre"].numpy(),
                   d["post"].numpy(), d["weight"].numpy(), name=d["name"], meta=meta,
                   pos=d["pos"].numpy() if d.get("pos") is not None else None)
