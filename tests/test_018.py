"""0.1.18 회귀 시험"""
import numpy as np
import pytest

import flydnet as fd
from flydnet.ganglion import backend as B


def _loop_circuit():
    """in → h → out 경로 + 입력으로 되돌아가는 고리 (in → back → in) + 출력을 지나 도는 고리 (out → after → out).
    버섯체의 PN → MBON → PN 고리와 같은 모양"""
    rng = np.random.default_rng(0)
    groups = {"in": np.arange(0, 10), "h": np.arange(10, 30), "out": np.arange(30, 40),
              "back": np.arange(40, 45), "after": np.arange(45, 50)}
    pre, post = [], []
    for a, b in (("in", "h"), ("h", "out"), ("in", "back"), ("back", "in"), ("out", "after"), ("after", "out")):
        for i in groups[a]:
            for j in rng.choice(groups[b], size=min(4, len(groups[b])), replace=False):
                pre.append(i), post.append(j)
    w = np.full(len(pre), 5.0, np.float32)
    return fd.Circuit.from_edges(np.array(pre), np.array(post), w, groups=groups)


def test_paths_do_not_loop_through_inputs_or_outputs():
    """입력으로 되돌아가는 우회 (in → back → in → h)와 출력을 지나 도는 고리 (out → after → out)는 경로가 아님 -
    예전에는 버섯체 PN → KC 모델에서 MBON이 중계로 잡혀 calibrate가 MBON 연결을 키우고, trainable="path"가
    PN>MBON·KC>MBON까지 학습했으며, reach가 "신호가 MBON에서 끊김"으로 잘못 판정했음"""
    c = _loop_circuit()
    layer = fd.Connectome(c, "in", "out", t_ms=30, device="cpu")
    assert layer._relay_groups(["out"]) == ["h"]
    assert fd.ConnectomeModel._path_pairs(c, ["in"], ["out"]) == ["h>out", "in>h"]
    rep = layer.reach(np.full((2, 10), 100.0, np.float32))
    assert set(rep.table.role[rep.table.group.isin(["back", "after"])]) == {""}


def test_interpret_names_all_candidates_when_steps_missing():
    """shuffled만 비교해 실제 배선이 이기면, 위치 구조와 세부 배선 둘 다 후보 (예전에는 '세부 배선'으로 단정)"""
    c = _loop_circuit()

    def run(circ, seed):
        noise = np.random.default_rng(seed).normal(0, 0.003)
        return 0.6 + noise if "shuffled" in circ.name else 0.7 + noise
    text = fd.compare(run, c, controls=["shuffled"], seeds=6, verbose=False).interpret()
    assert any("하나 이상" in x and "시야 위치" in x and "세부 배선" in x for x in text), text


def test_sign_flip_p_never_zero_when_sampled():
    """표본 순열 (n > 14)에서도 관측값을 한 경우로 세서 p > 0 (예전에는 0이 나왔음)"""
    p = fd.sign_flip_p(np.full(30, 1.0) + np.random.default_rng(0).normal(0, 0.01, 30), n_perm=1000)
    assert p == 1 / 1001


def test_reach_does_not_blame_relay_when_outputs_fire():
    """출력이 발화하면 꺼진 중계가 있어도 break_at은 None (다른 경로로 신호를 받음)"""
    t = __import__("pandas").DataFrame(dict(group=["in", "a", "out"], role=["입력", "중계", "출력"], n=[1, 1, 1],
                                            hops=[0.0, 1.0, 1.0], unreachable=[0.0, 0.0, 0.0],
                                            rate_hz=[50.0, 0.0, 12.0], active=[1.0, 0.0, 1.0]))
    from flydnet.ganglion.circuitry import Reach
    rep = Reach(t, ["out"])
    assert rep.break_at is None
    s = str(rep)
    assert "출력까지 신호가 감" in s and "중계 a는 0 Hz" in s


def _codes(c):
    return len({tuple(np.nonzero(r)[0]) for r in c})


def test_kc_expansion_keeps_information_with_few_features():
    """특징이 k_in(20) 이하이면 모든 PN이 같은 평균 → 평균 빼기 뒤 0 → 입력과 상관없이 같은 KC 코드였음"""
    from flydnet.data import missing
    import pytest
    if missing("flywire"):
        pytest.skip("FlyWire 데이터 없음")
    X = np.random.default_rng(0).random((6, 16)).astype(np.float32)
    c = fd.KCExpansion(fd.flywire(), n_in=16, device="cpu")(X).numpy()
    assert _codes(c) == 6 and c.max() > 1


def test_load_state_checks_buffer_shapes():
    """다른 크기로 만든 구조물의 상태를 불러오면 버퍼(투영 P 등)를 조용히 바꿔 끼우지 않고 오류"""
    import pytest
    a, b = fd.RateEncoder(10, 6, device="cpu"), fd.RateEncoder(10, 8, device="cpu")
    with pytest.raises(ValueError, match="P"):
        b.load_state(a.state())
    assert b.P.shape == (8, 10)


def test_verify_default_skips_bundles_not_downloaded(tmp_path, monkeypatch, capsys):
    """묶음을 안 주면 기본으로 받지 않는 worm은 받아 두지 않았으면 확인하지 않음 (정상 설치에서 종료 코드 1이 나왔음)"""
    from flydnet import __main__ as M, data as D
    monkeypatch.setenv("FLYDNET_WORM", str(tmp_path / "none"))
    monkeypatch.setattr(D, "verify", lambda kind, path=None: {n: "ok" for n in D.SOURCES[kind]})
    monkeypatch.setattr(M, "verify", D.verify)
    assert M.main(["verify"]) == 0
    assert "worm" not in capsys.readouterr().out


def _small():
    g = fd.graphs.layered([10, 60, 6], 0.3, weight=30, seed=0)            # 출력이 발화하는 세기
    X = np.random.default_rng(0).uniform(20, 100, (4, 10)).astype(np.float32)
    return g, X


def test_single_sample_inputs():
    """시료 하나 (n,)를 받는 곳이 층과 같게: 예전에는 predict·reach·calibrate·Neuropil·DopamineReadout가 오류로 죽음"""
    g, X = _small()
    m = fd.ConnectomeModel(g, "in", "out", n_in=10, n_classes=2, device="cpu", t_ms=20)
    m.fit(X, [0, 1, 0, 1], epochs=1, verbose=False)
    assert np.ndim(m.predict(X[0])) == 0 and m.predict(X[0]) == m.predict(X[:1])[0]
    L = fd.Connectome(g, "in", "out", t_ms=40, device="cpu")
    assert len(L.reach(X[0]).table) == 3
    L.calibrate(X[0], {"out": 12.5}, iters=20)            # 시료 하나·40 ms·6개: 발화율이 약 4.2 Hz 단위
    assert fd.Neuropil(g, "in", "h1", device="cpu")(X[0]).shape == (len(g.groups["h1"]),)
    D = fd.DopamineReadout(10, 2, device="cpu").fit(X.tolist(), [0, 1, 0, 1])
    assert np.ndim(D.predict(X[0])) == 0 and D.predict(X[0]) == D.predict(X[:1])[0]


def test_direct_gain_edit_applies_to_forward():
    """layer.gains[...]를 직접 바꾸면 순전파에도 반영 (예전: 저장 파일에만 들어가 저장 전후 동작이 달랐음)"""
    g, X = _small()
    L = fd.Connectome(g, "in", "out", t_ms=40, device="cpu")
    with fd.quiescent():
        before = L(X, seed=0).numpy()
        L.gains["in>h1"] = 2.0
        after = L(X, seed=0).numpy()
        ref = fd.Connectome(g, "in", "out", t_ms=40, device="cpu", gains={"in>h1": 2.0})(X, seed=0).numpy()
    assert before.mean() > 0 and not np.array_equal(before, after)
    np.testing.assert_array_equal(after, ref)


def test_duplicate_seeds_rejected():
    """같은 seed를 두 번 주면 같은 짝을 두 번 세어 p가 작아짐 (유사 반복) → 오류"""
    import pytest
    g, X = _small()
    with pytest.raises(ValueError, match="같은 값"):
        fd.compare(lambda c, s: 0.5, g, seeds=[0, 0, 1], verbose=False)
    L = fd.Connectome(g, "in", "out", t_ms=20, device="cpu")
    with pytest.raises(ValueError, match="같은 값"):
        fd.genetics.screen(lambda L_, s: 1.0, L, fd.genetics.lines(g, by="group"), seeds=[1, 1], verbose=False)


def test_auto_target_uses_counting_window():
    """자동 목표 발화율은 스파이크를 세는 시간 (t_ms - count_from_ms) 기준"""
    g, _ = _small()
    a = fd.ConnectomeModel(g, "in", "out", n_in=10, n_classes=2, device="cpu", t_ms=200)
    b = fd.ConnectomeModel(g, "in", "out", n_in=10, n_classes=2, device="cpu", t_ms=200, count_from_ms=150)
    assert a.target_hz == 30.0 * 0 + min(30.0, max(5.0, 100 / (6 * 0.2)))
    assert b.target_hz == min(30.0, max(5.0, 100 / (6 * 0.05)))


def test_bool_signal_plus_number_follows_numpy():
    """불리언 신호 + 숫자: 숫자를 불리언으로 바꾸지 않음 (예전 [True, False] + 1 = [True, True] 논리합)"""
    x = np.array([True, False])
    for f in (lambda a: a + 1, lambda a: a * 3, lambda a: 2 - a, lambda a: a + True):
        np.testing.assert_array_equal(f(fd.Signal(x)).numpy(), f(x))


def test_mushroom_body_output_single_sample():
    from flydnet.ganglion.tissue import MushroomBodyOutput
    from flydnet.ganglion.physiology import kenyon_code
    X = np.random.default_rng(0).random((4, 5)).astype(np.float32)
    m = MushroomBodyOutput(5, 2, device="cpu").learn(X, [0, 1, 0, 1])
    assert np.ndim(m.predict(X[0])) == 0 and m.predict(X[0]) == m.predict(X[:1])[0]
    W = np.abs(np.random.default_rng(1).standard_normal((8, 5))).astype(np.float32)
    np.testing.assert_array_equal(kenyon_code(X[0], W, 3), kenyon_code(X[:1], W, 3))


def test_gpu_spmm_segmented_is_exact_and_deterministic():
    """긴 행을 조각내는 전용 커널: scipy와 같은 값, 반복 실행 비트 단위로 같음 (cuSPARSE는 실행마다 다름),
    짧은 행(조각 하나)은 예전 커널(spmm_rm)과 비트 단위로 같음. 버섯체 APL처럼 입력이 수천 개인 행 포함"""
    import pytest
    sps = pytest.importorskip("scipy.sparse")                      # scipy는 선택 설치
    from flydnet.ganglion import kernels as K
    if not fd.ganglion.backend.gpu_available() or K._cuda() is None:
        pytest.skip("GPU 없음")
    import cupy as cp
    rng = np.random.default_rng(0)
    rows = [rng.choice(3000, size=n, replace=False) for n in ([0, 1, 5, 63, 64, 65, 200, 2761] + [10] * 40)]
    indptr = np.r_[0, np.cumsum([len(r) for r in rows])]
    M = sps.csr_matrix((rng.standard_normal(indptr[-1]).astype(np.float32), np.concatenate(rows).astype(np.int32),
                        indptr), shape=(len(rows), 3000))
    Mg = fd.ganglion.backend.sparse("gpu").csr_matrix(M)
    Mg.indices = Mg.indices.astype(np.int32)
    for nb in (1, 7, 33, 256):
        x = rng.standard_normal((3000, nb)).astype(np.float32)
        xg = cp.asarray(x)
        a = K.spmm(Mg, xg)
        np.testing.assert_allclose(cp.asnumpy(a), M @ x, rtol=1e-4, atol=1e-3)
        assert all(bool(cp.array_equal(a, K.spmm(Mg, xg))) for _ in range(3))
        old = cp.empty_like(a)
        gw = min(32, 1 << max(0, (nb - 1).bit_length()))
        K._cuda().get_function("spmm_rm")(((len(rows) * gw + 255) // 256,), (256,),
                                          (Mg.indptr, Mg.indices, Mg.data, xg, old, np.int32(len(rows)), np.int32(nb),
                                           np.int32(gw)))
        short = cp.asarray(np.diff(indptr) <= K.SEG)
        assert bool(cp.array_equal(a[short], old[short]))


def test_neuropil_and_threefactor_deterministic_on_gpu():
    """Neuropil 전달·ThreeFactor(feedback="connectome")가 GPU에서 cuSPARSE 대신 전용 커널 → 반복 실행 비트 단위로 같음"""
    import pytest
    from flydnet.ganglion import backend as B
    if not B.gpu_available():
        pytest.skip("GPU 없음")
    from test_threefactor import _rec
    c = _rec(feedback_edges=True, strong=True)
    x = np.random.default_rng(0).random((16, 6)).astype(np.float32)
    outs = []
    for _ in range(3):
        n = fd.Neuropil(c, "IN", "H", device="gpu")
        xs = fd.Signal(x, plastic=True, device="gpu")
        y = n(xs)
        (y * y).sum().retrograde()
        outs.append((y.numpy(), B.numpy(xs.retro), B.numpy(n.log_scale.retro)))
    assert all(all(np.array_equal(a, b) for a, b in zip(outs[0], o)) for o in outs)
    X = (x * 150 + 50).astype(np.float32)
    res = []
    for _ in range(3):
        L = fd.Connectome(c, "IN", "O", t_ms=60, trainable=True, device="gpu")
        tf = fd.ThreeFactor(L, feedback="connectome")
        o = tf(X, seed=3)
        (o * o).sum().retrograde()
        res.append(B.numpy(tf.assign(o)))
    assert all(np.array_equal(res[0], r) for r in res)


@pytest.mark.skipif(fd.data.missing("flywire"), reason="FlyWire 데이터 없음")
def test_calibrate_damps_oscillation_on_lesioned_mushroom_body():
    """PN→KC 연결 80%를 끊은 버섯체: 예전엔 보폭 0.5로 두 배율 사이를 오가다 KC 8.1 Hz (목표 5)로 끝나며 경고.
    지금은 진동하는 그룹만 보폭을 줄여 목표 안으로. 손상 없는 회로의 배율은 예전과 같음 (위 진동이 없으므로)"""
    import warnings
    mb = fd.flywire()
    enc = fd.Glomeruli(mb)
    g = mb.group_of()
    pk = np.nonzero((g[mb.pre] == "PN") & (g[mb.post] == "KC"))[0]
    cut = np.random.default_rng(1000).choice(pk, int(0.8 * len(pk)), replace=False)
    keep = np.ones(len(mb.pre), bool)
    keep[cut] = False
    c = fd.Circuit(mb.root_ids, mb.groups, mb.pre[keep], mb.post[keep], mb.weight[keep], meta=mb.meta)
    X = fd.door_task(n_odors=24, noise=1.2, seed=0)[0][::3]
    L = fd.Connectome(c, "PN", "KC", t_ms=50, dt=0.5, input_mode="regular", device="cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tab = L.calibrate(enc(X), {"KC": 5})
    assert abs(tab.after_hz[0] - 5) <= 0.5
    L0 = fd.Connectome(mb, "PN", "KC", t_ms=50, dt=0.5, input_mode="regular", device="cpu")
    L0.calibrate(enc(fd.door_task(n_odors=12, seed=0)[0][::3]), {"KC": 5})
    assert L0.gains["PN>KC"] == pytest.approx(6.280486, rel=1e-5)        # 0.1.18 이전 보정과 같은 배율


def test_column_vector_labels_are_flattened_everywhere():
    """(n, 1) 라벨 (scikit-learn 습관): 예전엔 예측 (n,)과 비교할 때 (n, n)으로 퍼져 fd.evaluate가 23.0, AssocReadout.accuracy가
    조용히 다른 값. 이제 어디서나 (n,)으로 펴고, 그 밖의 2차원은 오류"""
    r = np.random.default_rng(0)
    y = r.integers(0, 2, 40)
    X = r.random((40, 6)).astype(np.float32)
    m = fd.Pathway(fd.Projection(6, 2, device="cpu"))
    assert fd.evaluate(m, X, y[:, None]) == fd.evaluate(m, X, y)
    for R in (fd.AssocReadout(6, 2, device="cpu"), fd.DopamineReadout(6, 2, device="cpu")):
        R2 = type(R)(6, 2, device="cpu")
        R.fit(X, y)
        R2.fit(X, y[:, None])
        np.testing.assert_array_equal(R2.W, R.W)
        assert R.accuracy(X, y[:, None]) == R.accuracy(X, y)
    fd.train(fd.Pathway(fd.Projection(6, 2, device="cpu")), X, y[:, None], epochs=1, verbose=False)
    fd.train_linear(X, y[:, None], X, y[:, None], epochs=1, device="cpu")
    with pytest.raises(ValueError, match="1차원"):
        fd.evaluate(m, X, np.c_[y, y])


def test_sign_flip_p_rejects_nan_and_flattens_column():
    """NaN이 섞인 짝 차이: 예전엔 평균이 NaN → 비교가 모두 거짓 → p = 0 (유의하다고 나옴). 이제 오류. (n, 1)은 폄"""
    with pytest.raises(ValueError, match="NaN"):
        fd.sign_flip_p([0.1, np.nan, 0.2, 0.1, 0.3, 0.2])
    with pytest.raises(ValueError, match="NaN"):
        fd.sign_flip_p([0.1, np.inf, 0.2])
    d = np.array([0.1, 0.2, -0.05, 0.3, 0.1, 0.2])
    assert fd.sign_flip_p(d[:, None]) == fd.sign_flip_p(d)
    with pytest.raises(ValueError, match="1차원"):
        fd.sign_flip_p(np.c_[d, d])


def test_readouts_accept_pandas():
    """AssocReadout·DopamineReadout.fit에 pandas: 예전엔 행 이름으로 골라 KeyError"""
    import pandas as pd
    r = np.random.default_rng(0)
    X = r.random((30, 6)).astype(np.float32)
    y = np.arange(30) % 3
    for R in (fd.AssocReadout, fd.DopamineReadout):
        a = R(6, 3, device="cpu").fit(X, y)
        b = R(6, 3, device="cpu").fit(pd.DataFrame(X), pd.Series(y))
        np.testing.assert_array_equal(a.W, b.W)
        assert b.accuracy(pd.DataFrame(X), pd.Series(y)) == a.accuracy(X, y)


def test_evaluate_rejects_non_2d_outputs():
    """출력이 (배치, 시간, 클래스)면 argmax가 (배치, 시간) - 시간 길이 = 배치면 라벨과 퍼져 조용히 틀린 정확도. 이제 오류"""
    class Seq(fd.Tissue):
        def forward(self, x):
            return fd.Signal(np.zeros((x.shape[0], x.shape[0], 3), np.float32))
    with pytest.raises(ValueError, match="시료, 클래스"):
        fd.evaluate(Seq(), np.zeros((4, 2), np.float32), [0, 1, 2, 0], batch=4)


def test_cli_help_succeeds_and_typo_names_command(capsys):
    """--help는 성공(0) - 예전엔 모르는 명령처럼 1. 오타는 무엇이 틀렸는지 알려 주고 1"""
    from flydnet.__main__ import main
    assert main(["--help"]) == 0
    assert main(["dta"]) == 1 and "모르는 명령: 'dta'" in capsys.readouterr().out



@pytest.mark.skipif(not __import__("flydnet").ganglion.backend.gpu_available(), reason="GPU 없음")
def test_where_condition_on_other_device():
    """where(GPU 조건, CPU 신호): 예전엔 GPU 조건을 numpy로 바로 바꾸다 TypeError"""
    from flydnet.ganglion import backend as B
    from flydnet.ganglion.signal import where
    x = np.arange(6, dtype=np.float32) - 2
    for cdev, sdev in (("gpu", "cpu"), ("cpu", "gpu")):
        s = fd.Signal(x, plastic=True, device=sdev)
        o = where(B.to(x > 0, cdev), s, 0.0)
        o.sum().retrograde()
        assert o.device == sdev and np.array_equal(B.numpy(o.data), np.where(x > 0, x, 0))
        assert np.array_equal(B.numpy(s.retro), (x > 0).astype(np.float32))



def test_layer_rejects_overlapping_groups():
    """그룹끼리 뉴런이 겹치는 회로: 예전엔 겹친 입력 뉴런이 두 번 들어가 순전파는 마지막 값만, 역전파는 두 자리 모두 (조용히 틀림)"""
    c = fd.Circuit(np.arange(4), {"A": np.array([0, 1]), "B": np.array([1, 2]), "C": np.array([3])},
                   np.array([0, 1, 2]), np.array([3, 3, 3]), np.ones(3, np.float32))
    with pytest.raises(ValueError, match="겹침"):
        fd.Connectome(c, ["A", "B"], "C", t_ms=20, device="cpu")
    with pytest.raises(ValueError, match="겹침"):                         # 입력에 안 쓰는 그룹이 겹쳐도 (연결 종류가 틀어짐)
        fd.Connectome(c, "C", "A", t_ms=20, device="cpu")
    with pytest.raises(ValueError, match="겹침"):                         # Neuropil도 (예전: 겹친 입력의 첫 자리 값을 버림)
        fd.Neuropil(c, ["A", "B"], "C", device="cpu")
    with pytest.raises(ValueError, match="여러 번"):
        fd.Neuropil(c, ["A", "A"], "C", device="cpu")



def test_trainable_accepts_single_pair_string():
    """trainable="IN>H" (문자열 하나): 예전엔 글자마다 연결 종류로 봐서 엉뚱한 오류"""
    from test_threefactor import _rec
    c = _rec()
    a = fd.Connectome(c, "IN", "O", device="cpu", trainable="IN>H")
    b = fd.Connectome(c, "IN", "O", device="cpu", trainable=["IN>H"])
    assert a.trainable and np.array_equal(B.numpy(a.train_pos), B.numpy(b.train_pos))



def test_compare_ceiling_only_for_unit_scores():
    """점수가 0~1이 아니면 (발화율 Hz 등) '과제가 너무 쉬움' 상한 경고를 내지 않음 - 예전엔 늘 붙었음"""
    from test_threefactor import _rec
    c = _rec()
    run = lambda circ, s: 40.0 + s + float(np.abs(circ.post - circ.pre).mean()) * 1e-3
    rep = fd.compare(run, c, controls=["shuffled"], seeds=6, verbose=False)
    assert not any("너무 쉬워" in w for w in rep.warnings)
    rep = fd.compare(lambda circ, s: 0.995, c, controls=["shuffled"], seeds=6, verbose=False, check_repeat=False)
    assert any("너무 쉬워" in w for w in rep.warnings)



def test_genetics_screen_and_driver_argument_checks():
    """driver(missing=오타)는 오류 (예전: 조용히 ignore처럼), screen에 빈 집단·조작 뒤 NaN은 어느 집단인지 알림"""
    from test_threefactor import _rec
    c = _rec()
    G = fd.genetics
    with pytest.raises(ValueError, match="missing"):
        G.driver(c, group="H", missing="skip")
    L = fd.Connectome(c, "IN", "O", t_ms=20, device="cpu")
    with pytest.raises(ValueError, match="비어"):
        G.screen(lambda l, s: 1.0, L, {}, seeds=2, verbose=False)
    calls = {"n": 0}

    def meas(l, s):
        return float("nan") if G.active(l) else 1.0
    with pytest.raises(ValueError, match="'h'"), pytest.warns(UserWarning):
        G.screen(meas, L, {"h": G.driver(c, group="H")}, seeds=2, verbose=False)



def test_observers_need_spiking_layer():
    """연속값 뉴런 층의 순전파는 관찰자를 부르지 않음 - 예전: STDP는 알기 어려운 AttributeError, 직접 만든 Monitor는 조용히 아무것도 안 함"""
    from test_threefactor import _rec
    L = fd.Connectome(_rec(), "IN", "O", neuron="graded", t_ms=6, dt=1.0, device="cpu", trainable=True)
    with pytest.raises(ValueError, match="스파이킹"):
        fd.STDP(L)
    with pytest.raises(ValueError, match="스파이킹"):
        fd.Monitor(L)



def test_gradcheck_restores_log_scale_in_place():
    """gradcheck가 학습값을 제자리로 되돌림 - 예전엔 사본으로 바꿔 끼워 같은 배열을 쓰던 torch 연결이 끊어짐"""
    from test_threefactor import _rec
    c = _rec(feedback_edges=True, strong=True)
    L = fd.Connectome(c, "IN", "O", t_ms=40, device="cpu", trainable=True, gains={"IN>H": 2.0, "H>O": 2.0})
    X = np.random.default_rng(0).uniform(60, 200, (3, 6)).astype(np.float32)
    before = L.log_scale.data
    snap = before.copy()
    fd.gradcheck(lambda l, s: l(X, seed=s).sum(), L)
    assert L.log_scale.data is before and np.array_equal(before, snap)
    with pytest.raises(ValueError, match="연결 종류가 없음"):
        fd.gradcheck(lambda l, s: l(X, seed=s).sum(), L, min_edges=10 ** 9)



def test_train_with_sequence_labels_returns_history():
    """손실을 직접 주고 (B, T) 라벨이면: 예전엔 모든 에폭을 마친 뒤 train_acc 계산에서 오류로 멈춤 → train_acc None"""
    class Seq(fd.Tissue):
        def __init__(self):
            super().__init__()
            self.p = fd.Projection(4, 3, device="cpu", seed=0)

        def forward(self, x):
            return self.p(x).reshape(x.shape[0], 1, 3) * fd.Signal(np.ones((1, 2, 1), np.float32))   # (B, T=2, C)

    def seq_loss(logits, y):
        return fd.surprise(logits.reshape(-1, 3), np.asarray(y).reshape(-1))
    X = np.random.default_rng(0).random((8, 4)).astype(np.float32)
    y = np.random.default_rng(1).integers(0, 3, (8, 2))
    h = fd.train(Seq(), X, y, epochs=2, loss=seq_loss, verbose=False)
    assert len(h["loss"]) == 2 and h["train_acc"] is None


def test_watts_strogatz_keeps_all_edges():
    """다시 잇기에서 자기 자신·이미 있는 상대를 피함 - 예전엔 겹친 것을 버려 연결이 n·k개보다 적었음 (beta 1에서 1.3% 적음)"""
    for beta in (0.0, 0.1, 0.5, 1.0):
        c = fd.graphs.watts_strogatz(200, 8, beta, seed=3)
        assert c.n_edges == 200 * 8 and not (c.pre == c.post).any()
        assert len(np.unique(c.pre * c.N + c.post)) == c.n_edges
        assert (np.bincount(c.pre, minlength=200) == 8).all()             # 보내는 쪽 차수는 그대로
