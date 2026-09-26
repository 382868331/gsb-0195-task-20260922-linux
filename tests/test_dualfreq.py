"""双频绝对周数确定的必要测试。

参考实现为*独立*的短范围穷举：直接枚举两个频率的整数周数
(n1, n2)，逐组合检验 |f_i*phi - w_i - 2*pi*n_i| <= err_i，
不调用 phasecut 的任何内部函数得到期望值。
"""

import math
import unittest

import phasecut
from phasecut import (
    IncompatibleObservationsError,
    InvalidPhaseInputError,
    resolve_absolute_cycles,
    wrap_phase,
    TWO_PI,
)

PI = math.pi


def ref_candidates(w1, w2, f1, f2, e1, e2, lo, hi):
    """独立参考：穷举所有 (n1, n2) 整数周数组合，逐对求交并数合并段数。

    与核心实现不共享任何代码：核心按 n1 枚举再用周期关系锁 n2，
    这里直接对两个频率各自的全部误差带做笛卡尔积求交。
    """
    n1_range = range(math.floor((f1 * lo - e1 - w1) / TWO_PI) - 1,
                     math.ceil((f1 * hi + e1 - w1) / TWO_PI) + 2)
    n2_range = range(math.floor((f2 * lo - e2 - w2) / TWO_PI) - 1,
                     math.ceil((f2 * hi + e2 - w2) / TWO_PI) + 2)
    bands1 = [((w1 + TWO_PI * n - e1) / f1, (w1 + TWO_PI * n + e1) / f1)
              for n in n1_range]
    bands2 = [((w2 + TWO_PI * n - e2) / f2, (w2 + TWO_PI * n + e2) / f2)
              for n in n2_range]
    segs = []
    for a1, b1 in bands1:
        for a2, b2 in bands2:
            lo_i = max(a1, a2, lo)
            hi_i = min(b1, b2, hi)
            if lo_i <= hi_i + 1e-12:
                segs.append((lo_i, hi_i))
    segs.sort()
    merged = 0
    last_hi = None
    for a, b in segs:
        if last_hi is None or a > last_hi + 1e-12:
            merged += 1
            last_hi = b
        else:
            last_hi = max(last_hi, b)
    return merged


def make_wrapped(phi, f1, f2, noise1=0.0, noise2=0.0):
    return wrap_phase(f1 * phi + noise1), wrap_phase(f2 * phi + noise2)


class TestHandComputed(unittest.TestCase):
    """可手算的互素频率 2 与 3：合成周期 2*pi。"""

    def test_unique_candidate_in_one_period(self):
        # 真值 phi=1.0，区间 [-pi, pi) 恰为一个合成周期 -> 唯一候选。
        w1, w2 = make_wrapped(1.0, 2, 3)
        res = resolve_absolute_cycles(
            [[w1]], [[w2]], 2, 3,
            err1=0.0, err2=0.0, phi_min=-PI, phi_max=PI,
        )
        iv = res.intervals[0][0]
        self.assertEqual(len(iv), 1)
        lo, hi = iv[0]
        self.assertAlmostEqual((lo + hi) / 2, 1.0, places=12)
        self.assertEqual(res.candidate_counts[0][0], 1)
        self.assertEqual(res.ambiguous_pixels, 0)

    def test_interval_spanning_multiple_synthetic_periods(self):
        # 区间跨 3 个合成周期 -> 3 个候选，绝不只挑最接近 0 的。
        w1, w2 = make_wrapped(1.0, 2, 3, noise1=0.003, noise2=-0.002)
        res = resolve_absolute_cycles(
            [[w1]], [[w2]], 2, 3,
            err1=0.01, err2=0.01, phi_min=-10.0, phi_max=10.0,
        )
        iv = res.intervals[0][0]
        self.assertEqual(len(iv), 3)
        centers = [(a + b) / 2 for a, b in iv]
        for expect in (-TWO_PI + 1.0, 1.0, TWO_PI + 1.0):
            self.assertTrue(
                any(abs(c - expect) < 0.01 for c in centers),
                "missing candidate near %r: %r" % (expect, centers),
            )
        self.assertEqual(res.ambiguous_pixels, 1)
        self.assertEqual(res.candidate_counts[0][0], 3)

    def test_error_bands_exactly_touching(self):
        # 误差带恰接触：频率1带中心1.0半宽0.5，频率2带中心2.0半宽0.5，
        # 接触点 phi=1.5 必须可行（含端点）。
        f1, f2 = 2, 3
        w1 = wrap_phase(f1 * 1.0)
        w2 = wrap_phase(f2 * 2.0)
        res = resolve_absolute_cycles(
            [[w1]], [[w2]], f1, f2,
            err1=1.0, err2=1.5, phi_min=0.0, phi_max=3.0,
        )
        iv = res.intervals[0][0]
        self.assertEqual(len(iv), 1)
        lo, hi = iv[0]
        self.assertAlmostEqual(lo, 1.5, places=12)
        self.assertAlmostEqual(hi, 1.5, places=12)

    def test_incompatible_observations_empty(self):
        # 不相容观测：频率1说 phi=0，频率2说 phi=1（err 很小），
        # 同一合成周期内无任何公共点。
        w1 = wrap_phase(2 * 0.0)
        w2 = wrap_phase(3 * 1.0)
        res = resolve_absolute_cycles(
            [[w1]], [[w2]], 2, 3,
            err1=0.05, err2=0.05, phi_min=-PI, phi_max=PI,
        )
        self.assertEqual(res.intervals[0][0], ())
        self.assertEqual(res.empty_pixels, 1)
        self.assertEqual(res.candidate_counts[0][0], 0)

    def test_incompatible_strict_raises(self):
        w1 = wrap_phase(2 * 0.0)
        w2 = wrap_phase(3 * 1.0)
        with self.assertRaises(IncompatibleObservationsError) as ctx:
            resolve_absolute_cycles(
                [[w1]], [[w2]], 2, 3,
                err1=0.05, err2=0.05, phi_min=-PI, phi_max=PI, strict=True,
            )
        self.assertEqual((ctx.exception.row, ctx.exception.col), (0, 0))


class TestAgainstExhaustiveReference(unittest.TestCase):
    """短范围穷举周数参考：候选段数必须与独立参考一致。"""

    def test_random_pixels_match_reference(self):
        import random
        rng = random.Random(20260926)
        for _ in range(60):
            f1, f2 = rng.choice([(2, 3), (3, 4), (2, 5), (3, 5), (4, 5)])
            phi = rng.uniform(-8.0, 8.0)
            e1 = rng.choice([0.0, 0.02, 0.1])
            e2 = rng.choice([0.0, 0.03, 0.1])
            w1, w2 = make_wrapped(
                phi, f1, f2,
                noise1=rng.uniform(-e1, e1) if e1 else 0.0,
                noise2=rng.uniform(-e2, e2) if e2 else 0.0,
            )
            res = resolve_absolute_cycles(
                [[w1]], [[w2]], f1, f2,
                err1=e1, err2=e2, phi_min=-8.0, phi_max=8.0,
            )
            got = res.candidate_counts[0][0]
            expect = ref_candidates(w1, w2, f1, f2, e1, e2, -8.0, 8.0)
            self.assertEqual(got, expect, (f1, f2, phi, e1, e2))
            # 真值必须落在某个可行区间内（噪声在误差带内）。
            self.assertTrue(
                any(a - 1e-9 <= phi <= b + 1e-9
                    for a, b in res.intervals[0][0]),
                (f1, f2, phi),
            )


class TestMaskAndSmoothing(unittest.TestCase):
    def _field(self, phis, f1, f2):
        p1 = [[wrap_phase(f1 * p) for p in row] for row in phis]
        p2 = [[wrap_phase(f2 * p) for p in row] for row in phis]
        return p1, p2

    def test_mask_positions_not_propagated(self):
        phis = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        p1, p2 = self._field(phis, 2, 3)
        mask = [[True, False, True], [True, True, True]]
        res = resolve_absolute_cycles(
            p1, p2, 2, 3, err1=0.01, err2=0.01,
            phi_min=-PI, phi_max=PI, mask=mask,
        )
        self.assertIsNone(res.intervals[0][1])
        self.assertIsNone(res.candidate_counts[0][1])
        # 被掩码的洞不影响其余像素的唯一解。
        self.assertEqual(res.candidate_counts[0][0], 1)
        self.assertEqual(res.candidate_counts[1][1], 1)

    def test_neighborhood_filter_is_opt_in_and_flagged(self):
        # 区间 [-5.3, 6] 跨约两个合成周期：中间像素 phi=1.0 有两个候选
        # {-5.283, 1.0}；两侧像素 phi=0.7 在该区间只有候选 0.7，
        # 其误差带与中间像素的 1.0 候选重叠、与 -5.283 候选分离。
        phis = [[0.7, 1.0, 0.7]]
        p1, p2 = self._field(phis, 2, 3)
        wide = dict(err1=0.4, err2=0.6, phi_min=-5.3, phi_max=6.0)
        # 默认：不做任何平滑，中间像素如实保留 2 个候选。
        res = resolve_absolute_cycles(p1, p2, 2, 3, **wide)
        self.assertEqual(res.candidate_counts[0][1], 2)
        self.assertEqual(res.candidate_counts[0][0], 1)
        self.assertFalse(res.smoothing_applied)
        # 显式开启邻域筛选：邻域共同区间把中间像素筛到唯一。
        res2 = resolve_absolute_cycles(
            p1, p2, 2, 3, neighborhood_filter=True, **wide
        )
        self.assertTrue(res2.smoothing_applied)
        self.assertEqual(res2.candidate_counts[0][1], 1)
        lo, hi = res2.intervals[0][1][0]
        self.assertTrue(lo <= 1.0 <= hi)
        # 被筛掉的 -5.283 候选确实曾是可行解（不是被静默改写输入）。
        self.assertTrue(any(lo < -5.0 for lo, _ in res.intervals[0][1]))


class TestDualValidation(unittest.TestCase):
    def test_rejects_non_coprime(self):
        with self.assertRaises(InvalidPhaseInputError):
            resolve_absolute_cycles(
                [[0.0]], [[0.0]], 2, 4,
                err1=0.0, err2=0.0, phi_min=-1.0, phi_max=1.0,
            )

    def test_rejects_bool_and_bad_scalars(self):
        base = dict(err1=0.0, err2=0.0, phi_min=-1.0, phi_max=1.0)
        with self.assertRaises(InvalidPhaseInputError):
            resolve_absolute_cycles([[0.0]], [[0.0]], True, 3, **base)
        with self.assertRaises(InvalidPhaseInputError):
            resolve_absolute_cycles([[0.0]], [[0.0]], 2, 3,
                                    **{**base, "err1": -0.1})
        with self.assertRaises(InvalidPhaseInputError):
            resolve_absolute_cycles([[0.0]], [[0.0]], 2, 3,
                                    **{**base, "phi_min": 2.0, "phi_max": 1.0})
        with self.assertRaises(InvalidPhaseInputError):
            resolve_absolute_cycles([[0.0]], [[0.0]], 2, 3,
                                    **{**base, "err2": float("nan")})

    def test_rejects_shape_mismatch_and_nan(self):
        base = dict(err1=0.0, err2=0.0, phi_min=-1.0, phi_max=1.0)
        with self.assertRaises(InvalidPhaseInputError):
            resolve_absolute_cycles([[0.0, 0.0]], [[0.0]], 2, 3, **base)
        with self.assertRaises(InvalidPhaseInputError) as ctx:
            resolve_absolute_cycles([[0.0, float("nan")]], [[0.0, 0.0]],
                                    2, 3, **base)
        self.assertEqual((ctx.exception.row, ctx.exception.col), (0, 1))

    def test_old_interface_untouched(self):
        # 原单幅相位展开接口保留。
        res = phasecut.unwrap_phase([[0.1, 0.2], [0.3, 0.4]],
                                    [[True, True], [True, True]])
        self.assertEqual(len(res.regions), 1)
        self.assertEqual(res.unmet_edges, ())


if __name__ == "__main__":
    unittest.main()
