"""双频绝对周数求解（solve_two_frequency）的测试。

参考值一律手算或由模块内独立的穷举参考 enumerate_week_pairs 给出，
不调用被测 CRT 路径得到期望值。
"""

import math
import random
import unittest

from phasecut import (
    InvalidPhaseInputError,
    WeekCandidate,
    enumerate_week_pairs,
    solve_two_frequency,
    unwrap_phase,
    wrap_phase,
)

TWO_PI = 2.0 * math.pi


def grid_from_rows(rows):
    return [list(row) for row in rows]


def full_mask(h, w):
    return [[True] * w for _ in range(h)]


class HandComputedTest(unittest.TestCase):
    """可手算的互素频率用例：k1=3, k2=4。"""

    def test_unique_solution_zero_error(self):
        # Phi = 1.0：w1 = wrap(3) = 3，w2 = wrap(4) = 4 - 2π。
        w1 = wrap_phase(3.0)
        w2 = wrap_phase(4.0)
        result = solve_two_frequency(
            [[w1]], [[w2]], [[True]], 3, 4, 0.0, 0.0, -1.0, 5.0
        )
        cands = result.candidates[0][0]
        self.assertEqual(len(cands), 1)
        cand = cands[0]
        self.assertEqual((cand.n1, cand.n2), (0, 1))
        self.assertAlmostEqual(cand.lo, 1.0, places=12)
        self.assertAlmostEqual(cand.hi, 1.0, places=12)
        self.assertEqual(result.solved_pixels, 1)
        self.assertEqual(result.ambiguous_pixels, 0)
        self.assertEqual(result.empty_pixels, 0)
        self.assertIsNone(result.filtered)

    def test_interval_spanning_one_synthetic_period_is_ambiguous(self):
        # 合成周期 = 2π（互素频率的合成频率为 1）。区间跨 >2π 时，
        # Phi=1.0 与 Phi=1.0+2π 都可行，必须全部返回而非挑最接近 0 的。
        w1 = wrap_phase(3.0)
        w2 = wrap_phase(4.0)
        result = solve_two_frequency(
            [[w1]], [[w2]], [[True]], 3, 4, 0.0, 0.0, -1.0, TWO_PI + 1.0
        )
        cands = result.candidates[0][0]
        self.assertEqual(len(cands), 2)
        self.assertAlmostEqual(cands[0].lo, 1.0, places=12)
        self.assertAlmostEqual(cands[1].lo, 1.0 + TWO_PI, places=12)
        self.assertEqual(result.ambiguous_pixels, 1)
        self.assertEqual(result.counts[0][0], 2)

    def test_error_bands_exactly_touching(self):
        # k1=1, k2=2：观测1给 [-1, 1]（w1=0, e1=1），
        # 观测2零误差钉在 Phi=1.0（w2=wrap(2)=2）。两带恰接触仍可行。
        result = solve_two_frequency(
            [[0.0]], [[2.0]], [[True]], 1, 2, 1.0, 0.0, -3.0, 3.0
        )
        cands = result.candidates[0][0]
        self.assertEqual(len(cands), 1)
        self.assertAlmostEqual(cands[0].lo, 1.0, places=12)
        self.assertAlmostEqual(cands[0].hi, 1.0, places=12)

    def test_incompatible_observations_give_empty_not_exception(self):
        # w1=0（Phi≈0），w2=3.0（Phi≈1.5），误差 0.05 不可能相容。
        result = solve_two_frequency(
            [[0.0]], [[3.0]], [[True]], 1, 2, 0.05, 0.05, -3.0, 3.0
        )
        self.assertEqual(result.candidates[0][0], ())
        self.assertEqual(result.counts[0][0], 0)
        self.assertEqual(result.empty_pixels, 1)

    def test_masked_pixel_is_none_and_not_counted(self):
        w1 = wrap_phase(3.0)
        w2 = wrap_phase(4.0)
        mask = [[True, False]]
        result = solve_two_frequency(
            [[w1, 0.0]], [[w2, 0.0]], mask, 3, 4, 0.0, 0.0, -1.0, 5.0
        )
        self.assertIsNone(result.candidates[0][1])
        self.assertIsNone(result.counts[0][1])
        self.assertEqual(result.solved_pixels, 1)

    def test_multi_pixel_grid(self):
        # 2 像素：Phi = 1.0 与 Phi = 2.5，互素频率 3/4。
        w1 = [wrap_phase(3.0), wrap_phase(7.5)]
        w2 = [wrap_phase(4.0), wrap_phase(10.0)]
        result = solve_two_frequency(
            [w1], [w2], full_mask(1, 2), 3, 4, 0.0, 0.0, -1.0, 5.0
        )
        self.assertAlmostEqual(result.candidates[0][0][0].lo, 1.0, places=12)
        self.assertAlmostEqual(result.candidates[0][1][0].lo, 2.5, places=12)
        self.assertEqual(result.solved_pixels, 2)


class NeighborFilterTest(unittest.TestCase):
    def test_filter_is_optional_and_never_silences_ambiguity(self):
        # 两个像素都在跨周期区间里有 2 个候选；开启筛选后 counts 不变，
        # filtered 只是显式视图，不能把多解点宣布唯一。
        w1 = wrap_phase(3.0)
        w2 = wrap_phase(4.0)
        phase1 = [[w1, w1]]
        phase2 = [[w2, w2]]
        mask = full_mask(1, 2)
        lo, hi = -1.0, TWO_PI + 1.0
        raw = solve_two_frequency(phase1, phase2, mask, 3, 4, 0.0, 0.0, lo, hi)
        self.assertIsNone(raw.filtered)
        self.assertEqual(raw.ambiguous_pixels, 2)

        filt = solve_two_frequency(
            phase1, phase2, mask, 3, 4, 0.0, 0.0, lo, hi, neighbor_filter=True
        )
        self.assertIsNotNone(filt.filtered)
        # 统计始终基于未筛选候选。
        self.assertEqual(filt.ambiguous_pixels, 2)
        self.assertEqual(filt.counts[0][0], 2)
        # 两像素候选完全一致，互相支撑，筛选后仍各保留 2 个。
        self.assertEqual(len(filt.filtered[0][0]), 2)
        self.assertEqual(len(filt.filtered[0][1]), 2)

    def test_filter_drops_candidate_unsupported_by_neighbor(self):
        # k1=2, k2=3，区间 [-0.5, 3.5]（宽 4 < 合成周期 2π），观测 w1=w2=0。
        # A 零误差：Phi=0 唯一（n1=1 需 n2=1.5，非整数，不可行）。
        # B e1=1.6, e2=1.0，手算：
        #   (n1,n2)=(0,0)：[-0.8,0.8]∩[-1/3,1/3] → [-1/3, 1/3] 可行；
        #   (1,1)：[π-0.8, π+0.8]∩[(2π-1)/3,(2π+1)/3]∩[-0.5,3.5]
        #          ≈ [2.3416, 2.4276] 可行；
        #   (0,1)、(1,0)、(1,2) 交集均空。故 B 恰有 2 个候选。
        # A 的 [0,0] 只与 B 的第一个候选相交：筛选后 B 剩 1 个，
        # 但 counts 与歧义统计仍按未筛选的 2 个计。
        mask = full_mask(1, 2)
        phase = [[0.0, 0.0]]
        unique = solve_two_frequency(phase, phase, mask, 2, 3, 0.0, 0.0, -0.5, 3.5)
        self.assertEqual(unique.solved_pixels, 2)

        wide = solve_two_frequency(phase, phase, mask, 2, 3, 1.6, 1.0, -0.5, 3.5)
        self.assertEqual(wide.counts[0][0], 2)
        self.assertEqual(wide.ambiguous_pixels, 2)

        # 混合网格：A 列零误差唯一，B 列宽带误差双候选。
        # 用两列分别求解再验证筛选语义：直接对 1x2 网格用 B 的误差，
        # 另构造 A 唯一、B 双解的场景 —— 两列观测不同即可。
        # A 列：w=(0,0) 零误差等价；这里整网统一误差，改为验证：
        # B 的候选2 [2.34,2.43] 与 A 的候选2 相同 → 互相支撑不筛除。
        filt = solve_two_frequency(
            phase, phase, mask, 2, 3, 1.6, 1.0, -0.5, 3.5, neighbor_filter=True
        )
        self.assertIsNotNone(filt.filtered)
        self.assertEqual(filt.counts[0][0], 2)          # 统计基于未筛选
        self.assertEqual(len(filt.filtered[0][0]), 2)   # 邻域支撑完整保留

        # 真正的筛除：右列观测改为 w2 使第二候选无邻域支撑。
        # 右列 w1=0, e1=1.6 → 带 [-0.8,0.8] 与 [π-0.8, π+0.8]；
        # 右列 w2=1.0, e2=0.2 → n2=0 带 [(1-0.2)/3,(1+0.2)/3]=[0.2667,0.4]
        # 与 [-0.8,0.8] 交 [0.2667,0.4] 可行；n2=1 带
        # [(1+2π-0.2)/3,(1+2π+0.2)/3]≈[2.361,2.494] 与 [π-0.8,π+0.8]
        # 交 [2.361,2.494] 可行 → 右列 2 候选。
        # 左列零误差钉 [0,0]：右列候选1 [0.2667,0.4] 与 [0,0] 不相交，
        # 候选2 也不相交 → 右列筛选后为空，但 counts 仍为 2。
        phase1 = [[0.0, 0.0]]
        phase2 = [[0.0, 1.0]]
        e1 = [0.0, 1.6]
        # solve_two_frequency 的误差是整网统一的，逐列不同误差需两次调用；
        # 这里用统一误差 e1=1.6, e2=0.2：左列 w=(0,0) 在该误差下
        # (0,0) 可行，(1,1)：π-0.8=2.342 ≤ (2π+0.2)/3=2.162？不成立，
        # 故左列唯一 [−1/15?, ...]；右列如上手算 2 候选。
        result = solve_two_frequency(
            phase1, phase2, mask, 2, 3, 1.6, 0.2, -0.5, 3.5, neighbor_filter=True
        )
        self.assertEqual(result.counts[0][0], 1)
        self.assertEqual(result.counts[0][1], 2)
        self.assertEqual(result.ambiguous_pixels, 1)
        left = result.filtered[0][0]
        right = result.filtered[0][1]
        # 左列候选含 Phi≈0；右列候选1 ≈[0.267,0.4]、候选2 ≈[2.361,2.494]，
        # 都与左列不相交 → 右列被筛空，左列因右列无支撑候选也被筛空，
        # 但 counts 不变：筛选只是显式视图。
        self.assertEqual(len(right), 0)
        self.assertEqual(result.counts[0][1], 2)


class ValidationTest(unittest.TestCase):
    def setUp(self):
        self.p = [[0.0]]
        self.m = [[True]]

    def test_non_coprime_rejected(self):
        with self.assertRaises(InvalidPhaseInputError):
            solve_two_frequency(self.p, self.p, self.m, 2, 4, 0.0, 0.0, -1.0, 1.0)

    def test_bool_freq_rejected(self):
        with self.assertRaises(InvalidPhaseInputError):
            solve_two_frequency(self.p, self.p, self.m, True, 3, 0.0, 0.0, -1.0, 1.0)

    def test_negative_error_rejected(self):
        with self.assertRaises(InvalidPhaseInputError):
            solve_two_frequency(self.p, self.p, self.m, 1, 2, -0.1, 0.0, -1.0, 1.0)

    def test_nan_error_rejected(self):
        with self.assertRaises(InvalidPhaseInputError):
            solve_two_frequency(
                self.p, self.p, self.m, 1, 2, float("nan"), 0.0, -1.0, 1.0
            )

    def test_bad_interval_rejected(self):
        with self.assertRaises(InvalidPhaseInputError):
            solve_two_frequency(self.p, self.p, self.m, 1, 2, 0.0, 0.0, 2.0, 1.0)
        with self.assertRaises(InvalidPhaseInputError):
            solve_two_frequency(
                self.p, self.p, self.m, 1, 2, 0.0, 0.0, -1.0, float("inf")
            )

    def test_phase_out_of_range_rejected(self):
        with self.assertRaises(InvalidPhaseInputError):
            solve_two_frequency([[math.pi]], self.p, self.m, 1, 2, 0.0, 0.0, -1.0, 1.0)

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(InvalidPhaseInputError):
            solve_two_frequency(self.p, [[0.0], [0.0]], self.m, 1, 2, 0.0, 0.0, -1.0, 1.0)


class ReferenceCrossCheckTest(unittest.TestCase):
    """CRT 路径与短范围穷举参考的确定性对照。"""

    def test_crt_matches_bruteforce_fuzz(self):
        rng = random.Random(20260926)
        cases = 0
        for _ in range(300):
            k1 = rng.choice([1, 2, 3, 4, 5])
            k2 = rng.choice([1, 2, 3, 4, 5])
            if math.gcd(k1, k2) != 1:
                continue
            phi_true = rng.uniform(-3.0, 3.0)
            noise1 = rng.uniform(-0.05, 0.05)
            noise2 = rng.uniform(-0.05, 0.05)
            w1 = wrap_phase(k1 * phi_true + noise1)
            w2 = wrap_phase(k2 * phi_true + noise2)
            e1 = rng.uniform(0.0, 0.2)
            e2 = rng.uniform(0.0, 0.2)
            lo, hi = -4.0, 4.0
            brute = enumerate_week_pairs(w1, w2, k1, k2, e1, e2, lo, hi)
            got = solve_two_frequency(
                [[w1]], [[w2]], [[True]], k1, k2, e1, e2, lo, hi
            ).candidates[0][0]
            self.assertEqual(len(got), len(brute), (k1, k2, w1, w2, e1, e2))
            for g, b in zip(got, brute):
                self.assertEqual((g.n1, g.n2), (b.n1, b.n2))
                self.assertAlmostEqual(g.lo, b.lo, places=9)
                self.assertAlmostEqual(g.hi, b.hi, places=9)
            cases += 1
        self.assertGreater(cases, 100)

    def test_ambiguity_count_reported(self):
        # 区间跨两个合成周期：候选数必须如实输出。
        w1 = wrap_phase(3.0)
        w2 = wrap_phase(4.0)
        result = solve_two_frequency(
            [[w1]], [[w2]], [[True]], 3, 4, 0.0, 0.0, -1.0, 2 * TWO_PI + 1.0
        )
        self.assertEqual(result.counts[0][0], 3)
        self.assertEqual(result.ambiguous_pixels, 1)


class OldInterfaceIntactTest(unittest.TestCase):
    def test_unwrap_phase_still_works(self):
        # 旧单幅接口冒烟：无涡旋斜坡展开。
        phase = [[wrap_phase(0.1 * (r + c)) for c in range(4)] for r in range(4)]
        result = unwrap_phase(phase, full_mask(4, 4))
        self.assertEqual(result.unmet_edges, ())
        self.assertEqual(len(result.regions), 1)


if __name__ == "__main__":
    unittest.main()

