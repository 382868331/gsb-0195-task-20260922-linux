"""align_frames（相邻两帧全局整数周对齐）的必要测试。

旧空间展开（unwrap_phase）的测试全部保留在 test_phasecut.py，本文件只覆盖
本轮新增的时间维入口。期望值不调用被测实现产生：

- 最优 shift 由测试内对整数 k∈[-K,K] 的独立小规模枚举（穷举 SSE）给出，
  并列时独立枚举同样取较小整数，与核心“凸二次只需比较 floor/floor+1”
  的实现互相独立；
- 跨 2π、精确半点并列、无共同像素等用例的 SSE 用手算闭式值核对；
- 两帧各自的空间结果直接与 unwrap_phase 的公开结果比对（旧入口契约）；
- 确定性 fuzz 只做性质检验（枚举 argmin 一致、掩码保留、纯函数）。
"""

import math
import random
import unittest

import phasecut
from phasecut import (
    DEFAULT_SEARCH_BUDGET,
    TWO_PI,
    FrameAlignment,
    InvalidPhaseInputError,
    NoSolutionError,
    SearchBudgetExhausted,
    align_frames,
    unwrap_phase,
    wrap_phase,
)

PI = math.pi
ENUM_HALF_RANGE = 8  # 独立枚举范围 k ∈ [-8, 8]，覆盖 32x32 内任何可能值


def all_true(h, w):
    return [[True] * w for _ in range(h)]


def independent_enum_shift(result_a, result_b, mask_a, mask_b):
    """独立小规模枚举：在共同有效像素上穷举 k，返回 (k, sse, shared)。

    SSE 直接由两个展开网格读数计算；并列取较小整数（range 升序、严格
    更优才替换），与核心的 floor/floor+1 解析做法完全独立。
    """
    h, w = len(mask_a), len(mask_a[0])
    shared = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if mask_a[r][c] and mask_b[r][c]
    ]
    d = [result_a.unwrapped[r][c] - result_b.unwrapped[r][c] for r, c in shared]
    best_k, best_sse = None, None
    for k in range(-ENUM_HALF_RANGE, ENUM_HALF_RANGE + 1):
        sse = sum((x - k * TWO_PI) ** 2 for x in d)
        if best_sse is None or sse < best_sse:  # 严格更优才换 -> 并列留小 k
            best_k, best_sse = k, sse
    return best_k, best_sse, shared


class TestAlignNormal(unittest.TestCase):
    def test_identical_frames_shift_zero_zero_sse(self):
        # 同一无涡旋斜坡输入两帧：shift=0，SSE 恰为 0，全部像素共同有效。
        h = w = 10
        g = [[wrap_phase(0.9 * r + 2.2 * c) for c in range(w)] for r in range(h)]
        mask = all_true(h, w)
        al = align_frames(g, mask, [row[:] for row in g], [row[:] for row in mask])

        self.assertIsInstance(al, FrameAlignment)
        self.assertTrue(al.aligned)
        self.assertEqual(al.shift, 0)
        self.assertEqual(al.squared_error, 0.0)
        self.assertEqual(len(al.shared_pixels), h * w)
        self.assertEqual(al.shared_pixels[0], (0, 0))
        self.assertEqual(al.shared_pixels[-1], (h - 1, w - 1))
        # 各自的空间结果就是旧入口对同输入的结果（旧管线原样复用）。
        self.assertEqual(al.frame_a, unwrap_phase(g, mask))
        self.assertEqual(al.frame_b, unwrap_phase(g, mask))
        self.assertEqual(al.frame_a.residues, ())
        self.assertEqual(al.frame_a.cuts, ())

    def test_global_cycle_across_two_pi_positive(self):
        # 手算跨 2π 用例：两个互不相邻的有效像素（各自成区，展开值=包裹值）。
        # frame_a = +π-ε，frame_b = -π+ε；物理上同点跨一个 2π 边界。
        # d = 2π-2ε ≈ 6.263：k=1 残差 -2ε，k=0 残差 ≈ 2π。
        eps = 0.01
        ga = [[PI - eps, 0.0], [0.0, PI - eps]]
        gb = [[-PI + eps, 0.0], [0.0, -PI + eps]]
        ma = [[True, False], [False, True]]
        mb = [[True, False], [False, True]]
        al = align_frames(ga, ma, gb, mb)

        self.assertTrue(al.aligned)
        self.assertEqual(al.shift, 1)
        self.assertEqual(al.shared_pixels, ((0, 0), (1, 1)))
        # 闭式手算：SSE(1) = 2*(2ε)^2 = 8e-4。
        self.assertAlmostEqual(al.squared_error, 2.0 * (2.0 * eps) ** 2, places=12)
        # 对齐后共同像素残差逐点核对：d-2π = -2ε（符号由 d 定义决定）。
        for r, c in al.shared_pixels:
            aligned_b = al.frame_b.unwrapped[r][c] + al.shift * TWO_PI
            self.assertAlmostEqual(
                al.frame_a.unwrapped[r][c] - aligned_b, -2.0 * eps, places=12
            )
        # 独立枚举给出同一 argmin。
        k, sse, _ = independent_enum_shift(al.frame_a, al.frame_b, ma, mb)
        self.assertEqual((k, sse), (1, al.squared_error))

    def test_global_cycle_across_two_pi_negative(self):
        # 反向跨边界：frame_a=-π+ε, frame_b=+π-ε -> d=-2π+2ε -> k=-1。
        eps = 0.02
        ga = [[-PI + eps]]
        gb = [[PI - eps]]
        mask = [[True]]
        al = align_frames(ga, mask, gb, mask)
        self.assertEqual(al.shift, -1)
        # 残差 d-(-1)·2π = (-2π+2ε)+2π = 2ε。
        self.assertAlmostEqual(al.squared_error, (2.0 * eps) ** 2, places=12)
        k, sse, _ = independent_enum_shift(al.frame_a, al.frame_b, mask, mask)
        self.assertEqual((k, sse), (-1, al.squared_error))

    def test_tie_at_exact_half_integer_picks_smaller(self):
        # 精确半点并列：d_i = π（两个像素），mean/2π = 0.5，
        # SSE(0)=2π²，SSE(1)=2(π-2π)²=2π²，必须取较小整数 0。
        ga = [[0.0, 0.0]]
        gb = [[-PI, -PI]]  # -π 是合法输入（半开区间闭端）
        mask = [[True, True]]
        al = align_frames(ga, mask, gb, mask)
        self.assertTrue(al.aligned)
        self.assertEqual(al.shift, 0)
        self.assertAlmostEqual(al.squared_error, 2.0 * PI * PI, places=12)
        # 独立枚举（升序、严格更优才换）同样停在 0。
        k, sse, _ = independent_enum_shift(al.frame_a, al.frame_b, mask, mask)
        self.assertEqual(k, 0)
        self.assertAlmostEqual(sse, 2.0 * PI * PI, places=12)

    def test_only_shared_pixels_count_local_gaps(self):
        # 局部缺测：frame_a 整行 3 像素有效，frame_b 仅两端有效。
        # 中间 A 独占像素取极端值 -π+ε：若被错误计入，均值会被拉向 k=-1；
        # 正确实现只在两端共同像素上选择 -> 两端 d 相同，shift 由它们决定。
        eps = 0.01
        ga = [[PI - eps, -PI + eps, PI - eps]]
        gb = [[-PI + eps, 0.0, -PI + eps]]
        ma = [[True, True, True]]
        mb = [[True, False, True]]
        al = align_frames(ga, ma, gb, mb)
        self.assertEqual(al.shared_pixels, ((0, 0), (0, 2)))
        self.assertEqual(al.shift, 1)
        self.assertAlmostEqual(al.squared_error, 2.0 * (2.0 * eps) ** 2, places=12)
        # 掩码原样保留：A 中间有值，B 中间为 None。
        self.assertIsNotNone(al.frame_a.unwrapped[0][1])
        self.assertIsNone(al.frame_b.unwrapped[0][1])
        k, sse, shared = independent_enum_shift(al.frame_a, al.frame_b, ma, mb)
        self.assertEqual((shared, k, sse), (
            [(0, 0), (0, 2)], 1, al.squared_error
        ))

    def test_frames_keep_own_residues_and_cuts(self):
        # 两帧各自带涡旋（不同位置 -> 不同切口分区），全局 k 仍按
        # 共同像素 SSE 选出；帧结果携带各自的 residues/cuts/unmet。
        h = w = 9

        def vortex(vr, vc):
            return [
                [wrap_phase(math.atan2(r - (vr + 0.5), c - (vc + 0.5)))
                 for c in range(w)]
                for r in range(h)
            ]

        ga, gb = vortex(2, 2), vortex(4, 4)
        ma = mb = all_true(h, w)
        al = align_frames(ga, ma, gb, mb)
        self.assertEqual(
            [(x.row, x.col, x.charge) for x in al.frame_a.residues], [(2, 2, 1)]
        )
        self.assertEqual(
            [(x.row, x.col, x.charge) for x in al.frame_b.residues], [(4, 4, 1)]
        )
        self.assertTrue(al.frame_a.cuts and al.frame_b.cuts)
        self.assertEqual(al.frame_a.unmet_edges, ())
        self.assertEqual(al.frame_b.unmet_edges, ())
        self.assertEqual(al.frame_a, unwrap_phase(ga, ma))
        self.assertEqual(al.frame_b, unwrap_phase(gb, mb))
        # 独立枚举（含远离 0 的 k）确认全局最优。
        k, _sse, shared = independent_enum_shift(al.frame_a, al.frame_b, ma, mb)
        self.assertEqual(len(shared), h * w)
        self.assertEqual(al.shift, k)
        self.assertEqual(al.shift, 0)

    def test_constant_sub_pi_temporal_evolution_shift_zero(self):
        # 两帧为同一斜坡加 2.0 rad 常量时间演化（|2|<π）：锚点差异不足
        # 一个整周，正确 shift=0，SSE 为每像素常量差的平方。
        h, w = 4, 6
        ga = [[wrap_phase(0.9 * r + 2.2 * c) for c in range(w)] for r in range(h)]
        gb = [[wrap_phase(0.9 * r + 2.2 * c + 2.0) for c in range(w)]
              for r in range(h)]
        ma = mb = all_true(h, w)
        al = align_frames(ga, ma, gb, mb)
        self.assertEqual(al.shift, 0)
        # 独立洪泛下两帧均单区，逐像素差恒为 -2.0（锚点包裹值之差）。
        for r, c in al.shared_pixels:
            self.assertAlmostEqual(
                al.frame_a.unwrapped[r][c] - al.frame_b.unwrapped[r][c],
                -2.0,
                places=9,
            )
        self.assertAlmostEqual(al.squared_error, h * w * 4.0, places=6)


class TestAlignNoOverlap(unittest.TestCase):
    def test_no_shared_pixel_is_unaligned_not_error(self):
        # 两帧掩码完全错开：返回未对齐状态而非异常；掩码/帧结果保留。
        ga = [[0.1, 0.2], [0.3, 0.4]]
        gb = [[0.5, -0.5], [1.0, -1.0]]
        ma = [[True, False], [False, True]]
        mb = [[False, True], [True, False]]
        al = align_frames(ga, ma, gb, mb)

        self.assertIsInstance(al, FrameAlignment)
        self.assertFalse(al.aligned)
        self.assertIsNone(al.shift)
        self.assertIsNone(al.squared_error)
        self.assertEqual(al.shared_pixels, ())
        # 两帧仍各自完成空间展开，无效位为 None，有效位保留。
        self.assertEqual(al.frame_a.unwrapped[0][0], 0.1)
        self.assertIsNone(al.frame_a.unwrapped[0][1])
        self.assertEqual(al.frame_b.unwrapped[0][1], -0.5)
        self.assertIsNone(al.frame_b.unwrapped[0][0])
        self.assertEqual(al.frame_a, unwrap_phase(ga, ma))
        self.assertEqual(al.frame_b, unwrap_phase(gb, mb))

    def test_one_frame_all_masked_out(self):
        ga = gb = [[0.0, 0.1], [0.2, 0.3]]
        ma = [[True] * 2 for _ in range(2)]
        mb = [[False] * 2 for _ in range(2)]
        al = align_frames(ga, ma, gb, mb)
        self.assertFalse(al.aligned)
        self.assertIsNone(al.shift)
        self.assertEqual(al.shared_pixels, ())
        # B 全无效：0 个区、全 None 网格，但仍是正常返回。
        self.assertEqual(al.frame_b.regions, ())
        self.assertTrue(
            all(v is None for row in al.frame_b.unwrapped for v in row)
        )


class TestAlignValidationAndFailures(unittest.TestCase):
    def _g(self):
        return [[0.1, 0.2], [0.3, 0.4]]

    def _m(self):
        return [[True, True], [True, True]]

    def test_shape_mismatch_between_frames_rejected(self):
        with self.assertRaises(InvalidPhaseInputError):
            align_frames(
                self._g(), self._m(),
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[True] * 3 for _ in range(2)],
            )
        with self.assertRaises(InvalidPhaseInputError):
            align_frames(
                self._g(), self._m(),
                [[0.1, 0.2]], [[True, True]],
            )

    def test_invalid_value_in_either_frame_rejected_with_location(self):
        # frame_a 非法：与单帧同样的可定位错误。
        bad_a = self._g()
        bad_a[1][0] = float("nan")
        with self.assertRaises(InvalidPhaseInputError) as cm:
            align_frames(bad_a, self._m(), self._g(), self._m())
        self.assertEqual((cm.exception.row, cm.exception.col), (1, 0))
        # frame_b 非法（+pi 越界）同样拒绝。
        bad_b = self._g()
        bad_b[0][1] = PI
        with self.assertRaises(InvalidPhaseInputError) as cm:
            align_frames(self._g(), self._m(), bad_b, self._m())
        self.assertEqual((cm.exception.row, cm.exception.col), (0, 1))

    def test_bad_mask_and_budget_types_rejected(self):
        with self.assertRaises(InvalidPhaseInputError):
            align_frames(self._g(), [[1, 1], [1, 1]], self._g(), self._m())
        with self.assertRaises(InvalidPhaseInputError):
            align_frames(self._g(), self._m(), self._g(), self._m(),
                         search_budget=0)
        with self.assertRaises(InvalidPhaseInputError):
            align_frames(self._g(), self._m(), self._g(), self._m(),
                         search_budget=True)

    def test_per_frame_no_solution_propagates_with_no_partial_result(self):
        # frame_b 唯一残差困在无效海孤岛：align 必须整体失败（不返回
        # FrameAlignment），与 unwrap_phase 的 NoSolutionError 语义一致。
        h = w = 11
        island = [[False] * w for _ in range(h)]
        for r in range(4, 7):
            for c in range(4, 7):
                island[r][c] = True
        ga = [[0.0] * w for _ in range(h)]
        gb = [
            [wrap_phase(math.atan2(r - 4.5, c - 4.5)) if island[r][c] else 0.0
             for c in range(w)]
            for r in range(h)
        ]
        with self.assertRaises(NoSolutionError):
            align_frames(ga, [[True] * w for _ in range(h)], gb, island)

    def test_per_frame_budget_exhausted_propagates(self):
        # 任一帧 BFS 预算耗尽 -> SearchBudgetExhausted（“算不完”），
        # 不会静默返回未对齐或 shift=0。
        h = w = 9
        g = [
            [wrap_phase(math.atan2(r - 4.5, c - 4.5)) for c in range(w)]
            for r in range(h)
        ]
        mask = all_true(h, w)
        with self.assertRaises(SearchBudgetExhausted):
            align_frames(g, mask, g, mask, search_budget=1)
        # 同输入默认预算即成功。
        al = align_frames(g, mask, g, mask)
        self.assertTrue(al.aligned)
        self.assertEqual(al.shift, 0)
        self.assertEqual(al.squared_error, 0.0)


class TestAlignDeterminismAndPure(unittest.TestCase):
    def test_deterministic_and_inputs_unchanged(self):
        h = w = 8
        rng = random.Random(424242)
        ma = [[rng.random() < 0.8 for _ in range(w)] for _ in range(h)]
        mb = [[rng.random() < 0.8 for _ in range(w)] for _ in range(h)]
        ga = [[wrap_phase(rng.uniform(-PI + 1e-6, PI - 1e-6)) for _ in range(w)]
              for _ in range(h)]
        gb = [[wrap_phase(rng.uniform(-PI + 1e-6, PI - 1e-6)) for _ in range(w)]
              for _ in range(h)]
        snap = ([row[:] for row in ga], [row[:] for row in ma],
                [row[:] for row in gb], [row[:] for row in mb])
        r1 = align_frames(ga, ma, gb, mb)
        r2 = align_frames([row[:] for row in ga], [row[:] for row in ma],
                          [row[:] for row in gb], [row[:] for row in mb])
        self.assertEqual(r1, r2)
        self.assertEqual((ga, ma, gb, mb), snap)

    def test_default_budget_constant_exported(self):
        al = align_frames([[0.1]], [[True]], [[0.2]], [[True]],
                          search_budget=DEFAULT_SEARCH_BUDGET)
        self.assertEqual(al.shift, 0)


class TestAlignFuzz(unittest.TestCase):
    """固定种子随机两帧：选择逻辑必须与独立枚举 argmin 处处一致。"""

    def test_random_frames_match_independent_enumeration(self):
        rng = random.Random(20260924)
        aligned_cases = unaligned_cases = 0
        nonzero_shifts = 0
        for _ in range(120):
            h = rng.randint(1, 10)
            w = rng.randint(1, 10)
            pa = rng.choice((0.5, 0.7, 1.0))
            pb = rng.choice((0.5, 0.7, 1.0))
            ma = [[rng.random() < pa for _ in range(w)] for _ in range(h)]
            mb = [[rng.random() < pb for _ in range(w)] for _ in range(h)]
            # 两帧同一基础场 + 小幅时间扰动，制造真实的相邻帧关系。
            base = [[rng.uniform(-PI + 1e-6, PI - 1e-6) for _ in range(w)]
                    for _ in range(h)]
            ga = [[wrap_phase(base[r][c]) for c in range(w)] for r in range(h)]
            gb = [[wrap_phase(base[r][c] + rng.uniform(-0.4, 0.4))
                   for c in range(w)] for r in range(h)]
            try:
                al = align_frames(ga, ma, gb, mb)
            except NoSolutionError:
                continue  # 单帧无解语义已在旧测试覆盖
            k, sse, shared = independent_enum_shift(
                al.frame_a, al.frame_b, ma, mb
            )
            if not shared:
                unaligned_cases += 1
                self.assertFalse(al.aligned)
                self.assertIsNone(al.shift)
                self.assertIsNone(al.squared_error)
                continue
            aligned_cases += 1
            self.assertTrue(al.aligned)
            self.assertEqual(al.shift, k)
            self.assertEqual(list(al.shared_pixels), shared)
            self.assertAlmostEqual(al.squared_error, sse, places=8)
            # 全局单一整数 k 的 SSE 最优，性质在“均值”而非逐点：
            # 均值差 mean(d)-2πk 必落在半周 [-π,π] 内（凸二次 argmin）；
            # 单个残差可超 π——两帧切口分区基准不同，一个全局整数本就
            # 无法逐点抹平（这正是只做全局整数周、不做逐像素运动的含义）。
            raw_d = [
                al.frame_a.unwrapped[r][c] - al.frame_b.unwrapped[r][c]
                for r, c in shared
            ]
            mean_d = sum(raw_d) / len(raw_d)
            self.assertLessEqual(abs(mean_d - al.shift * TWO_PI), PI + 1e-9)
            # 局部最优独立复核：k 对相邻整数 k±1 不更差。
            sse_here = sum((x - al.shift * TWO_PI) ** 2 for x in raw_d)
            self.assertLessEqual(sse_here,
                                 sum((x - (al.shift - 1) * TWO_PI) ** 2
                                     for x in raw_d) + 1e-9)
            self.assertLessEqual(sse_here,
                                 sum((x - (al.shift + 1) * TWO_PI) ** 2
                                     for x in raw_d) + 1e-9)
            if k != 0:
                nonzero_shifts += 1
        self.assertGreater(aligned_cases, 80)
        self.assertGreaterEqual(unaligned_cases, 1)
        # 稀疏 mask 下必然出现非零整周（孤立像素包裹值之差可超 π）。
        self.assertGreaterEqual(nonzero_shifts, 1)


if __name__ == "__main__":
    unittest.main()
