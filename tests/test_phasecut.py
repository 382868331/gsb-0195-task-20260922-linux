"""phasecut 的必要测试。

期望值全部由本文件内的*独立实现*产生：
- 包裹用 atan2(sin x, cos x)（核心用模运算 (x+pi)%2pi-pi）；
- 残差环和用显式四点 atan2 包裹差独立求和；
- 指定电荷的测试场由解析多涡旋场 Phi=Σ s·atan2(...) 包裹生成，
  并由 ref_residues 逐环独立核验实际电荷（不预设生成即正确）；
- 最短路径/等距选择用测试内独立 BFS 复核；
- 切口分区、洪泛一致性用独立拓扑/边校验复核。
不通过调用被测核心来得到期望值；往返一致仅作补充断言。
"""

import math
import random
import unittest
from collections import deque

import phasecut
from phasecut import (
    AlignmentConflictError,
    InvalidPhaseInputError,
    NoSolutionError,
    PhaseReference,
    SearchBudgetExhausted,
    align_regions,
    wrap_phase,
    unwrap_phase,
    TWO_PI,
)

PI = math.pi


# --------------------------------------------------------------------------
# 独立参考实现（与 phasecut 内部无共享代码）
# --------------------------------------------------------------------------
def ref_wrap(x):
    """独立包裹：atan2(sin, cos)，值域 (-pi, pi]，再把 +pi 归一到 -pi。"""
    y = math.atan2(math.sin(x), math.cos(x))
    return -PI if y == PI else y


def ref_loop_sum(w, r, c, mask):
    if not (mask[r][c] and mask[r][c + 1] and mask[r + 1][c + 1]
            and mask[r + 1][c]):
        return None
    d1 = ref_wrap(w[r][c + 1] - w[r][c])
    d2 = ref_wrap(w[r + 1][c + 1] - w[r][c + 1])
    d3 = ref_wrap(w[r + 1][c] - w[r + 1][c + 1])
    d4 = ref_wrap(w[r][c] - w[r + 1][c])
    return d1 + d2 + d3 + d4


def ref_residues(w, mask):
    h, wd = len(mask), len(mask[0])
    out = []
    for r in range(h - 1):
        for c in range(wd - 1):
            s = ref_loop_sum(w, r, c, mask)
            if s is None:
                continue
            q = int(math.floor(abs(s) / TWO_PI + 0.5))
            q = q if s >= 0 else -q
            if q != 0:
                out.append((r, c, q))
    return sorted(out)


def all_true(h, w):
    return [[True] * w for _ in range(h)]


def analytic_vortices(h, w, specs):
    """独立测试数据生成器：在指定环角点放置整数涡旋电荷。

    构造未包裹实场 Phi = Σ s·atan2(r-(R+.5), c-(C+.5)) 后包裹。
    除奇点外场无旋光滑，包裹邻差只在奇点环累计 2π·s；
    奇点相距 >=2 格时单边梯度远小于 pi，不产生寄生电荷。
    实际电荷一律由 ref_residues 独立核验。
    specs: {(R, C): sgn}，奇点位于环 (R,C) 的角点 (R+0.5, C+0.5)。
    """
    grid = [[0.0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            v = 0.0
            for (R, C), s in specs.items():
                v += s * math.atan2(r - (R + 0.5), c - (C + 0.5))
            grid[r][c] = wrap_phase(v)
    return grid


def analytic_vortex(h, w, vr, vc):
    """解析单涡旋 atan2 场（奇点在像素角点 (vr+0.5, vc+0.5)），再包裹。"""
    return analytic_vortices(h, w, {(vr, vc): 1})


def independent_bfs_dist(start, mask, h, w):
    """测试内独立对偶图 BFS 距离（与核心分别实现，方向序也不同）。"""
    dist = {start: 0}
    q = deque([start])
    while q:
        r, c = q.popleft()
        cand = []
        if mask[r][c] and mask[r][c + 1] and r > 0:
            cand.append((r - 1, c))
        if mask[r + 1][c] and mask[r + 1][c + 1] and r < h - 2:
            cand.append((r + 1, c))
        if mask[r][c] and mask[r + 1][c] and c > 0:
            cand.append((r, c - 1))
        if mask[r][c + 1] and mask[r + 1][c + 1] and c < w - 2:
            cand.append((r, c + 1))
        for cell in cand:
            if cell not in dist:
                dist[cell] = dist[(r, c)] + 1
                q.append(cell)
    return dist


def cut_components(cuts, valid):
    """独立：删掉切口边后，有效像素的连通区（锚点=最小坐标）。"""
    blocked = set(cuts)
    seen = set()
    comps = []
    for start in sorted(valid):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        members = []
        while stack:
            p = stack.pop()
            members.append(p)
            r, c = p
            for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
                np_ = (r + dr, c + dc)
                if np_ in valid and np_ not in seen:
                    a, b = sorted((p, np_))
                    if (a[0], a[1], b[0], b[1]) not in blocked:
                        seen.add(np_)
                        stack.append(np_)
        comps.append((min(members), sorted(members)))
    return sorted(comps)


def independent_flood_check(g, mask, cuts):
    """独立洪泛并复核每条非切口边；返回 (区数, 矛盾边列表)。"""
    h, w = len(mask), len(mask[0])
    valid = {(r, c) for r in range(h) for c in range(w) if mask[r][c]}
    blocked = set(cuts)
    phi = {}
    regions = 0
    for start in sorted(valid):
        if start in phi:
            continue
        regions += 1
        phi[start] = g[start[0]][start[1]]
        stack = [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
                nr, nc = r + dr, c + dc
                nxt = (nr, nc)
                if nxt not in valid or nxt in phi:
                    continue
                a, b = sorted(((r, c), nxt))
                if (a[0], a[1], b[0], b[1]) in blocked:
                    continue
                d = ref_wrap(g[nr][nc] - g[r][c])
                expect = phi[(r, c)] + d
                k = int(math.floor(abs((expect - g[nr][nc]) / TWO_PI) + 0.5))
                k = k if expect - g[nr][nc] >= 0 else -k
                phi[nxt] = g[nr][nc] + k * TWO_PI
                stack.append(nxt)
    bad = []
    for (r, c) in sorted(valid):
        for dr, dc in ((0, 1), (1, 0)):
            nr, nc = r + dr, c + dc
            if (nr, nc) not in valid:
                continue
            a, b = sorted(((r, c), (nr, nc)))
            if (a[0], a[1], b[0], b[1]) in blocked:
                continue
            residual = phi[(nr, nc)] - phi[(r, c)] - ref_wrap(
                g[nr][nc] - g[r][c]
            )
            if abs(residual) > 1e-9:
                bad.append((r, c, nr, nc, residual))
    return regions, bad


def edge_is_cuttable(mask, r1, c1, r2, c2):
    """一条像素边能否被某条对偶路径跨过：至少邻接一个四点全有效的环。

    两侧相邻环都无效的边（1 像素洞/细脖子边）不可能成为切口。
    """
    h, w = len(mask), len(mask[0])
    if r1 == r2:  # 水平边 (r,c)-(r,c+1)：上环 (r-1,c)、下环 (r,c)
        r, c = r1, min(c1, c2)
        upper = r > 0 and all((
            mask[r - 1][c], mask[r - 1][c + 1],
            mask[r][c], mask[r][c + 1]))
        lower = r < h - 1 and all((
            mask[r][c], mask[r][c + 1],
            mask[r + 1][c], mask[r + 1][c + 1]))
        return upper or lower
    else:  # 竖直边 (r,c)-(r+1,c)：左环 (r,c-1)、右环 (r,c)
        r, c = min(r1, r2), c1
        left = c > 0 and all((
            mask[r][c - 1], mask[r][c],
            mask[r + 1][c - 1], mask[r + 1][c]))
        right = c < w - 1 and all((
            mask[r][c], mask[r][c + 1],
            mask[r + 1][c], mask[r + 1][c + 1]))
        return left or right


# --------------------------------------------------------------------------
# 测试
# --------------------------------------------------------------------------
class TestRamp(unittest.TestCase):
    def test_ramp_no_vortex_exact_recovery(self):
        h = w = 12
        ar, ac = 0.9, 2.5  # 2.5 > pi/2，列向频繁跨 pi 包裹（跨 pi 斜坡）
        truth = [[ar * r + ac * c for c in range(w)] for r in range(h)]
        wrapped = [[wrap_phase(v) for v in row] for row in truth]
        mask = all_true(h, w)

        self.assertEqual(ref_residues(wrapped, mask), [])
        res = unwrap_phase(wrapped, mask)

        self.assertEqual(res.residues, ())
        self.assertEqual(res.cuts, ())
        self.assertEqual(res.unmet_edges, ())
        self.assertEqual(len(res.regions), 1)
        self.assertEqual(res.regions[0].anchor, (0, 0))
        self.assertEqual(res.regions[0].baseline, 0)

        # 锚点基准为 0；展开差值精确还原真值梯度（口径 1e-9）。
        anchor_phi = res.unwrapped[0][0]
        for r in range(h):
            for c in range(w):
                got = res.unwrapped[r][c] - anchor_phi
                self.assertLess(abs(got - (ar * r + ac * c)), 1e-9)
                # 相对包裹值恰差整数个 2π（补充性质）。
                k = (res.unwrapped[r][c] - wrapped[r][c]) / TWO_PI
                self.assertLess(abs(k - round(k)), 1e-12)

    def test_ramp_with_offset_independent(self):
        h = w = 8
        g = [[wrap_phase(0.3 * r - 0.7 * c + 1.0) for c in range(w)]
             for r in range(h)]
        res = unwrap_phase(g, all_true(h, w))
        regions, bad = independent_flood_check(g, all_true(h, w), res.cuts)
        self.assertEqual(regions, 1)
        self.assertEqual(bad, [])


class TestVortex(unittest.TestCase):
    def test_single_vortex_reference_goes_to_border(self):
        # 解析单涡旋参考：独立环和算出恰一个残差，必须连外边界。
        g = analytic_vortex(8, 8, 0, 0)
        mask = all_true(8, 8)
        expected = ref_residues(g, mask)
        self.assertEqual(expected, [(0, 0, 1)])

        res = unwrap_phase(g, mask)
        self.assertEqual(
            [(x.row, x.col, x.charge) for x in res.residues], expected
        )
        self.assertGreaterEqual(len(res.cuts), 1)
        # 最短出口长度为 1（残差环就在顶角），北/西等距时方向序取北。
        self.assertEqual(len(res.cuts), 1)
        self.assertIn(
            phasecut.CutEdge(0, 0, 0, 1), set(res.cuts)
        )  # 首边即出图像顶边
        regions, bad = independent_flood_check(g, mask, res.cuts)
        self.assertEqual(bad, [])
        self.assertEqual(len(res.regions), regions)
        self.assertEqual(res.unmet_edges, ())

    def test_interior_dipole_pairs_with_shortest_path(self):
        # (4,4)+1 与 (4,6)-1，曼哈顿距离 2：直接配对、不触外边界。
        h = w = 9
        g = analytic_vortices(h, w, {(4, 4): +1, (4, 6): -1})
        mask = all_true(h, w)
        self.assertEqual(ref_residues(g, mask), [(4, 4, 1), (4, 6, -1)])
        self.assertEqual(independent_bfs_dist((4, 4), mask, h, w)[(4, 6)], 2)

        res = unwrap_phase(g, mask)
        cutset = set(res.cuts)
        border_touch = [
            e for e in res.cuts
            if min(e.row1, e.row2) == 0 or min(e.col1, e.col2) == 0
            or max(e.row1, e.row2) == h - 1 or max(e.col1, e.col2) == w - 1
        ]
        self.assertEqual(border_touch, [])
        self.assertEqual(len(res.cuts), 2)  # 恰为最短路径长度，无分叉
        # 东向路径跨过两条竖向像素边（独立 BFS 亦给出距离 2）。
        self.assertEqual(
            cutset,
            {phasecut.CutEdge(4, 5, 5, 5), phasecut.CutEdge(4, 6, 5, 6)},
        )

        def degree(cell):  # 残差格在切口图中的度
            r, c = cell
            incident = {
                phasecut.CutEdge(r, c, r, c + 1),
                phasecut.CutEdge(r, c + 1, r + 1, c + 1),
                phasecut.CutEdge(r + 1, c, r + 1, c + 1),
                phasecut.CutEdge(r, c, r + 1, c),
            }
            return len(incident & cutset)

        self.assertEqual(degree((4, 4)), 1)
        self.assertEqual(degree((4, 6)), 1)
        regions, bad = independent_flood_check(g, mask, res.cuts)
        self.assertEqual(bad, [])
        self.assertEqual(regions, len(res.regions))

    def test_equidistant_tie_picks_lexicographic_target(self):
        # (2,4)+ 最先处理；异号目标 (2,6)- 与 (4,4)- 都在距离 2，
        # 必须选字典序更小的 (2,6)；被剩下的 (4,4)- 向东最短出界（4 步）。
        h = w = 9
        g = analytic_vortices(h, w, {(2, 4): +1, (2, 6): -1, (4, 4): -1})
        mask = all_true(h, w)
        self.assertEqual(
            ref_residues(g, mask), [(2, 4, 1), (2, 6, -1), (4, 4, -1)]
        )
        d = independent_bfs_dist((2, 4), mask, h, w)
        self.assertEqual(d[(2, 6)], 2)
        self.assertEqual(d[(4, 4)], 2)

        res = unwrap_phase(g, mask)
        cutset = set(res.cuts)
        # 配对路径 (2,4)->(2,6)：两条竖边。
        self.assertIn(phasecut.CutEdge(2, 5, 3, 5), cutset)
        self.assertIn(phasecut.CutEdge(2, 6, 3, 6), cutset)
        # 绝不能配向 (4,4)（那条路径首步向南，会经过水平边 (3,4)-(3,5)）。
        self.assertNotIn(phasecut.CutEdge(3, 4, 3, 5), cutset)
        # 剩余 (4,4)- 东出右边界：col=5,6,7,8 四条竖边。
        for c in range(5, 9):
            self.assertIn(phasecut.CutEdge(4, c, 5, c), cutset)
        self.assertEqual(len(res.cuts), 6)
        regions, bad = independent_flood_check(g, mask, res.cuts)
        self.assertEqual(bad, [])
        self.assertEqual(regions, len(res.regions))


class TestMasks(unittest.TestCase):
    def test_balanced_vortices_on_isolated_island(self):
        # 无效海中的 5x5 孤岛（不触图像边界），岛内偶极岛内配对。
        h = w = 11
        mask = [[False] * w for _ in range(h)]
        for r in range(3, 8):
            for c in range(3, 8):
                mask[r][c] = True
        g = analytic_vortices(h, w, {(4, 4): +1, (4, 6): -1})
        self.assertEqual(ref_residues(g, mask), [(4, 4, 1), (4, 6, -1)])

        res = unwrap_phase(g, mask)
        self.assertEqual(len(res.cuts), 2)
        for e in res.cuts:  # 切口边两端有效、且全部落在岛内部
            self.assertTrue(mask[e.row1][e.col1] and mask[e.row2][e.col2])
            self.assertTrue(
                3 <= min(e.row1, e.row2) and max(e.row1, e.row2) <= 7
                and 3 <= min(e.col1, e.col2) and max(e.col1, e.col2) <= 7
            )
        regions, bad = independent_flood_check(g, mask, res.cuts)
        self.assertEqual(bad, [])
        self.assertEqual(regions, 1)
        self.assertEqual(res.unmet_edges, ())

    def test_unbalanced_island_is_proven_no_solution(self):
        # 唯一残差困在孤岛：完整 BFS 既无异号也无外边界出口 -> 证明无解。
        h = w = 11
        mask = [[False] * w for _ in range(h)]
        for r in range(4, 7):
            for c in range(4, 7):
                mask[r][c] = True
        g = analytic_vortices(h, w, {(4, 4): +1})
        self.assertEqual(ref_residues(g, mask), [(4, 4, 1)])
        with self.assertRaises(NoSolutionError):
            unwrap_phase(g, mask)

    def test_two_mask_components_two_regions(self):
        h = w = 10
        mask = [[False] * w for _ in range(h)]
        for r in range(0, 3):
            for c in range(0, 3):
                mask[r][c] = True
        for r in range(6, 9):
            for c in range(6, 9):
                mask[r][c] = True
        g = [[wrap_phase(0.2 * r + 0.1 * c) for c in range(w)]
             for r in range(h)]
        res = unwrap_phase(g, mask)
        self.assertEqual(res.residues, ())
        self.assertEqual(len(res.regions), 2)
        self.assertEqual({rg.anchor for rg in res.regions}, {(0, 0), (6, 6)})
        for rg in res.regions:
            self.assertEqual(rg.baseline, 0)
            self.assertEqual(len(rg.pixels), len(rg.offsets))
            self.assertEqual(rg.offsets[0], 0)  # 锚点整数偏移为 0

    def test_invalid_pixels_are_none_and_input_immutable(self):
        g = [[0.0, 1.0], [2.0, 3.0]]
        mask = [[True, False], [True, True]]
        snapshot = [row[:] for row in g]
        res = unwrap_phase(g, mask)
        self.assertIsNone(res.unwrapped[0][1])
        self.assertEqual(g, snapshot)  # 纯函数，输入不被修改

    def test_narrow_channel_routes_cut_along_corridor(self):
        # 哑铃形 mask：两个 4x4 房间由一条 2 行宽走廊连接（1 像素宽走廊
        # 不含 2x2 环格，对偶路径无法通过——这是对偶图的固有几何）。
        h = w = 9
        mask = [[False] * w for _ in range(h)]
        for r in range(0, 4):
            for c in range(0, 4):
                mask[r][c] = True
        for r in range(5, 9):
            for c in range(5, 9):
                mask[r][c] = True
        for r in (3, 4):
            for c in range(2, 7):
                mask[r][c] = True
        g = analytic_vortices(h, w, {(1, 1): +1, (6, 6): -1})
        self.assertEqual(ref_residues(g, mask), [(1, 1, 1), (6, 6, -1)])
        d = independent_bfs_dist((1, 1), mask, h, w)
        self.assertEqual(d[(6, 6)], 10)

        res = unwrap_phase(g, mask)
        # 切口数恰为独立 BFS 给出的最短距离。
        self.assertEqual(len(res.cuts), 10)
        # 路径确实穿过走廊（含走廊内的水平/竖直边）。
        corridor_edges = [
            e for e in res.cuts
            if e.row1 in (3, 4) and 2 <= min(e.col1, e.col2)
            and max(e.col1, e.col2) <= 6
        ]
        self.assertGreaterEqual(len(corridor_edges), 2)
        regions, bad = independent_flood_check(g, mask, res.cuts)
        self.assertEqual(bad, [])
        self.assertEqual(regions, len(res.regions))
        self.assertEqual(res.unmet_edges, ())


class TestCutsTopology(unittest.TestCase):
    def test_multi_residue_layout_full_independent_audit(self):
        # 复杂布局：两对可配残差 + 字典序配对链；独立拓扑全面复核。
        h = w = 12
        g = analytic_vortices(
            h, w,
            {(2, 2): +1, (2, 5): -1, (8, 8): -1, (8, 3): +1, (5, 10): +1},
        )
        mask = all_true(h, w)
        res = unwrap_phase(g, mask)
        self.assertEqual(
            [(x.row, x.col, x.charge) for x in res.residues],
            ref_residues(g, mask),
        )
        for e in res.cuts:  # 切口边两端有效、四邻接
            self.assertTrue(mask[e.row1][e.col1] and mask[e.row2][e.col2])
            self.assertEqual(abs(e.row1 - e.row2) + abs(e.col1 - e.col2), 1)
        valid = {(r, c) for r in range(h) for c in range(w) if mask[r][c]}
        comps = cut_components(res.cuts, valid)
        self.assertEqual(len(comps), len(res.regions))
        for (anchor, members), rg in zip(comps, res.regions):
            self.assertEqual(anchor, rg.anchor)
            self.assertEqual(members, list(rg.pixels))
        regions, bad = independent_flood_check(g, mask, res.cuts)
        self.assertEqual(bad, [])
        self.assertEqual(regions, len(res.regions))
        self.assertEqual(res.unmet_edges, ())
        # 字典序配对链中路径触及右边界列 col=11（独立校验的结构性事实）。
        self.assertTrue(any(max(e.col1, e.col2) == 11 for e in res.cuts))

    def test_every_residue_incident_to_cut(self):
        h = w = 10
        g = analytic_vortices(
            h, w, {(1, 1): -1, (1, 7): +1, (7, 2): +1, (7, 8): -1}
        )
        mask = all_true(h, w)
        res = unwrap_phase(g, mask)
        cutset = set(res.cuts)
        for x in res.residues:
            r, c = x.row, x.col
            incident = {
                phasecut.CutEdge(r, c, r, c + 1),
                phasecut.CutEdge(r, c + 1, r + 1, c + 1),
                phasecut.CutEdge(r + 1, c, r + 1, c + 1),
                phasecut.CutEdge(r, c, r + 1, c),
            }
            self.assertTrue(incident & cutset)
        self.assertEqual(res.unmet_edges, ())


class TestInputValidation(unittest.TestCase):
    def _g(self):
        return [[0.1, 0.2], [0.3, 0.4]]

    def _m(self):
        return [[True, True], [True, True]]

    def test_nan_rejected_with_location(self):
        g = self._g()
        g[1][0] = float("nan")
        with self.assertRaises(InvalidPhaseInputError) as cm:
            unwrap_phase(g, self._m())
        self.assertEqual(cm.exception.row, 1)
        self.assertEqual(cm.exception.col, 0)

    def test_infinity_rejected(self):
        g = self._g()
        g[0][1] = float("inf")
        with self.assertRaises(InvalidPhaseInputError) as cm:
            unwrap_phase(g, self._m())
        self.assertEqual((cm.exception.row, cm.exception.col), (0, 1))
        g[0][1] = float("-inf")
        with self.assertRaises(InvalidPhaseInputError):
            unwrap_phase(g, self._m())

    def test_out_of_range_rejected(self):
        with self.assertRaises(InvalidPhaseInputError) as cm:
            unwrap_phase([[PI, 0.0], [0.0, 0.0]], self._m())
        self.assertEqual((cm.exception.row, cm.exception.col), (0, 0))
        with self.assertRaises(InvalidPhaseInputError):
            unwrap_phase([[-PI - 1e-12, 0.0], [0.0, 0.0]], self._m())
        unwrap_phase([[-PI, 0.0], [0.0, 0.0]], self._m())  # -pi 合法

    def test_bool_phase_and_int_mask_rejected(self):
        with self.assertRaises(InvalidPhaseInputError):
            unwrap_phase([[True, 0.0], [0.0, 0.0]], self._m())
        with self.assertRaises(InvalidPhaseInputError) as cm:
            unwrap_phase(self._g(), [[1, 0], [1, 1]])
        self.assertEqual((cm.exception.row, cm.exception.col), (0, 0))

    def test_bool_search_budget_rejected(self):
        with self.assertRaises(InvalidPhaseInputError):
            unwrap_phase(self._g(), self._m(), search_budget=True)

    def test_shape_mismatch(self):
        with self.assertRaises(InvalidPhaseInputError):
            unwrap_phase([[0.0, 0.1], [0.2, 0.3]], [[True, True, True]] * 2)
        with self.assertRaises(InvalidPhaseInputError):
            unwrap_phase([[0.0, 0.1], [0.2, 0.3, 0.4]],
                         [[True, True], [True, True, True]])

    def test_size_limits_and_empty(self):
        with self.assertRaises(InvalidPhaseInputError):
            unwrap_phase([[0.0] * 32 for _ in range(33)],
                         [[True] * 32 for _ in range(33)])
        with self.assertRaises(InvalidPhaseInputError):
            unwrap_phase([[0.0] * 33 for _ in range(32)],
                         [[True] * 33 for _ in range(32)])
        unwrap_phase([[0.0] * 32 for _ in range(32)],
                     [[True] * 32 for _ in range(32)])  # 32x32 上限可用
        with self.assertRaises(InvalidPhaseInputError):
            unwrap_phase([], [])
        with self.assertRaises(InvalidPhaseInputError):
            unwrap_phase([[]], [[]])

    def test_bad_types(self):
        with self.assertRaises(InvalidPhaseInputError):
            unwrap_phase([[complex(1, 1), 0.0], [0.0, 0.0]], self._m())
        with self.assertRaises(InvalidPhaseInputError):
            unwrap_phase("0102", self._m())
        with self.assertRaises(InvalidPhaseInputError):
            unwrap_phase(self._g(), "1111")
        with self.assertRaises(InvalidPhaseInputError):
            unwrap_phase(self._g(), self._m(), search_budget=0)
        with self.assertRaises(InvalidPhaseInputError):
            unwrap_phase(self._g(), self._m(), search_budget=1.5)


class TestBudget(unittest.TestCase):
    def test_budget_exhausted_distinct_from_no_solution(self):
        # 9x9 中心解析涡旋：一个 +1 与其负 x 射线上的一个 -1（独立核验）。
        g = analytic_vortex(9, 9, 4, 4)
        mask = all_true(9, 9)
        # 奇点恰在像素角点时，补偿电荷沿射线从图像左边界逸出，界内仅 +1。
        self.assertEqual(ref_residues(g, mask), [(4, 4, 1)])
        # 预算 1：起点之外一步都完不成——“算不完”，不是无解。
        with self.assertRaises(SearchBudgetExhausted):
            unwrap_phase(g, mask, search_budget=1)
        # 同输入给足预算即成功（两种失败语义严格分开）。
        res = unwrap_phase(g, mask, search_budget=1_000_000)
        self.assertEqual(len(res.residues), 1)
        self.assertEqual(res.unmet_edges, ())


class TestDeterminismAndRoundTrip(unittest.TestCase):
    def test_deterministic_output(self):
        g = analytic_vortex(10, 10, 3, 3)
        mask = all_true(10, 10)
        r1 = unwrap_phase(g, mask)
        r2 = unwrap_phase([row[:] for row in g], [row[:] for row in mask])
        self.assertEqual(r1, r2)

    def test_wrap_round_trip_supplementary(self):
        # 补充性质（不作为任何期望值来源）：wrap 后与原值差整数个 2π。
        for x in (-5.0, -PI, 0.0, 1.23456, PI - 1e-12, 40.0):
            y = wrap_phase(x)
            self.assertTrue(-PI <= y < PI)
            k = (x - y) / TWO_PI
            self.assertLess(abs(k - round(k)), 1e-12)


class TestDegenerateGrids(unittest.TestCase):
    def test_single_row_and_single_column_have_no_plaquettes(self):
        # 无 2x2 环 => 无残差、无切口；有效像素即便不四连通也各自成区。
        g1 = [[wrap_phase(0.1 * c) for c in range(5)]]
        m1 = [[True, False, True, True, False]]
        r1 = unwrap_phase(g1, m1)
        self.assertEqual(r1.residues, ())
        self.assertEqual(r1.cuts, ())
        self.assertEqual(len(r1.regions), 2)  # col0 与 col2-3 两段
        self.assertIsNone(r1.unwrapped[0][1])
        self.assertEqual(r1.unwrapped[0][2], g1[0][2])

        g2 = [[wrap_phase(0.2 * r)] for r in range(4)]
        m2 = [[True], [True], [False], [True]]
        r2 = unwrap_phase(g2, m2)
        self.assertEqual(r2.residues, ())
        self.assertEqual(len(r2.regions), 2)

    def test_single_pixel(self):
        r = unwrap_phase([[0.3]], [[True]])
        self.assertEqual(r.regions[0].anchor, (0, 0))
        self.assertEqual(r.unwrapped, ((0.3,),))
        r0 = unwrap_phase([[0.3]], [[False]])
        self.assertEqual(r0.regions, ())
        self.assertEqual(r0.unwrapped, ((None,),))


class TestFuzzGuarantees(unittest.TestCase):
    """固定种子随机网格上的强保证（不是期望值来源，是性质检验）。"""

    def test_random_grands_always_consistent_or_proven_trapped(self):
        rng = random.Random(20260922)
        checked = trapped = 0
        for _ in range(200):
            h = rng.randint(2, 12)
            w = rng.randint(2, 12)
            # 以较高密度取有效像素，保证出现过各种环格与孤立结构。
            p = rng.choice((0.35, 0.55, 0.75, 1.0))
            mask = [[rng.random() < p for _ in range(w)] for _ in range(h)]
            if not any(any(row) for row in mask):
                continue
            g = [
                [wrap_phase(rng.uniform(-PI + 1e-6, PI - 1e-6))
                 for _ in range(w)]
                for _ in range(h)
            ]
            try:
                res = unwrap_phase(g, mask)
            except NoSolutionError:
                trapped += 1
                # 独立充要证书：至少存在一个“闭合（无任何一步可跨出图像
                # 外边界）且内部电荷和非零”的对偶连通分量。
                # 闭合分量里异号电荷无论怎么配对都抵消不完，残余又出不去
                # ——这才是被独立证明的无解，与搜索预算无关。
                residues = ref_residues(g, mask)
                self.assertTrue(residues)
                seen = set()
                impossible = False
                for (r, c, _q) in residues:
                    if (r, c) in seen:
                        continue
                    dist = independent_bfs_dist((r, c), mask, h, w)
                    seen |= set(dist)
                    total = sum(
                        qq for (R, C, qq) in residues if (R, C) in dist
                    )
                    can_exit = any(
                        (mask[R][C] and mask[R][C + 1] and R == 0)
                        or (mask[R][C + 1] and mask[R + 1][C + 1]
                            and C == w - 2)
                        or (mask[R + 1][C] and mask[R + 1][C + 1]
                            and R == h - 2)
                        or (mask[R][C] and mask[R + 1][C] and C == 0)
                        for (R, C) in dist
                    )
                    if total != 0 and not can_exit:
                        impossible = True
                self.assertTrue(impossible)
                continue
            checked += 1
            # 合法包裹数据的结构性保证：
            # 1) 独立洪泛复核：删切口边后区内一致；核心的矛盾边集合必须
            #    与独立检查器完全一致（不得掩盖、不得虚报）。
            regions, bad = independent_flood_check(g, mask, res.cuts)
            self.assertEqual(regions, len(res.regions))
            self.assertEqual(
                sorted((e.row1, e.col1, e.row2, e.col2)
                       for e in res.unmet_edges),
                sorted((a, b, cc, d) for (a, b, cc, d, _v) in bad),
            )
            for (a, b, cc, d, v) in bad:
                # 未满足残差必须是整数个 2π 的真实卷绕矛盾，而非浮点噪声。
                self.assertLess(abs(v / TWO_PI - round(v / TWO_PI)), 1e-9)
            # 2) 残差与独立四点环和结果完全一致；
            self.assertEqual(
                [(x.row, x.col, x.charge) for x in res.residues],
                ref_residues(g, mask),
            )
            # 3) 每个残差格至少与一条切口相邻（被配对或被引到边界/死角）。
            cutset = set(res.cuts)
            for x in res.residues:
                r, c = x.row, x.col
                incident = {
                    phasecut.CutEdge(r, c, r, c + 1),
                    phasecut.CutEdge(r, c + 1, r + 1, c + 1),
                    phasecut.CutEdge(r + 1, c, r + 1, c + 1),
                    phasecut.CutEdge(r, c, r + 1, c),
                }
                self.assertTrue(incident & cutset)
        self.assertGreater(checked, 150)  # 确保绝大多数用例走到了成功分支
        self.assertGreaterEqual(trapped, 1)  # 也覆盖到了证明无解分支


class TestHoleWinding(unittest.TestCase):
    """卷绕绕 mask 洞而非落在完整 2x2 环上：必须进 unmet_edges，不得掩盖。"""

    def test_vortex_centered_on_removed_pixel_is_reported_not_hidden(self):
        h = w = 9
        mask = [[True] * w for _ in range(h)]
        mask[4][4] = False  # 1 像素洞，恰好盖住涡核像素
        g = [[wrap_phase(math.atan2(r - 4.5, c - 4.5)) for c in range(w)]
             for r in range(h)]

        # 没有任何四点全有效的环携带电荷。
        self.assertEqual(ref_residues(g, mask), [])
        res = unwrap_phase(g, mask)
        self.assertEqual(res.residues, ())
        # 无残差则无切口可布；绕洞卷绕以非空 unmet_edges 如实呈现。
        self.assertEqual(res.cuts, ())
        self.assertGreaterEqual(len(res.unmet_edges), 1)
        for e in res.unmet_edges:
            self.assertLess(
                abs(e.residual / TWO_PI - round(e.residual / TWO_PI)), 1e-9
            )
            self.assertGreaterEqual(abs(e.residual), PI)
        # 独立检查器在同一组边上得出同样的矛盾（非平滑、非偶然）。
        _regions, bad = independent_flood_check(g, mask, res.cuts)
        self.assertEqual(
            sorted((e.row1, e.col1, e.row2, e.col2) for e in res.unmet_edges),
            sorted((a, b, c, d) for (a, b, c, d, _v) in bad),
        )

    def test_uncuttable_edge_contradiction_is_reported(self):
        # 复现固定种子 fuzz 案例 9：矛盾边两侧相邻环都无效（不可切口边）。
        rng = random.Random(20260922)
        target = None
        for _ in range(200):
            h = rng.randint(2, 12)
            w = rng.randint(2, 12)
            p = rng.choice((0.35, 0.55, 0.75, 1.0))
            mask = [[rng.random() < p for _ in range(w)] for _ in range(h)]
            if not any(any(row) for row in mask):
                continue
            g = [[wrap_phase(rng.uniform(-PI + 1e-6, PI - 1e-6))
                  for _ in range(w)] for _ in range(h)]
            try:
                res = unwrap_phase(g, mask)
            except NoSolutionError:
                continue
            if res.unmet_edges:
                target = (g, mask, res)
                break
        self.assertIsNotNone(target)
        g, mask, res = target
        # 每条 unmet 边两侧都无完整 2x2 环——几何上任何对偶路径都无法
        # 跨过它，因此它不可能被表示为切口，只能如实留在 unmet_edges。
        for e in res.unmet_edges:
            self.assertFalse(edge_is_cuttable(
                mask, e.row1, e.col1, e.row2, e.col2
            ))
        _regions, bad = independent_flood_check(g, mask, res.cuts)
        self.assertEqual(
            sorted((e.row1, e.col1, e.row2, e.col2) for e in res.unmet_edges),
            sorted((a, b, c, d) for (a, b, c, d, _v) in bad),
        )


    def test_random_grands_at_size_bound_32(self):
        # 接口上限 32x32：少量整上限随机网格，确认可计算且结构性质成立。
        rng = random.Random(99173)
        for _ in range(6):
            h = w = 32
            mask = [[rng.random() < 0.7 for _ in range(w)] for _ in range(h)]
            g = [[wrap_phase(rng.uniform(-PI + 1e-6, PI - 1e-6))
                  for _ in range(w)] for _ in range(h)]
            try:
                res = unwrap_phase(g, mask)
            except NoSolutionError:
                continue
            self.assertEqual(
                [(x.row, x.col, x.charge) for x in res.residues],
                ref_residues(g, mask),
            )
            regions, bad = independent_flood_check(g, mask, res.cuts)
            self.assertEqual(regions, len(res.regions))
            self.assertEqual(
                sorted((e.row1, e.col1, e.row2, e.col2)
                       for e in res.unmet_edges),
                sorted((a, b, c, d) for (a, b, c, d, _v) in bad),
            )


class TestAlignRegions(unittest.TestCase):
    """区域参考点对齐：整体 2π 偏移、冲突必报、无参考区域保持原样。"""

    def two_islands(self):
        h = w = 10
        mask = [[False] * w for _ in range(h)]
        for r in range(0, 3):
            for c in range(0, 3):
                mask[r][c] = True
        for r in range(6, 9):
            for c in range(6, 9):
                mask[r][c] = True
        g = [[wrap_phase(0.2 * r + 0.1 * c) for c in range(w)]
             for r in range(h)]
        return g, mask

    def test_align_one_region_leaves_other_untouched(self):
        g, mask = self.two_islands()
        res = unwrap_phase(g, mask)
        self.assertEqual(len(res.regions), 2)
        # 独立期望：第二区 (6,6) 的局部展开值 + 3·2π 作为目标。
        target = res.unwrapped[6][6] + 3 * TWO_PI
        aligned = align_regions(res, [(6, 6, target)])
        self.assertEqual(aligned.regions[1].baseline, 3)
        self.assertTrue(all(k == 3 for k in aligned.regions[1].offsets))
        self.assertAlmostEqual(aligned.unwrapped[6][6], target, places=12)
        # 无参考的第一区保持未对齐，局部结果原样保留。
        self.assertEqual(aligned.regions[0], res.regions[0])
        for r, c in res.regions[0].pixels:
            self.assertEqual(aligned.unwrapped[r][c], res.unwrapped[r][c])
        # 切口、残差、未满足边报告不因对齐而消失或改变。
        self.assertEqual(aligned.cuts, res.cuts)
        self.assertEqual(aligned.residues, res.residues)
        self.assertEqual(aligned.unmet_edges, res.unmet_edges)
        # 入参不被修改（纯函数）。
        self.assertEqual(res.regions[1].baseline, 0)

    def test_agreeing_references_same_region(self):
        g, mask = self.two_islands()
        res = unwrap_phase(g, mask)
        t1 = res.unwrapped[6][6] - 2 * TWO_PI
        t2 = res.unwrapped[8][8] - 2 * TWO_PI
        aligned = align_regions(
            res, [PhaseReference(6, 6, t1), (8, 8, t2)]
        )
        self.assertEqual(aligned.regions[1].baseline, -2)
        self.assertAlmostEqual(aligned.unwrapped[8][8], t2, places=12)

    def test_conflicting_references_reported_not_smoothed(self):
        g, mask = self.two_islands()
        res = unwrap_phase(g, mask)
        t1 = res.unwrapped[6][6] + 1 * TWO_PI
        t2 = res.unwrapped[8][8] + 2 * TWO_PI  # 要求不同的整数偏移
        with self.assertRaises(AlignmentConflictError) as ctx:
            align_regions(res, [(6, 6, t1), (8, 8, t2)])
        exc = ctx.exception
        self.assertEqual(exc.anchor, (6, 6))
        self.assertEqual(len(exc.details), 2)
        self.assertEqual(sorted(d[3] for d in exc.details), [1, 2])
        self.assertIsInstance(exc, phasecut.PhaseCutError)

    def test_target_inconsistent_with_local_phase_conflicts(self):
        g, mask = self.two_islands()
        res = unwrap_phase(g, mask)
        # 目标与任何整数个 2π 平移都对不上（偏差 1.0 rad，超口径）。
        bad_target = res.unwrapped[6][6] + 1.0
        with self.assertRaises(AlignmentConflictError):
            align_regions(res, [(6, 6, bad_target)])

    def test_reference_on_invalid_pixel_rejected(self):
        g, mask = self.two_islands()
        res = unwrap_phase(g, mask)
        with self.assertRaises(InvalidPhaseInputError) as ctx:
            align_regions(res, [(4, 4, 0.0)])  # mask 无效像素
        self.assertEqual((ctx.exception.row, ctx.exception.col), (4, 4))
        with self.assertRaises(InvalidPhaseInputError):
            align_regions(res, [(10, 0, 0.0)])  # 越界

    def test_bad_reference_types_rejected(self):
        g, mask = self.two_islands()
        res = unwrap_phase(g, mask)
        for bad in (
            [(6, 6)],                    # 元素数不足
            [(6, 6, float("nan"))],      # NaN 目标
            [(6, 6, float("inf"))],      # Infinity 目标
            [(True, 6, 0.0)],            # bool 行坐标
            [(6, 6, "0.0")],             # 字符串目标
        ):
            with self.assertRaises(InvalidPhaseInputError):
                align_regions(res, bad)

    def test_no_references_returns_equal_result(self):
        g, mask = self.two_islands()
        res = unwrap_phase(g, mask)
        aligned = align_regions(res, [])
        self.assertEqual(aligned, res)

    def test_alignment_preserves_cuts_and_unmet_edges(self):
        # 带真实 unmet_edges 的场景：1 像素洞上的绕洞卷绕。
        h = w = 9
        mask = [[True] * w for _ in range(h)]
        mask[4][4] = False
        g = [[wrap_phase(math.atan2(r - 4.5, c - 4.5)) for c in range(w)]
             for r in range(h)]
        res = unwrap_phase(g, mask)
        self.assertGreaterEqual(len(res.unmet_edges), 1)
        target = res.unwrapped[0][0] + 2 * TWO_PI
        aligned = align_regions(res, [(0, 0, target)])
        self.assertEqual(aligned.unmet_edges, res.unmet_edges)
        self.assertEqual(aligned.cuts, res.cuts)
        self.assertEqual(aligned.residues, res.residues)
        self.assertEqual(aligned.regions[0].baseline, 2)


class TestContradictionContract(unittest.TestCase):
    def test_contradiction_attributes(self):
        # 对真正落在 [-pi,pi) 的数据，四点包裹差之和必为 2π 整数倍
        # （望远镜相消恒等式），因此该异常在合法输入下不会触发；
        # 这里只固定其对外契约（坐标与实测环和可定位）。
        from phasecut import UnwrappingContradictionError
        exc = UnwrappingContradictionError(2, 3, 0.7)
        self.assertEqual((exc.row, exc.col, exc.loop_sum), (2, 3, 0.7))
        self.assertIsInstance(exc, phasecut.PhaseCutError)


if __name__ == "__main__":
    unittest.main()
