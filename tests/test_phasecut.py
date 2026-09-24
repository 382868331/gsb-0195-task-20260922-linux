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
    InvalidPhaseInputError,
    NoSolutionError,
    SearchBudgetExhausted,
    wrap_phase,
    unwrap_phase,
    align_frame_pair,
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


class TestContradictionContract(unittest.TestCase):
    def test_contradiction_attributes(self):
        # 对真正落在 [-pi,pi) 的数据，四点包裹差之和必为 2π 整数倍
        # （望远镜相消恒等式），因此该异常在合法输入下不会触发；
        # 这里只固定其对外契约（坐标与实测环和可定位）。
        from phasecut import UnwrappingContradictionError
        exc = UnwrappingContradictionError(2, 3, 0.7)
        self.assertEqual((exc.row, exc.col, exc.loop_sum), (2, 3, 0.7))
        self.assertIsInstance(exc, phasecut.PhaseCutError)


# --------------------------------------------------------------------------
# 相邻两帧时间对齐（本轮新增能力）
# --------------------------------------------------------------------------
def ref_flood_unwrap(g, mask):
    """独立单源/多源洪泛展开（各区锚点 k=0），不布切口。

    仅在 ref_residues 为空、且不存在绕 mask 洞卷绕的场上使用
    （完整 mask 的斜坡场、或全场真值落在 (-pi,pi) 内时严格成立）。
    """
    h, w = len(mask), len(mask[0])
    valid = {(r, c) for r in range(h) for c in range(w) if mask[r][c]}
    phi = {}
    for start in sorted(valid):
        if start in phi:
            continue
        phi[start] = g[start[0]][start[1]]
        stack = [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
                nxt = (r + dr, c + dc)
                if nxt not in valid or nxt in phi:
                    continue
                d = ref_wrap(g[nxt[0]][nxt[1]] - g[r][c])
                expect = phi[(r, c)] + d
                k = int(math.floor(abs((expect - g[nxt[0]][nxt[1]]) / TWO_PI) + 0.5))
                k = k if expect - g[nxt[0]][nxt[1]] >= 0 else -k
                phi[nxt] = g[nxt[0]][nxt[1]] + k * TWO_PI
                stack.append(nxt)
    return phi


def ref_align_enumerate(phi1, phi2, m1, m2):
    """独立参考：在共同有效像素上全范围枚举整数 K，取 SSE 最小，并列取小者。

    枚举范围对测试用的小网格取固定的 [-50, 50]（远超任何可达偏移），
    与核心的范围自适应逻辑完全独立。
    """
    h, w = len(m1), len(m1[0])
    common = tuple(
        (r, c) for r in range(h) for c in range(w) if m1[r][c] and m2[r][c]
    )
    diffs = [phi1[(r, c)] - phi2[(r, c)] for (r, c) in common]
    best_k, best_sse = None, None
    for k in range(-50, 51):
        shift = k * TWO_PI
        v = sum((d - shift) ** 2 for d in diffs)
        if best_sse is None or v < best_sse - 1e-9:
            best_k, best_sse = k, v
    return common, best_k, best_sse


class TestFrameAlignmentHandComputed(unittest.TestCase):
    """手算例子：期望值直接写死，不经任何实现推导。"""

    def test_single_common_pixel_positive_offset(self):
        # 1x2 单行（无 2x2 环、无切口），仅 (0,1) 共同有效。
        # φ1=2.5，φ2=-1.5 => d=4.0。
        # K=0: SSE=16；K=1: (4-2π)²≈5.21 => 最优 K=1。
        g1 = [[2.5, 2.5]]
        g2 = [[0.0, -1.5]]  # (0,0) 在帧2无效，值随便给但需合法
        m1 = [[True, True]]
        m2 = [[False, True]]
        res = align_frame_pair(g1, m1, g2, m2)
        self.assertTrue(res.aligned)
        self.assertEqual(res.offset_cycles, 1)
        self.assertEqual(res.common_pixels, ((0, 1),))
        self.assertAlmostEqual(res.sse, (4.0 - TWO_PI) ** 2, places=12)
        # 对齐后帧2共同像素：-1.5 + 2π，与帧2自己的无效位保留 None。
        self.assertIsNone(res.aligned_frame2[0][0])
        self.assertAlmostEqual(
            res.aligned_frame2[0][1], -1.5 + TWO_PI, places=12
        )
        # 帧1独有像素不受影响；两帧的空间展开结果都完整保留。
        self.assertEqual(res.frame1.unwrapped[0][0], 2.5)
        self.assertIsNone(res.frame2.unwrapped[0][0])

    def test_tie_at_pi_picks_smaller_integer(self):
        # 恰好在 ±π 分界并列：唯一共同像素 d=-π（g1=-π 合法，g2=0）。
        # SSE(0)=π²，SSE(-1)=(-π+2π)²=π² 精确并列 => 取较小整数 -1。
        res = align_frame_pair([[-PI]], [[True]], [[0.0]], [[True]])
        self.assertTrue(res.aligned)
        self.assertEqual(res.offset_cycles, -1)
        self.assertAlmostEqual(res.sse, PI ** 2, places=12)
        self.assertAlmostEqual(res.aligned_frame2[0][0], -TWO_PI, places=12)

    def test_multi_pixel_cross_2pi_ramp_with_local_missing(self):
        # 1x6 斜坡，帧2缺前两个测点：φ1=2.5c（多次跨 2π 包裹），
        # 帧2真值 = 2.5c - 4.0、锚点在 c=2（φ2(2)=1.0）。
        # 四个共同像素 d 恒为 4.0 => K=1，SSE=4·(4-2π)²，手算可核。
        g1 = [[wrap_phase(2.5 * c) for c in range(6)]]
        g2 = [[wrap_phase(2.5 * c - 4.0) for c in range(6)]]
        m1 = [[True] * 6]
        m2 = [[False, False] + [True] * 4]
        res = align_frame_pair(g1, m1, g2, m2)
        self.assertTrue(res.aligned)
        self.assertEqual(res.offset_cycles, 1)
        self.assertEqual(
            res.common_pixels, tuple((0, c) for c in range(2, 6))
        )
        self.assertAlmostEqual(res.sse, 4.0 * (4.0 - TWO_PI) ** 2, places=9)
        # 对齐后每个共同像素的残差恒为 4-2π ≈ -2.2832。
        for c in range(2, 6):
            self.assertLess(
                abs(res.frame1.unwrapped[0][c]
                    - res.aligned_frame2[0][c] - (4.0 - TWO_PI)),
                1e-9,
            )
        # 两帧残差、切口皆为空（单行无环）。
        self.assertEqual(res.frame1.residues, ())
        self.assertEqual(res.frame2.residues, ())
        self.assertEqual(res.frame1.cuts, ())

    def test_frame2_only_pixel_still_gets_global_shift(self):
        # 帧1在 (1,1) 有局部缺测，帧2有效：全局 K 仍作用于帧2全部有效像素。
        g1 = [[wrap_phase(0.3 * r + 0.2 * c) for c in range(3)]
              for r in range(3)]
        g2 = [[wrap_phase(0.3 * r + 0.2 * c + 4.0) for c in range(3)]
              for r in range(3)]
        m1 = [[True] * 3 for _ in range(3)]
        m1[1][1] = False
        m2 = [[True] * 3 for _ in range(3)]
        res = align_frame_pair(g1, m1, g2, m2)
        self.assertTrue(res.aligned)
        self.assertNotIn((1, 1), res.common_pixels)
        k = res.offset_cycles
        # 帧2独有像素也被整体平移（不是 None，也不是未平移值）。
        self.assertAlmostEqual(
            res.aligned_frame2[1][1],
            res.frame2.unwrapped[1][1] + k * TWO_PI,
            places=12,
        )


class TestFrameAlignmentRamps(unittest.TestCase):
    """跨 2π 变化的斜坡帧对：独立洪泛 + 独立枚举核对。"""

    def test_steep_ramps_independent_reference(self):
        h = w = 10
        t1 = [[0.9 * r + 2.5 * c for c in range(w)] for r in range(h)]
        g1 = [[wrap_phase(v) for v in row] for row in t1]
        m1 = all_true(h, w)
        phi1 = ref_flood_unwrap(g1, m1)
        for delta in (0.4, -2.0, 5.0, -5.0):
            with self.subTest(delta=delta):
                g2 = [[wrap_phase(v + delta) for v in row] for row in t1]
                m2 = all_true(h, w)
                self.assertEqual(ref_residues(g1, m1), [])
                self.assertEqual(ref_residues(g2, m2), [])
                phi2 = ref_flood_unwrap(g2, m2)
                common, exp_k, exp_sse = ref_align_enumerate(
                    phi1, phi2, m1, m2
                )
                res = align_frame_pair(g1, m1, g2, m2)
                self.assertTrue(res.aligned)
                self.assertEqual(res.offset_cycles, exp_k)
                self.assertEqual(res.common_pixels, common)
                self.assertAlmostEqual(res.sse, exp_sse, places=8)
                # 手算关系：同锚点同梯度 => d 恒为 -wrap(delta)，
                # 故 |wrap(delta)|<π 时 K 必为 0。
                self.assertEqual(exp_k, 0)
                self.assertAlmostEqual(
                    exp_sse, h * w * ref_wrap(delta) ** 2, places=8
                )

    def test_differing_anchor_enables_nonzero_k(self):
        # 完整 2D 斜坡 + 帧2锚点像素缺测：独立参考给出非零 K。
        h, w = 3, 6
        t1 = [[2.5 * c for c in range(w)] for _ in range(h)]
        t2 = [[2.5 * c - 4.0 for c in range(w)] for _ in range(h)]
        g1 = [[wrap_phase(v) for v in row] for row in t1]
        g2 = [[wrap_phase(v) for v in row] for row in t2]
        m1 = all_true(h, w)
        m2 = [[True] * w for _ in range(h)]
        m2[0][0] = False
        phi1 = ref_flood_unwrap(g1, m1)
        phi2 = ref_flood_unwrap(g2, m2)
        _common, exp_k, exp_sse = ref_align_enumerate(phi1, phi2, m1, m2)
        self.assertEqual(exp_k, 1)  # 独立参考自身先确认是非平凡解
        res = align_frame_pair(g1, m1, g2, m2)
        self.assertEqual(res.offset_cycles, 1)
        self.assertAlmostEqual(res.sse, exp_sse, places=8)

    def test_same_wrapped_field_zero_shift_is_perfect(self):
        # 帧2真值恰好多整数个 2π：包裹输入完全相同 => K=0、SSE=0。
        h = w = 7
        t = [[0.9 * r + 2.5 * c for c in range(w)] for r in range(h)]
        g1 = [[wrap_phase(v) for v in row] for row in t]
        g2 = [[wrap_phase(v + 3 * TWO_PI) for v in row] for row in t]
        res = align_frame_pair(g1, all_true(h, w), g2, all_true(h, w))
        self.assertEqual(res.offset_cycles, 0)
        # 同包裹场展开差只有 ~1e-14 的浮点散布，SSE 在 1e-27 量级。
        self.assertLess(res.sse, 1e-18)


class TestFrameAlignmentMasksAndFailures(unittest.TestCase):
    def test_no_common_pixels_is_unaligned_and_preserves_masks(self):
        # 两帧有效像素完全不相交：返回未对齐状态，无异常。
        h = w = 4
        m1 = [[False] * w for _ in range(h)]
        m2 = [[False] * w for _ in range(h)]
        m1[0][0] = True
        m2[3][3] = True
        g = [[0.1 * c for c in range(w)] for _ in range(h)]
        res = align_frame_pair(g, m1, g, m2)
        self.assertFalse(res.aligned)
        self.assertIsNone(res.offset_cycles)
        self.assertIsNone(res.sse)
        self.assertEqual(res.common_pixels, ())
        # 两帧展开结果仍完整保留；aligned_frame2 退化为帧2展开。
        self.assertEqual(res.aligned_frame2, res.frame2.unwrapped)
        self.assertIsNone(res.aligned_frame2[0][0])
        self.assertEqual(res.aligned_frame2[3][3], g[3][3])
        self.assertEqual(res.frame1.regions[0].anchor, (0, 0))
        self.assertEqual(res.frame2.regions[0].anchor, (3, 3))

    def test_both_frames_empty_mask(self):
        g = [[0.0] * 3 for _ in range(3)]
        m = [[False] * 3 for _ in range(3)]
        res = align_frame_pair(g, m, g, m)
        self.assertFalse(res.aligned)
        self.assertEqual(res.frame1.regions, ())
        self.assertEqual(res.common_pixels, ())

    def test_shape_mismatch_rejected(self):
        g = [[0.0] * 3 for _ in range(3)]
        m = [[True] * 3 for _ in range(3)]
        g2 = [[0.0] * 2 for _ in range(3)]
        m2 = [[True] * 2 for _ in range(3)]
        with self.assertRaises(InvalidPhaseInputError):
            align_frame_pair(g, m, g2, m2)

    def test_invalid_value_in_frame2_rejected_and_inputs_untouched(self):
        g1 = [[0.1, 0.2], [0.3, 0.4]]
        g2 = [[0.1, float("nan")], [0.3, 0.4]]
        m = [[True, True], [True, True]]
        snap1 = [row[:] for row in g1]
        with self.assertRaises(InvalidPhaseInputError) as cm:
            align_frame_pair(g1, m, g2, m)
        self.assertEqual((cm.exception.row, cm.exception.col), (0, 1))
        self.assertEqual(g1, snap1)
        # NaN != NaN，逐元素核对 g2 除故障点外未被修改。
        self.assertTrue(math.isnan(g2[0][1]))
        self.assertEqual(g2[0][0], 0.1)
        self.assertEqual(g2[1], [0.3, 0.4])

    def test_bool_mask_in_frame1_rejected(self):
        with self.assertRaises(InvalidPhaseInputError):
            align_frame_pair(
                [[0.0, 0.0]], [[1, 1]], [[0.0, 0.0]], [[True, True]]
            )

    def test_bool_budget_rejected(self):
        g = [[0.0]]
        m = [[True]]
        with self.assertRaises(InvalidPhaseInputError):
            align_frame_pair(g, m, g, m, search_budget=False)

    def test_frame2_no_solution_propagates(self):
        # 帧2是孤岛单涡旋：空间展开已证明无解，对齐直接传播、不返回半结果。
        h = w = 11
        island = [[False] * w for _ in range(h)]
        for r in range(4, 7):
            for c in range(4, 7):
                island[r][c] = True
        g = analytic_vortices(h, w, {(4, 4): 1})
        full = all_true(h, w)
        with self.assertRaises(NoSolutionError):
            align_frame_pair(g, full, g, island)

    def test_frame1_budget_exhausted_propagates(self):
        g = analytic_vortex(9, 9, 4, 4)
        m = all_true(9, 9)
        with self.assertRaises(SearchBudgetExhausted):
            align_frame_pair(g, m, g, m, search_budget=1)


class TestFrameAlignmentFuzz(unittest.TestCase):
    """固定种子随机 mask 对：真值全程落在 (-pi,pi)，空间展开恒等于真值，
    于是 K/SSE 可由解析关系直接独立核对。"""

    def test_random_mask_pairs_analytic_and_deterministic(self):
        rng = random.Random(20260924)
        h = w = 7
        shifts = (-1.7, -0.6, 0.0, 0.9, 2.0)  # 均使帧2真值留在 (-pi,pi)
        n_aligned = n_unaligned = 0
        # 首个用例固定为掩码完全不相交（确定性覆盖未对齐分支），
        # 其余 59 个用随机掩码对。
        forced_disjoint = (
            [[(r + c) % 2 == 0 for c in range(w)] for r in range(h)],
            [[(r + c) % 2 == 1 for c in range(w)] for r in range(h)],
        )
        random_pairs = []
        for _ in range(59):
            # 两帧密度独立选取；低密度时也会随机出现无共同像素的帧对。
            p1 = rng.choice((0.2, 0.5, 0.8))
            p2 = rng.choice((0.2, 0.5, 0.8))
            random_pairs.append((
                [[rng.random() < p1 for _ in range(w)] for _ in range(h)],
                [[rng.random() < p2 for _ in range(w)] for _ in range(h)],
            ))
        for m1, m2 in [forced_disjoint] + random_pairs:
            s = rng.choice(shifts)
            t1 = [[0.2 * r + 0.1 * c - 1.2 for c in range(w)]
                  for r in range(h)]
            t2 = [[v + s for v in row] for row in t1]
            g1 = [[wrap_phase(v) for v in row] for row in t1]
            g2 = [[wrap_phase(v) for v in row] for row in t2]
            res = align_frame_pair(g1, m1, g2, m2)
            common = tuple(
                (r, c) for r in range(h) for c in range(w)
                if m1[r][c] and m2[r][c]
            )
            self.assertEqual(res.common_pixels, common)
            if not common:
                n_unaligned += 1
                self.assertFalse(res.aligned)
                self.assertIsNone(res.offset_cycles)
                self.assertEqual(res.aligned_frame2, res.frame2.unwrapped)
                continue
            n_aligned += 1
            self.assertTrue(res.aligned)
            # 真值在包裹区间内 => 展开即真值，d=-s，|s|<π => K=0。
            self.assertEqual(res.offset_cycles, 0)
            self.assertAlmostEqual(res.sse, len(common) * s * s, places=9)
            for (r, c) in common:
                self.assertAlmostEqual(
                    res.aligned_frame2[r][c] - res.frame1.unwrapped[r][c],
                    s,
                    places=9,
                )
                self.assertIsNotNone(res.aligned_frame2[r][c])
            # 确定性：同输入两次调用完全相等。
            res2 = align_frame_pair(
                [row[:] for row in g1], [row[:] for row in m1],
                [row[:] for row in g2], [row[:] for row in m2],
            )
            self.assertEqual(res, res2)
        self.assertGreater(n_aligned, 40)
        self.assertGreaterEqual(n_unaligned, 1)


if __name__ == "__main__":
    unittest.main()
