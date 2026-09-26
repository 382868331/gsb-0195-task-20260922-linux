"""phasecut — 二维包裹相位的残差切口（branch cut）展开模块。

只依赖 Python 3.14 标准库。核心流程：

1. 对每个四点全有效的 2×2 环路，用包裹到 [-pi, pi) 的邻差求和，
   四舍五入到最近整数得到整数残差电荷（典型为 -1/0/+1）。
2. 残差按 (row, col) 字典序处理：每个未配对残差在对偶格点图上 BFS，
   找曼哈顿距离最短的异号残差配对，等距时取字典序最小者；
   配不上的剩余残差各自走最短有效路径到图像外边界。
3. 路径跨过的像素邻接边置为切口，洪泛积分时不可跨越。
4. 每个切口连通区从字典序最小像素起洪凡展开，锚点整数偏移基准为 0。
5. 对所有非切口有效边做一致性复核，矛盾以 unmet_edges 如实输出，
   不用平滑结果掩盖。

接口与错误语义见模块 README。
"""

from __future__ import annotations

import math
from collections import deque
from typing import NamedTuple, Optional

__all__ = [
    "MAX_SIZE",
    "TWO_PI",
    "RESIDUE_TOLERANCE",
    "UNWRAP_TOLERANCE",
    "DEFAULT_SEARCH_BUDGET",
    "MERGE_TOLERANCE",
    "FEASIBILITY_TOLERANCE",
    "Residue",
    "CutEdge",
    "UnmetEdge",
    "Region",
    "UnwrapResult",
    "DualFrequencyResult",
    "PhaseCutError",
    "InvalidPhaseInputError",
    "SearchBudgetExhausted",
    "NoSolutionError",
    "UnwrappingContradictionError",
    "IncompatibleObservationsError",
    "wrap_phase",
    "unwrap_phase",
    "resolve_absolute_cycles",
]

MAX_SIZE = 32
TWO_PI = 2.0 * math.pi
PI = math.pi
# 环和判定为 2π 整数倍、以及展开边一致性的允许口径（弧度）。
# 32×32 内四点包裹差与任意长度路径的浮点散布实测在 1e-13 量级，
# 1e-9 是留有千倍余量的严格口径，不是用来掩盖边界错误的宽容差。
RESIDUE_TOLERANCE = 1e-9
UNWRAP_TOLERANCE = 1e-9
DEFAULT_SEARCH_BUDGET = 1_000_000
# 可行区间合并口径：两个区间端点间距不超过它时视为相接并合并。
# 只用于把“误差带恰接触”的相邻区间表示为一段，不影响单点可行性判定
# （单点判定用严格的不等式 |f*phi - w - 2*pi*n| <= err，边界含端点）。
MERGE_TOLERANCE = 1e-12
# 整数周数枚举与零宽区间判定的浮点口径（弧度，绝对相位单位）。
# 误差带端点由不同路径的浮点运算得到，理论重合点可能有 ~1e-16 的散布；
# 1e-9 留有巨大余量，真正的间隔（误差带分离）必然远大于它。
FEASIBILITY_TOLERANCE = 1e-9

# 固定的四方向字典序：北、东、南、西。BFS 首次访问序由此唯一确定，
# 从而等距路径也是确定性的。
_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


# --------------------------------------------------------------------------
# 公开数据结构（全部不可变）
# --------------------------------------------------------------------------
class Residue(NamedTuple):
    """一个有效 2×2 环路的整数残差。

    (row, col) 为环路左上角像素坐标；charge 为环和 / (2π) 的最近整数。
    """

    row: int
    col: int
    charge: int


class CutEdge(NamedTuple):
    """一条切口边：两个四邻接有效像素，端点按字典序排列。"""

    row1: int
    col1: int
    row2: int
    col2: int

    def as_pixels(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return (self.row1, self.col1), (self.row2, self.col2)


class UnmetEdge(NamedTuple):
    """一条非切口有效边的展开矛盾。

    residual = (phi_b - phi_a) - wrap(w_b - w_a)，单位弧度，
    端点字典序、方向取小坐标指向大坐标；|residual| > UNWRAP_TOLERANCE 才记录。
    """

    row1: int
    col1: int
    row2: int
    col2: int
    residual: float


class Region(NamedTuple):
    """切口图上的一个像素连通区。

    anchor: 区内字典序最小像素；baseline: 锚点整数偏移 k（恒为 0，
    各区之间相差任意 2π 整数倍，本模块不做全局对齐）。
    pixels 按字典序排列，offsets[i] 为 pixels[i] 的整数偏移 k，
    展开值 phi = wrapped + 2π * k。
    """

    anchor: tuple[int, int]
    baseline: int
    pixels: tuple[tuple[int, int], ...]
    offsets: tuple[int, ...]


class UnwrapResult(NamedTuple):
    residues: tuple[Residue, ...]
    cuts: tuple[CutEdge, ...]
    regions: tuple[Region, ...]
    #: 与输入同形的展开相位；无效像素位置为 None。
    unwrapped: tuple[tuple[Optional[float], ...], ...]
    unmet_edges: tuple[UnmetEdge, ...]


# --------------------------------------------------------------------------
# 错误类型
# --------------------------------------------------------------------------
class PhaseCutError(Exception):
    """本模块所有异常的基类。"""


class InvalidPhaseInputError(PhaseCutError, ValueError):
    """非法输入；row/col 给出可定位坐标（结构错误时可能为 None）。"""

    def __init__(self, message: str, row: Optional[int] = None, col: Optional[int] = None):
        self.message = message
        self.row = row
        self.col = col
        location = ""
        if row is not None and col is not None:
            location = f" [row={row}, col={col}]"
        elif row is not None:
            location = f" [row={row}]"
        super().__init__(message + location)


class SearchBudgetExhausted(PhaseCutError):
    """BFS 访问节点数达到 search_budget，搜索未能完成。

    这是“不知道是否有解”，区别于 NoSolutionError 的“已证明无解”。
    """


class NoSolutionError(PhaseCutError):
    """完整 BFS 证明剩余残差既无异号可配、也无有效路径到达外边界。"""


class UnwrappingContradictionError(PhaseCutError):
    """有效 2×2 环的包裹邻差之和不接近 2π 的整数倍（数据自相矛盾）。"""

    def __init__(self, row: int, col: int, loop_sum: float):
        self.row = row
        self.col = col
        self.loop_sum = loop_sum
        super().__init__(
            f"wrapped loop sum at (row={row}, col={col}) is {loop_sum!r}, "
            f"not within {RESIDUE_TOLERANCE:g} rad of an integer multiple of 2*pi"
        )


class IncompatibleObservationsError(PhaseCutError):
    """可选的严格模式：某有效像素在给定误差带与区间内没有任何可行解。

    默认模式（strict=False）不会抛出本异常，而是在结果中把该像素记为
    空区间元组并计入 empty_pixels；strict=True 时遇到第一个不相容像素
    即抛出，携带可定位的 row/col。
    """

    def __init__(self, row: int, col: int):
        self.row = row
        self.col = col
        super().__init__(
            f"no feasible absolute phase at (row={row}, col={col}): "
            f"the two wrapped observations are incompatible within their "
            f"error bands on the given interval"
        )


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------
def wrap_phase(value: float) -> float:
    """把任意相位包裹到 [-pi, pi)。"""
    return (value + PI) % TWO_PI - PI


def _nearest_int(x: float) -> int:
    n = math.floor(abs(x) + 0.5)
    return n if x >= 0.0 else -n


def _canonical_edge(a: tuple[int, int], b: tuple[int, int]) -> CutEdge:
    if a <= b:
        (r1, c1), (r2, c2) = a, b
    else:
        (r1, c1), (r2, c2) = b, a
    return CutEdge(r1, c1, r2, c2)


# --------------------------------------------------------------------------
# 输入校验
# --------------------------------------------------------------------------
def _validate(phase, mask, search_budget: int):
    if not isinstance(search_budget, int) or isinstance(search_budget, bool):
        raise InvalidPhaseInputError(
            "search_budget must be an int; bool is not accepted"
        )
    if search_budget < 1:
        raise InvalidPhaseInputError("search_budget must be >= 1")

    if isinstance(phase, (str, bytes)) or not isinstance(phase, (list, tuple)):
        raise InvalidPhaseInputError(
            "phase must be a sequence of numeric rows"
        )
    if isinstance(mask, (str, bytes)) or not isinstance(mask, (list, tuple)):
        raise InvalidPhaseInputError("mask must be a sequence of bool rows")
    if len(phase) == 0:
        raise InvalidPhaseInputError("phase must have at least 1 row", row=0)
    if len(mask) != len(phase):
        raise InvalidPhaseInputError(
            f"mask has {len(mask)} rows but phase has {len(phase)}"
        )
    if len(phase) > MAX_SIZE:
        raise InvalidPhaseInputError(
            f"phase height {len(phase)} exceeds MAX_SIZE={MAX_SIZE}", row=len(phase) - 1
        )

    height = len(phase)
    width: Optional[int] = None
    p_grid: list[list[float]] = []
    m_grid: list[list[bool]] = []

    for r, (prow, mrow) in enumerate(zip(phase, mask)):
        if isinstance(prow, (str, bytes)) or not isinstance(prow, (list, tuple)):
            raise InvalidPhaseInputError("phase row must be a sequence", row=r)
        if isinstance(mrow, (str, bytes)) or not isinstance(mrow, (list, tuple)):
            raise InvalidPhaseInputError("mask row must be a sequence of bool", row=r)
        if width is None:
            width = len(prow)
            if width == 0:
                raise InvalidPhaseInputError("phase row must be non-empty", row=r, col=0)
            if width > MAX_SIZE:
                raise InvalidPhaseInputError(
                    f"phase width {width} exceeds MAX_SIZE={MAX_SIZE}", row=r, col=width - 1
                )
        elif len(prow) != width:
            raise InvalidPhaseInputError(
                f"phase row length {len(prow)} != {width}", row=r
            )
        if len(mrow) != len(prow):
            raise InvalidPhaseInputError(
                f"mask row length {len(mrow)} != phase row length {len(prow)}", row=r
            )

        p_out: list[float] = []
        m_out: list[bool] = []
        for c, (v, mv) in enumerate(zip(prow, mrow)):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise InvalidPhaseInputError(
                    "phase value must be a real int/float (bool, complex, str rejected)",
                    row=r,
                    col=c,
                )
            fv = float(v)
            if math.isnan(fv) or math.isinf(fv):
                raise InvalidPhaseInputError(
                    "phase value must be finite (NaN/Infinity rejected)", row=r, col=c
                )
            if not (-PI <= fv < PI):
                raise InvalidPhaseInputError(
                    "phase value must lie in [-pi, pi); use wrap_phase() first",
                    row=r,
                    col=c,
                )
            if not isinstance(mv, bool):
                raise InvalidPhaseInputError(
                    "mask value must be bool (int 0/1 not silently accepted)", row=r, col=c
                )
            p_out.append(fv)
            m_out.append(mv)
        p_grid.append(p_out)
        m_grid.append(m_out)

    assert width is not None
    return p_grid, m_grid, height, width


# --------------------------------------------------------------------------
# 残差
# --------------------------------------------------------------------------
def _loop_sum(w: list[list[float]], r: int, c: int) -> float:
    tl, tr, br, bl = w[r][c], w[r][c + 1], w[r + 1][c + 1], w[r + 1][c]
    # 逆时针：TL→TR→BR→BL→TL 的包裹邻差之和。
    return (
        wrap_phase(tr - tl)
        + wrap_phase(br - tr)
        + wrap_phase(bl - br)
        + wrap_phase(tl - bl)
    )


def _compute_residues(w, mask, height, width):
    residues: list[Residue] = []
    charges: dict[tuple[int, int], int] = {}
    for r in range(height - 1):
        for c in range(width - 1):
            if not (
                mask[r][c]
                and mask[r][c + 1]
                and mask[r + 1][c + 1]
                and mask[r + 1][c]
            ):
                continue
            s = _loop_sum(w, r, c)
            q = _nearest_int(s / TWO_PI)
            if abs(s - q * TWO_PI) > RESIDUE_TOLERANCE:
                raise UnwrappingContradictionError(r, c, s)
            if q != 0:
                residues.append(Residue(r, c, q))
                charges[(r, c)] = q
    residues.sort()
    return tuple(residues), charges


# --------------------------------------------------------------------------
# 对偶图 BFS
# --------------------------------------------------------------------------
def _dual_moves(cell, mask, height, width):
    """从对偶格 cell=(r,c) 出发的步：(nr,nc) 或 None(出界), 跨过的像素边, 方向序。

    一步有效当且仅当被跨过的像素边两端都是有效像素。
    """
    r, c = cell
    moves = []
    # 北：跨过 (r,c)-(r,c+1)
    edge = ((r, c), (r, c + 1))
    if mask[r][c] and mask[r][c + 1]:
        moves.append((None if r == 0 else (r - 1, c), edge, 0))
    # 东：跨过 (r,c+1)-(r+1,c+1)
    edge = ((r, c + 1), (r + 1, c + 1))
    if mask[r][c + 1] and mask[r + 1][c + 1]:
        moves.append((None if c == width - 2 else (r, c + 1), edge, 1))
    # 南：跨过 (r+1,c)-(r+1,c+1)
    edge = ((r + 1, c), (r + 1, c + 1))
    if mask[r + 1][c] and mask[r + 1][c + 1]:
        moves.append((None if r == height - 2 else (r + 1, c), edge, 2))
    # 西：跨过 (r,c)-(r+1,c)
    edge = ((r, c), (r + 1, c))
    if mask[r][c] and mask[r + 1][c]:
        moves.append((None if c == 0 else (r, c - 1), edge, 3))
    return moves


def _dual_bfs(start, mask, height, width, budget: int):
    """从残差格做完整 BFS，返回 (dist, parent_edge, used)。

    parent_edge[cell] = (上一单元格, 跨过的 CutEdge)；起点为 None。
    超出 budget 抛 SearchBudgetExhausted（结果丢弃，不产生部分切口）。
    """
    dist = {start: 0}
    parent: dict[tuple[int, int], Optional[tuple[tuple[int, int], CutEdge]]] = {
        start: None
    }
    queue = deque([start])
    used = 1
    while queue:
        cell = queue.popleft()
        for nxt, edge_px, _dir in _dual_moves(cell, mask, height, width):
            if nxt is None or nxt in dist:
                continue
            if used >= budget:
                raise SearchBudgetExhausted(
                    f"dual-graph BFS node budget {budget} exhausted from residue {start}"
                )
            used += 1
            dist[nxt] = dist[cell] + 1
            parent[nxt] = (cell, _canonical_edge(*edge_px))
            queue.append(nxt)
    return dist, parent, used


def _reconstruct(parent, target) -> list[CutEdge]:
    edges: list[CutEdge] = []
    cell = target
    while parent[cell] is not None:
        prev, edge = parent[cell]
        edges.append(edge)
        cell = prev
    edges.reverse()
    return edges


def _border_path(parent, exit_cell, exit_edge: CutEdge) -> list[CutEdge]:
    return _reconstruct(parent, exit_cell) + [exit_edge]


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------
def unwrap_phase(phase, mask, *, search_budget: int = DEFAULT_SEARCH_BUDGET) -> UnwrapResult:
    """残差切口相位展开（纯函数：不修改 phase/mask）。

    非法输入抛 InvalidPhaseInputError（带 row/col）；搜索预算耗尽抛
    SearchBudgetExhausted；剩余残差证明无法到达外边界抛 NoSolutionError；
    环和不是 2π 整数倍抛 UnwrappingContradictionError。
    """
    w, m, height, width = _validate(phase, mask, search_budget)

    residues, charges = _compute_residues(w, m, height, width)

    cuts: set[CutEdge] = set()
    unpaired: set[tuple[int, int]] = {(res.row, res.col) for res in residues}

    # ---- 第一遍：字典序贪心，异号最短曼哈顿配对 --------------------------
    # 先处理的残差若当时没有可达的异号对象，会自然留在 unpaired 中；
    # 之后处理到的异号残差做 BFS 时仍可把它配走（无向图，可达对称）。
    for res in residues:
        src = (res.row, res.col)
        if src not in unpaired:
            continue
        targets = [p for p in unpaired if p != src and charges[p] == -charges[src]]
        if not targets:
            continue
        dist, parent, _used = _dual_bfs(src, m, height, width, search_budget)
        reachable = [(dist[p], p) for p in targets if p in dist]
        if not reachable:
            continue
        reachable.sort(key=lambda t: (t[0], t[1]))
        target = reachable[0][1]
        path = _reconstruct(parent, target)
        cuts.update(path)
        unpaired.remove(src)
        unpaired.remove(target)

    # ---- 第二遍：剩余残差（按字典序）各自连到外边界 ----------------------
    for src in sorted(unpaired):
        dist, parent, _used = _dual_bfs(src, m, height, width, search_budget)
        best = None  # (总长, 出口格字典序, 方向序, 路径)
        for cell, d in dist.items():
            r, c = cell
            for nxt, edge_px, dir_idx in _dual_moves(cell, m, height, width):
                if nxt is not None:
                    continue  # 仅 nxt is None 表示这一步直接跨出图像外边界
                key = (d + 1, cell, dir_idx)
                if best is None or key < best[0]:
                    best = (
                        key,
                        _border_path(parent, cell, _canonical_edge(*edge_px)),
                    )
        if best is None:
            raise NoSolutionError(
                f"residue {src} charge {charges[src]} has no opposite mate and no "
                f"valid path to the outer border (trapped by the mask)"
            )
        cuts.update(best[1])

    cut_sorted = tuple(sorted(cuts))

    # ---- 切口图上的连通区与洪泛积分 --------------------------------------
    valid = {(r, c) for r in range(height) for c in range(width) if m[r][c]}
    visited: set[tuple[int, int]] = set()
    regions_out: list[Region] = []
    k_map: dict[tuple[int, int], int] = {}
    phi: dict[tuple[int, int], float] = {}

    for start in sorted(valid):
        if start in visited:
            continue
        anchor = start
        stack = [start]
        visited.add(start)
        k_map[start] = 0
        phi[start] = w[start[0]][start[1]]
        pixels: list[tuple[int, int]] = []
        while stack:
            pr, pc = stack.pop()
            pixels.append((pr, pc))
            for dr, dc in _DIRECTIONS:
                nr, nc = pr + dr, pc + dc
                nxt = (nr, nc)
                if nxt not in valid or nxt in visited:
                    continue
                if _canonical_edge((pr, pc), nxt) in cuts:
                    continue
                visited.add(nxt)
                delta = wrap_phase(w[nr][nc] - w[pr][pc])
                expected = phi[(pr, pc)] + delta
                k = _nearest_int((expected - w[nr][nc]) / TWO_PI)
                k_map[nxt] = k
                phi[nxt] = w[nr][nc] + k * TWO_PI
                stack.append(nxt)
        pixels.sort()
        regions_out.append(
            Region(
                anchor=anchor,
                baseline=0,
                pixels=tuple(pixels),
                offsets=tuple(k_map[p] for p in pixels),
            )
        )

    regions_out.sort(key=lambda rg: rg.anchor)

    # ---- 全局边一致性复核（矛盾通道，不掩盖） -----------------------------
    unmet: list[UnmetEdge] = []
    for r in range(height):
        for c in range(width):
            if not m[r][c]:
                continue
            for dr, dc in ((0, 1), (1, 0)):
                nr, nc = r + dr, c + dc
                if nr >= height or nc >= width or not m[nr][nc]:
                    continue
                edge = CutEdge(r, c, nr, nc)
                if edge in cuts:
                    continue
                residual = (phi[(nr, nc)] - phi[(r, c)]) - wrap_phase(
                    w[nr][nc] - w[r][c]
                )
                if abs(residual) > UNWRAP_TOLERANCE:
                    unmet.append(UnmetEdge(r, c, nr, nc, residual))
    unmet.sort()

    grid = tuple(
        tuple(phi[(r, c)] if m[r][c] else None for c in range(width))
        for r in range(height)
    )

    return UnwrapResult(
        residues=residues,
        cuts=cut_sorted,
        regions=tuple(regions_out),
        unwrapped=grid,
        unmet_edges=tuple(unmet),
    )


# ==========================================================================
# 双频包裹相位联合确定绝对周数
# ==========================================================================
class DualFrequencyResult(NamedTuple):
    """两幅同位置包裹相位在两个互素频率下的联合解。

    intervals / candidate_counts 与输入同形；mask 无效位置为 None。
    intervals[r][c] 为 ((lo, hi), ...)，按 lo 升序、互不重迭的可行绝对
    相位区间（含端点）；空元组表示该像素在误差带与区间内无可行解。
    candidate_counts[r][c] = len(intervals[r][c])，>1 即歧义像素。
    ambiguous_pixels / empty_pixels 为全图计数；smoothing_applied 记录
    是否应用了显式可选的邻域筛选。
    """

    freq1: int
    freq2: int
    phi_min: float
    phi_max: float
    intervals: tuple[tuple[Optional[tuple[tuple[float, float], ...]], ...], ...]
    candidate_counts: tuple[tuple[Optional[int], ...], ...]
    ambiguous_pixels: int
    empty_pixels: int
    smoothing_applied: bool


def _validate_dual(phase1, phase2, freq1, freq2, err1, err2,
                   phi_min, phi_max, mask):
    for name, f in (("freq1", freq1), ("freq2", freq2)):
        if not isinstance(f, int) or isinstance(f, bool):
            raise InvalidPhaseInputError(f"{name} must be an int; bool is not accepted")
        if f < 1:
            raise InvalidPhaseInputError(f"{name} must be a positive integer")
    if math.gcd(freq1, freq2) != 1:
        raise InvalidPhaseInputError(
            f"freq1={freq1} and freq2={freq2} must be coprime"
        )
    for name, e in (("err1", err1), ("err2", err2)):
        if isinstance(e, bool) or not isinstance(e, (int, float)):
            raise InvalidPhaseInputError(f"{name} must be a real int/float")
        if math.isnan(float(e)) or math.isinf(float(e)) or float(e) < 0.0:
            raise InvalidPhaseInputError(f"{name} must be a finite number >= 0")
    for name, p in (("phi_min", phi_min), ("phi_max", phi_max)):
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            raise InvalidPhaseInputError(f"{name} must be a real int/float")
        if math.isnan(float(p)) or math.isinf(float(p)):
            raise InvalidPhaseInputError(f"{name} must be finite")
    if not float(phi_min) < float(phi_max):
        raise InvalidPhaseInputError("phi_min must be strictly less than phi_max")

    # 复用 unwrap_phase 的结构/数值校验（相位域、有限性、mask 布尔性）。
    full_mask = [[True] * len(phase1[0]) for _ in range(len(phase1))] \
        if isinstance(phase1, (list, tuple)) and phase1 \
        and isinstance(phase1[0], (list, tuple)) else [[True]]
    w1, _m1, height, width = _validate(phase1, full_mask, 1)
    w2, _m2, h2, w2d = _validate(phase2, full_mask, 1)
    if (h2, w2d) != (height, width):
        raise InvalidPhaseInputError(
            f"phase2 shape ({h2}x{w2d}) != phase1 shape ({height}x{width})"
        )
    if mask is None:
        m_grid = [[True] * width for _ in range(height)]
    else:
        _w, m_grid, mh, mw = _validate(full_mask_phase(height, width), mask, 1)
        if (mh, mw) != (height, width):
            raise InvalidPhaseInputError(
                f"mask shape ({mh}x{mw}) != phase shape ({height}x{width})"
            )
    return w1, w2, m_grid, height, width


def full_mask_phase(height: int, width: int):
    """内部辅助：生成 height×width 的全零相位（仅用于复用 mask 校验）。"""
    return [[0.0] * width for _ in range(height)]


def _pixel_intervals(w1v, w2v, freq1, freq2, err1, err2, phi_min, phi_max):
    """单像素：返回 [phi_min, phi_max] 内所有可行绝对相位区间（升序）。

    第一步（周期关系缩小组合）：频率 1 的可行带中心为
    (w1 + 2*pi*n1)/f1，落在区间内的 n1 只有有限个，直接枚举；
    对每个 n1，频率 2 的周数 n2 由
    n2 ≈ (f2*(w1 + 2*pi*n1) - f1*w2) / (2*pi*f1) 取整唯一锁定
    （互素保证合成周期为 2*pi，n1 -> n1+f1、n2 -> n2+f2 给出下一个
    合成周期处的同构组合）。
    第二步（逐一代回两种频率检验）：把每个 (n1, n2) 组合的两条误差带
    [(w_i + 2*pi*n_i - err_i)/f_i, (w_i + 2*pi*n_i + err_i)/f_i]
    与 [phi_min, phi_max] 求交，非空（含端点恰接触）即保留，最后合并
    相接/重迭段。
    """
    lo1 = (freq1 * phi_min - err1 - w1v) / TWO_PI
    hi1 = (freq1 * phi_max + err1 - w1v) / TWO_PI
    n1_lo = math.ceil(lo1 - FEASIBILITY_TOLERANCE)
    n1_hi = math.floor(hi1 + FEASIBILITY_TOLERANCE)
    raw: list[tuple[float, float]] = []
    for n1 in range(n1_lo, n1_hi + 1):
        band1_lo = (w1v + TWO_PI * n1 - err1) / freq1
        band1_hi = (w1v + TWO_PI * n1 + err1) / freq1
        # 频率 2 侧：只可能与覆盖 band1 扩展范围的 n2 带相交。
        n2_lo = math.ceil((freq2 * band1_lo - err2 - w2v) / TWO_PI
                          - FEASIBILITY_TOLERANCE)
        n2_hi = math.floor((freq2 * band1_hi + err2 - w2v) / TWO_PI
                           + FEASIBILITY_TOLERANCE)
        for n2 in range(n2_lo, n2_hi + 1):
            lo = max(band1_lo, (w2v + TWO_PI * n2 - err2) / freq2, phi_min)
            hi = min(band1_hi, (w2v + TWO_PI * n2 + err2) / freq2, phi_max)
            # 端点恰接触（lo == hi）也是可行点；1e-9 口径吸收零宽带的
            # 浮点散布，真正的间隔必然远大于它。
            if lo <= hi + FEASIBILITY_TOLERANCE:
                hi = max(lo, hi)
                raw.append((lo, hi))
    if not raw:
        return ()
    raw.sort()
    merged: list[list[float]] = [[raw[0][0], raw[0][1]]]
    for lo, hi in raw[1:]:
        if lo <= merged[-1][1] + MERGE_TOLERANCE:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return tuple((lo, hi) for lo, hi in merged)


def _intervals_overlap(a, b) -> bool:
    return any(lo <= bh + MERGE_TOLERANCE and blo <= h + MERGE_TOLERANCE
               for lo, h in a for blo, bh in b)


def resolve_absolute_cycles(phase1, phase2, freq1, freq2, *,
                            err1, err2, phi_min, phi_max, mask=None,
                            neighborhood_filter: bool = False,
                            strict: bool = False) -> DualFrequencyResult:
    """两幅同位置包裹相位 + 两个互素频率 -> 每像素全部可行绝对相位区间。

    phase1/phase2：同形包裹相位（[-pi, pi)，校验口径同 unwrap_phase）。
    freq1/freq2：互素正整数频率（拒绝 bool、非互素）。
    err1/err2：两幅观测各自独立的包裹相位误差上限（弧度，>= 0）。
    [phi_min, phi_max]：有限的绝对相位搜索区间（含端点）。
    mask：可选 bool 网格；无效像素不参与任何空间传播，结果中为 None。
    neighborhood_filter：显式可选的邻域多数筛选（默认关闭）。开启后，
        对每个多解像素，若其全部四邻接有效像素的可行区间与自身某一
        区间公共相交，则只保留该区间。这只是后处理筛选，可能丢弃真实
        多解，结果中 smoothing_applied=True 如实标记。
    strict：默认 False，不相容像素在结果中记为空区间并计入
        empty_pixels；True 时遇到第一个不相容像素即抛
        IncompatibleObservationsError（带 row/col）。

    绝不“挑最接近 0 的一个”：所有可行区间一律如实返回，歧义以
    candidate_counts / ambiguous_pixels 显式暴露。
    """
    if not isinstance(neighborhood_filter, bool):
        raise InvalidPhaseInputError("neighborhood_filter must be a bool")
    if not isinstance(strict, bool):
        raise InvalidPhaseInputError("strict must be a bool")
    w1, w2, m, height, width = _validate_dual(
        phase1, phase2, freq1, freq2, err1, err2, phi_min, phi_max, mask
    )
    err1 = float(err1)
    err2 = float(err2)
    phi_min = float(phi_min)
    phi_max = float(phi_max)

    grid: list[list[Optional[tuple[tuple[float, float], ...]]]] = [
        [None] * width for _ in range(height)
    ]
    empty = 0
    for r in range(height):
        for c in range(width):
            if not m[r][c]:
                continue
            iv = _pixel_intervals(
                w1[r][c], w2[r][c], freq1, freq2, err1, err2, phi_min, phi_max
            )
            if not iv and strict:
                raise IncompatibleObservationsError(r, c)
            if not iv:
                empty += 1
            grid[r][c] = iv

    smoothing = False
    if neighborhood_filter:
        smoothing = True
        for r in range(height):
            for c in range(width):
                cur = grid[r][c]
                if cur is None or len(cur) <= 1:
                    continue
                common = list(cur)
                for dr, dc in _DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    nb = grid[nr][nc]
                    if nb is None or not nb:
                        continue
                    common = [iv for iv in common if _intervals_overlap([iv], nb)]
                if len(common) == 1:
                    grid[r][c] = (common[0],)

    counts = tuple(
        tuple(None if grid[r][c] is None else len(grid[r][c]) for c in range(width))
        for r in range(height)
    )
    ambiguous = sum(
        1 for r in range(height) for c in range(width)
        if counts[r][c] is not None and counts[r][c] > 1
    )
    return DualFrequencyResult(
        freq1=freq1,
        freq2=freq2,
        phi_min=phi_min,
        phi_max=phi_max,
        intervals=tuple(tuple(row) for row in grid),
        candidate_counts=counts,
        ambiguous_pixels=ambiguous,
        empty_pixels=empty,
        smoothing_applied=smoothing,
    )
