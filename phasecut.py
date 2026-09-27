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
    "Residue",
    "CutEdge",
    "UnmetEdge",
    "Region",
    "PhaseReference",
    "UnwrapResult",
    "PhaseCutError",
    "InvalidPhaseInputError",
    "SearchBudgetExhausted",
    "NoSolutionError",
    "UnwrappingContradictionError",
    "AlignmentConflictError",
    "wrap_phase",
    "unwrap_phase",
    "align_regions",
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


class PhaseReference(NamedTuple):
    """一个区域对齐参考点：已知像素位置与目标相位。

    (row, col) 必须是有效像素；target 为该位置展开后应取的目标相位
    （弧度，任意实数，不限于 [-pi, pi)）。所属连通区据此确定整体
    整数偏移 k，使展开值平移 2π·k 后该点等于 target（在口径内）。
    """

    row: int
    col: int
    target: float


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


class AlignmentConflictError(PhaseCutError):
    """同一连通区的参考点要求不同的整数偏移，或目标相位与局部结果矛盾。

    anchor: 冲突连通区的锚点；details: 该冲突涉及的全部参考点及其各自
    要求的整数偏移，元素为 (row, col, target, required_shift)。
    冲突如实抛出，绝不用平滑/折中偏移掩盖。
    """

    def __init__(self, anchor, details):
        self.anchor = anchor
        self.details = tuple(details)
        rendered = ", ".join(
            f"(row={r}, col={c}, target={t!r}, required_shift={k})"
            for r, c, t, k in self.details
        )
        super().__init__(
            f"conflicting alignment references in region anchored at {anchor}: "
            + rendered
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


# --------------------------------------------------------------------------
# 区域参考点对齐
# --------------------------------------------------------------------------
def _validate_reference(ref, index: int) -> PhaseReference:
    if isinstance(ref, (str, bytes)) or not isinstance(ref, (list, tuple, PhaseReference)):
        raise InvalidPhaseInputError(
            f"reference #{index} must be a (row, col, target) sequence"
        )
    if len(ref) != 3:
        raise InvalidPhaseInputError(
            f"reference #{index} must have exactly 3 elements (row, col, target)"
        )
    row, col, target = ref
    if not isinstance(row, int) or isinstance(row, bool):
        raise InvalidPhaseInputError(
            f"reference #{index} row must be an int (bool rejected)"
        )
    if not isinstance(col, int) or isinstance(col, bool):
        raise InvalidPhaseInputError(
            f"reference #{index} col must be an int (bool rejected)", row=row
        )
    if isinstance(target, bool) or not isinstance(target, (int, float)):
        raise InvalidPhaseInputError(
            f"reference #{index} target must be a real int/float "
            "(bool, complex, str rejected)",
            row=row,
            col=col,
        )
    ftarget = float(target)
    if math.isnan(ftarget) or math.isinf(ftarget):
        raise InvalidPhaseInputError(
            f"reference #{index} target must be finite (NaN/Infinity rejected)",
            row=row,
            col=col,
        )
    return PhaseReference(row, col, ftarget)


def align_regions(
    result: UnwrapResult,
    references,
    *,
    tolerance: float = UNWRAP_TOLERANCE,
) -> UnwrapResult:
    """按参考点把各连通区整体平移整数个 2π，返回新的 UnwrapResult。

    references: (row, col, target) 序列的序列（或 PhaseReference）。
    每个参考点落在某个连通区内，要求该区整体偏移 k 使该点展开值
    平移 2π·k 后在 tolerance 口径内等于 target。

    - 同一区域的多个参考点必须要求**同一个** k，且每个 target 都必须
      与“局部结果 + 2π·k”在口径内一致；否则抛 AlignmentConflictError
      （带 anchor 与每个参考点各自要求的偏移），不做任何平滑或折中。
    - 没有参考点的区域保持未对齐：baseline/offsets/unwrapped 原样保留。
    - residues、cuts、unmet_edges 原样保留——对齐不消灭切口与未满足边
      的报告。
    - 参考位置越界或落在无效像素上抛 InvalidPhaseInputError（带坐标）。
    - 纯函数：不修改入参 result；任何失败都抛异常，不返回部分对齐结果。
    """
    if not isinstance(result, UnwrapResult):
        raise InvalidPhaseInputError("result must be an UnwrapResult")
    if isinstance(references, (str, bytes)) or not isinstance(references, (list, tuple)):
        raise InvalidPhaseInputError(
            "references must be a sequence of (row, col, target)"
        )
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise InvalidPhaseInputError("tolerance must be a real number")
    tol = float(tolerance)
    if math.isnan(tol) or math.isinf(tol) or tol < 0.0:
        raise InvalidPhaseInputError("tolerance must be a finite, non-negative number")

    refs = tuple(_validate_reference(ref, i) for i, ref in enumerate(references))

    height = len(result.unwrapped)
    width = len(result.unwrapped[0]) if height else 0

    # 像素 -> 区域下标
    pixel_region: dict[tuple[int, int], int] = {}
    for idx, region in enumerate(result.regions):
        for p in region.pixels:
            pixel_region[p] = idx

    # 逐参考点求所需整数偏移，按区域归组。
    region_refs: dict[int, list[tuple[int, int, float, int]]] = {}
    for ref in refs:
        if not (0 <= ref.row < height and 0 <= ref.col < width):
            raise InvalidPhaseInputError(
                "reference position out of grid bounds", row=ref.row, col=ref.col
            )
        pos = (ref.row, ref.col)
        if pos not in pixel_region:
            raise InvalidPhaseInputError(
                "reference position is not a valid (masked-in) pixel",
                row=ref.row,
                col=ref.col,
            )
        current = result.unwrapped[ref.row][ref.col]
        k = _nearest_int((ref.target - current) / TWO_PI)
        region_refs.setdefault(pixel_region[pos], []).append(
            (ref.row, ref.col, ref.target, k)
        )

    # 冲突检测：同区 k 不一致，或 target 与 局部结果+2π·k 超口径。
    region_shift: dict[int, int] = {}
    for idx in sorted(region_refs):
        entries = region_refs[idx]
        shifts = {k for _r, _c, _t, k in entries}
        anchor = result.regions[idx].anchor
        if len(shifts) != 1:
            raise AlignmentConflictError(anchor, entries)
        k = shifts.pop()
        bad = [
            (r, c, t, kk)
            for r, c, t, kk in entries
            if abs(result.unwrapped[r][c] + kk * TWO_PI - t) > tol
        ]
        if bad:
            raise AlignmentConflictError(anchor, entries)
        region_shift[idx] = k

    # 应用偏移：有参考的区域整体平移，其余区域原样保留。
    new_regions: list[Region] = []
    new_phi: dict[tuple[int, int], float] = {}
    for idx, region in enumerate(result.regions):
        k = region_shift.get(idx, 0)
        for p, off in zip(region.pixels, region.offsets):
            new_phi[p] = result.unwrapped[p[0]][p[1]] + k * TWO_PI
        new_regions.append(
            Region(
                anchor=region.anchor,
                baseline=region.baseline + k,
                pixels=region.pixels,
                offsets=tuple(off + k for off in region.offsets),
            )
        )

    grid = tuple(
        tuple(
            new_phi[(r, c)] if (r, c) in new_phi else None
            for c in range(width)
        )
        for r in range(height)
    )

    return UnwrapResult(
        residues=result.residues,
        cuts=result.cuts,
        regions=tuple(new_regions),
        unwrapped=grid,
        unmet_edges=result.unmet_edges,
    )
