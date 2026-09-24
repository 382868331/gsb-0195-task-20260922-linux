"""phasecut — 二维包裹相位的残差切口（branch cut）展开模块。

只依赖 Python 3.14 标准库。单帧空间流程：

1. 对每个四点全有效的 2×2 环路，用包裹到 [-pi, pi) 的邻差求和，
   四舍五入到最近整数得到整数残差电荷（典型为 -1/0/+1）。
2. 残差按 (row, col) 字典序处理：每个未配对残差在对偶格点图上 BFS，
   找曼哈顿距离最短的异号残差配对，等距时取字典序最小者；
   配不上的剩余残差各自走最短有效路径到图像外边界。
3. 路径跨过的像素邻接边置为切口，洪泛积分时不可跨越。
4. 每个切口连通区从字典序最小像素起洪凡展开，锚点整数偏移基准为 0。
5. 对所有非切口有效边做一致性复核，矛盾以 unmet_edges 如实输出，
   不用平滑结果掩盖。

相邻两帧（align_frames）：两帧各自独立完成上述空间展开后，取共同有效
像素，选一个全局整数周 k 最小化 Σ(phi_a-(phi_b+2πk))²；凸二次只需比较
floor(mean(d)/2π) 与 +1，并列取较小整数；无共同像素返回未对齐状态。

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
    "UnwrapResult",
    "FrameAlignment",
    "PhaseCutError",
    "InvalidPhaseInputError",
    "SearchBudgetExhausted",
    "NoSolutionError",
    "UnwrappingContradictionError",
    "wrap_phase",
    "unwrap_phase",
    "align_frames",
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
    #: 与输入同形的展开相位；无效像素为 None。
    unwrapped: tuple[tuple[Optional[float], ...], ...]
    unmet_edges: tuple[UnmetEdge, ...]


class FrameAlignment(NamedTuple):
    """相邻两帧各自空间展开后的全局整数周对齐结果。

    frame_a / frame_b：两帧各自的 UnwrapResult（旧空间展开，逐帧独立）。
    aligned：两帧是否存在共同有效像素（False 时其余对齐字段为 None；
             两帧的展开结果与掩码仍完整保留）。
    shift：全局整数周偏移 k，使 frame_b 对齐到 frame_a：
           phi_b_aligned = phi_b + 2π·k，在共同有效像素上平方差最小；
           并列（平方差相等）时取较小整数。
    shared_pixels：共同有效像素坐标，按字典序排列；无共同像素时为 ()。
    squared_error：对齐后共同像素上的 Σ(phi_a - (phi_b + 2π·k))²（rad²）；
                   无共同像素时为 None。
    """

    frame_a: UnwrapResult
    frame_b: UnwrapResult
    aligned: bool
    shift: Optional[int]
    shared_pixels: tuple[tuple[int, int], ...]
    squared_error: Optional[float]


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
def _unwrap_validated(w, m, height, width, search_budget: int) -> UnwrapResult:
    """输入已经整体校验并复制为 float/bool 网格后的空间展开主体。"""
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


def unwrap_phase(phase, mask, *, search_budget: int = DEFAULT_SEARCH_BUDGET) -> UnwrapResult:
    """残差切口相位展开（纯函数：不修改 phase/mask）。

    非法输入抛 InvalidPhaseInputError（带 row/col）；搜索预算耗尽抛
    SearchBudgetExhausted；剩余残差证明无法到达外边界抛 NoSolutionError；
    环和不是 2π 整数倍抛 UnwrappingContradictionError。
    """
    w, m, height, width = _validate(phase, mask, search_budget)
    return _unwrap_validated(w, m, height, width, search_budget)


# --------------------------------------------------------------------------
# 相邻两帧：全局整数周对齐
# --------------------------------------------------------------------------
def _sse_for_shift(diffs: list[float], k: int) -> float:
    """Σ(d - 2π·k)²，d = phi_a - phi_b，仅遍历共同有效像素一次。"""
    total = 0.0
    for d in diffs:
        e = d - k * TWO_PI
        total += e * e
    return total


def align_frames(
    phase_a,
    mask_a,
    phase_b,
    mask_b,
    *,
    search_budget: int = DEFAULT_SEARCH_BUDGET,
) -> FrameAlignment:
    """相邻两帧：各自残差切口空间展开后，用共同有效像素做全局整数周对齐。

    步骤（先旧空间展开，再全局对齐）：
    1. 两帧分别独立调用与 unwrap_phase 完全相同的空间流程（沿用旧数据结构
       与错误语义）；
    2. 取两帧 mask 均为 True 的共同有效像素；无共同像素 -> aligned=False，
       shift=None, squared_error=None, shared_pixels=()，两帧结果原样保留；
    3. 有共同像素时，d_i = phi_a_i - phi_b_i，选整数 k 最小化
       Σ(d_i - 2π·k)²。该 SSE 关于 k 是严格凸二次函数，唯一连续最优点为
       mean(d)/2π，故整数最优只需比较 floor 与 floor+1；平方差严格更小才
       取 +1，并列（含理论上的均值恰为半点）取较小整数。

    为避免半更新状态：两帧输入先整体校验（shape/类型/数值/budget 一次
    查完），通过后才开始任何展开计算；任何一帧失败（含预算耗尽、证明
    无解）都直接抛异常，不返回部分 FrameAlignment。纯函数，不修改输入。
    """
    grid_a, boolmask_a, ha, wa_w = _validate(phase_a, mask_a, search_budget)
    grid_b, boolmask_b, hb, wb_w = _validate(phase_b, mask_b, search_budget)
    if (ha, wa_w) != (hb, wb_w):
        raise InvalidPhaseInputError(
            f"frame shapes differ: frame_a is {ha}x{wa_w}, "
            f"frame_b is {hb}x{wb_w}"
        )
    height, width = ha, wa_w

    # 校验全部通过后才计算；任一帧展开失败即整体失败（无部分结果）。
    result_a = _unwrap_validated(grid_a, boolmask_a, height, width, search_budget)
    result_b = _unwrap_validated(grid_b, boolmask_b, height, width, search_budget)

    shared: list[tuple[int, int]] = []
    diffs: list[float] = []
    for r in range(height):
        for c in range(width):
            if boolmask_a[r][c] and boolmask_b[r][c]:
                shared.append((r, c))
                diffs.append(result_a.unwrapped[r][c] - result_b.unwrapped[r][c])

    if not shared:
        return FrameAlignment(
            frame_a=result_a,
            frame_b=result_b,
            aligned=False,
            shift=None,
            shared_pixels=(),
            squared_error=None,
        )

    # 连续最优 k* = mean(d)/2π；凸二次只需枚举 floor(k*) 与 floor(k*)+1。
    n = len(diffs)
    mean_k = sum(diffs) / (n * TWO_PI)
    k0 = math.floor(mean_k)
    sse0 = _sse_for_shift(diffs, k0)
    sse1 = _sse_for_shift(diffs, k0 + 1)
    # 严格更小才取较大整数；相等（浮点口径内的并列）取较小整数。
    shift = k0 if sse0 <= sse1 else k0 + 1
    sse = sse0 if shift == k0 else sse1

    return FrameAlignment(
        frame_a=result_a,
        frame_b=result_b,
        aligned=True,
        shift=shift,
        shared_pixels=tuple(shared),
        squared_error=sse,
    )
