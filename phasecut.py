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
    "UnwrapResult",
    "PhaseCutError",
    "InvalidPhaseInputError",
    "SearchBudgetExhausted",
    "NoSolutionError",
    "UnwrappingContradictionError",
    "wrap_phase",
    "unwrap_phase",
    "solve_two_frequency",
    "enumerate_week_pairs",
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


class WeekCandidate(NamedTuple):
    """双频绝对相位的一个可行候选。

    n1/n2 为两种频率各自的整数周数：观测 i 的绝对相位为
    wrapped_i + 2*pi*n_i，对应公共绝对相位
    Phi = (wrapped_1 + 2*pi*n1)/freq1 = (wrapped_2 + 2*pi*n2)/freq2
    （误差带内）。[lo, hi] 为 Phi 在两条观测误差带与给定搜索区间
    交集下仍可行的闭区间；恰接触（lo == hi）也算可行。
    """

    n1: int
    n2: int
    lo: float
    hi: float

    @property
    def center(self) -> float:
        return 0.5 * (self.lo + self.hi)


class TwoFreqResult(NamedTuple):
    """solve_two_frequency 的返回结果（不可变）。

    candidates[r][c]：该像素全部可行 WeekCandidate（按 lo 排序），
    mask 无效位置为 None。counts[r][c] 为候选数（无效位置 None）。
    solved/ambiguous/empty_pixels 分别为候选数 ==1 / >1 / ==0 的
    有效像素数；歧义候选数如实输出，不挑最接近 0 的候选。
    filtered 仅在 neighbor_filter=True 时给出：保留与每个有效四邻
    至少一个候选区间相交的候选；它只是显式筛选视图，counts 与
    歧义统计始终基于未筛选的 candidates。
    """

    freq1: int
    freq2: int
    phi_min: float
    phi_max: float
    candidates: tuple[tuple[Optional[tuple[WeekCandidate, ...]], ...], ...]
    counts: tuple[tuple[Optional[int], ...], ...]
    solved_pixels: int
    ambiguous_pixels: int
    empty_pixels: int
    filtered: Optional[tuple[tuple[Optional[tuple[WeekCandidate, ...]], ...]]]


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
# 双频绝对周数求解
# --------------------------------------------------------------------------
def _validate_two_freq(
    phase1, phase2, mask, freq1, freq2, error1, error2, phi_min, phi_max,
):
    for name, f in (("freq1", freq1), ("freq2", freq2)):
        if not isinstance(f, int) or isinstance(f, bool):
            raise InvalidPhaseInputError(
                f"{name} must be a positive int; bool is not accepted"
            )
        if f < 1:
            raise InvalidPhaseInputError(f"{name} must be >= 1")
    if math.gcd(freq1, freq2) != 1:
        raise InvalidPhaseInputError(
            f"freq1={freq1} and freq2={freq2} must be coprime"
        )
    for name, e in (("error1", error1), ("error2", error2)):
        if isinstance(e, bool) or not isinstance(e, (int, float)):
            raise InvalidPhaseInputError(
                f"{name} must be a real number; bool is not accepted"
            )
        if math.isnan(float(e)) or math.isinf(float(e)) or float(e) < 0.0:
            raise InvalidPhaseInputError(
                f"{name} must be finite and >= 0"
            )
    for name, b in (("phi_min", phi_min), ("phi_max", phi_max)):
        if isinstance(b, bool) or not isinstance(b, (int, float)):
            raise InvalidPhaseInputError(
                f"{name} must be a real number; bool is not accepted"
            )
        if math.isnan(float(b)) or math.isinf(float(b)):
            raise InvalidPhaseInputError(f"{name} must be finite")
    if not float(phi_min) < float(phi_max):
        raise InvalidPhaseInputError(
            f"phi_min ({phi_min!r}) must be < phi_max ({phi_max!r})"
        )
    if isinstance(phase2, (str, bytes)) or not isinstance(phase2, (list, tuple)):
        raise InvalidPhaseInputError("phase2 must be a sequence of numeric rows")
    if len(phase2) != len(phase1):
        raise InvalidPhaseInputError(
            f"phase2 has {len(phase2)} rows but phase1 has {len(phase1)}"
        )
    return float(error1), float(error2), float(phi_min), float(phi_max)


def _week_candidates_pixel(
    w1: float,
    w2: float,
    k1: int,
    k2: int,
    e1: float,
    e2: float,
    lo: float,
    hi: float,
) -> tuple[WeekCandidate, ...]:
    """单像素：先用周期关系缩小 (n1, n2)，再逐一代回两种频率检验。

    观测 i 给出 Phi 的可行带 (wrapped_i + 2*pi*n_i)/k_i 加减 e_i/k_i。
    两带一致要求 k2*n1 - k1*n2 = r，其中 r 为整数且
    |r - t| <= (k2*e1 + k1*e2)/(2*pi)，t = (k1*w2 - k2*w1)/(2*pi)。
    对每个可行 r 用 CRT（k1, k2 互素）把 n1 钉在一个 mod k1 剩余类上，
    只枚举该剩余类落在区间内的 n1，再逐一代回两种频率求闭区间交集。
    """
    t = (k1 * w2 - k2 * w1) / TWO_PI
    slack = (k2 * e1 + k1 * e2) / TWO_PI
    # 区间搜索余量：仅容纳浮点散布，不改变数学判定。
    eps = 1e-9
    r_lo = math.ceil(t - slack - eps)
    r_hi = math.floor(t + slack + eps)
    if r_lo > r_hi:
        return ()
    inv_k2 = pow(k2 % k1, -1, k1)  # k2 在 mod k1 下的逆元（互素保证存在）
    n1_min = math.ceil((k1 * lo - e1 - w1) / TWO_PI - eps)
    n1_max = math.floor((k1 * hi + e1 - w1) / TWO_PI + eps)
    out: list[WeekCandidate] = []
    for r in range(r_lo, r_hi + 1):
        # k2*n1 ≡ r (mod k1)  →  n1 ≡ r*inv_k2 (mod k1)
        base = (r * inv_k2) % k1
        if base < n1_min:
            base += k1 * ((n1_min - base + k1 - 1) // k1)
        # 归一化到 [n1_min, n1_min + k1)：负周数也必须覆盖。
        base -= k1 * ((base - n1_min) // k1)
        for n1 in range(base, n1_max + 1, k1):
            num = k2 * n1 - r
            if num % k1 != 0:
                continue  # 防御：CRT 已保证整除
            n2 = num // k1
            c_lo = max(lo, (w1 + TWO_PI * n1 - e1) / k1,
                       (w2 + TWO_PI * n2 - e2) / k2)
            c_hi = min(hi, (w1 + TWO_PI * n1 + e1) / k1,
                       (w2 + TWO_PI * n2 + e2) / k2)
            if c_lo <= c_hi + eps:
                out.append(WeekCandidate(n1, n2, c_lo, max(c_lo, c_hi)))
    out.sort(key=lambda c: (c.lo, c.hi, c.n1, c.n2))
    return tuple(out)


def enumerate_week_pairs(
    w1: float,
    w2: float,
    freq1: int,
    freq2: int,
    error1: float,
    error2: float,
    phi_min: float,
    phi_max: float,
) -> tuple[WeekCandidate, ...]:
    """短范围穷举参考：直接枚举 (n1, n2) 整数对逐一求交集。

    与 solve_two_frequency 的 CRT 路径互相独立，供测试对照；
    只适合区间/频率都很小的情形。
    """
    e1, e2 = float(error1), float(error2)
    eps = 1e-9
    n1_min = math.ceil((freq1 * phi_min - e1 - w1) / TWO_PI - eps)
    n1_max = math.floor((freq1 * phi_max + e1 - w1) / TWO_PI + eps)
    n2_min = math.ceil((freq2 * phi_min - e2 - w2) / TWO_PI - eps)
    n2_max = math.floor((freq2 * phi_max + e2 - w2) / TWO_PI + eps)
    out: list[WeekCandidate] = []
    for n1 in range(n1_min, n1_max + 1):
        for n2 in range(n2_min, n2_max + 1):
            c_lo = max(phi_min, (w1 + TWO_PI * n1 - e1) / freq1,
                       (w2 + TWO_PI * n2 - e2) / freq2)
            c_hi = min(phi_max, (w1 + TWO_PI * n1 + e1) / freq1,
                       (w2 + TWO_PI * n2 + e2) / freq2)
            if c_lo <= c_hi + eps:
                out.append(WeekCandidate(n1, n2, c_lo, max(c_lo, c_hi)))
    out.sort(key=lambda c: (c.lo, c.hi, c.n1, c.n2))
    return tuple(out)


def solve_two_frequency(
    phase1,
    phase2,
    mask,
    freq1: int,
    freq2: int,
    error1: float,
    error2: float,
    phi_min: float,
    phi_max: float,
    *,
    neighbor_filter: bool = False,
) -> TwoFreqResult:
    """两幅同位置包裹相位 + 互素频率，求有限区间内全部一致解。

    phase_i 为频率 freq_i 下观测到的包裹相位（[-pi, pi)），
    error_i 为该观测独立的误差上限（弧度，>=0）。对每个有效像素
    返回 [phi_min, phi_max] 内所有可行 WeekCandidate；不存在的像素
    候选为空元组（如实报告，不抛异常、不挑最接近 0 的候选）。
    mask 为 False 的位置不参与任何计算与邻域筛选。
    neighbor_filter=True 时额外给出 filtered 视图（见 TwoFreqResult），
    它只做显式筛选，绝不把本来多解的点宣布唯一。
    """
    e1, e2, lo, hi = _validate_two_freq(
        phase1, phase2, mask, freq1, freq2, error1, error2, phi_min, phi_max,
    )
    # 复用单幅接口的结构/数值校验（search_budget 用默认值，不展开）。
    w1, m, height, width = _validate(phase1, mask, DEFAULT_SEARCH_BUDGET)
    w2, _m2, h2, w2d = _validate(phase2, mask, DEFAULT_SEARCH_BUDGET)
    assert (h2, w2d) == (height, width)

    cand_rows: list[list[Optional[tuple[WeekCandidate, ...]]]] = []
    count_rows: list[list[Optional[int]]] = []
    solved = ambiguous = empty = 0
    for r in range(height):
        cand_row: list[Optional[tuple[WeekCandidate, ...]]] = []
        count_row: list[Optional[int]] = []
        for c in range(width):
            if not m[r][c]:
                cand_row.append(None)
                count_row.append(None)
                continue
            cands = _week_candidates_pixel(
                w1[r][c], w2[r][c], freq1, freq2, e1, e2, lo, hi,
            )
            cand_row.append(cands)
            count_row.append(len(cands))
            if len(cands) == 1:
                solved += 1
            elif len(cands) > 1:
                ambiguous += 1
            else:
                empty += 1
        cand_rows.append(cand_row)
        count_rows.append(count_row)

    filtered = None
    if neighbor_filter:
        filtered_rows: list[list[Optional[tuple[WeekCandidate, ...]]]] = []
        for r in range(height):
            f_row: list[Optional[tuple[WeekCandidate, ...]]] = []
            for c in range(width):
                cands = cand_rows[r][c]
                if cands is None:
                    f_row.append(None)
                    continue
                kept = list(cands)
                for dr, dc in _DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    neigh = cand_rows[nr][nc]
                    if neigh is None:
                        continue  # 掩码位置不参与筛选
                    if not neigh:
                        continue  # 邻像素自身无可行解时不据此筛除
                    kept = [
                        cand
                        for cand in kept
                        if any(
                            cand.lo <= other.hi and other.lo <= cand.hi
                            for other in neigh
                        )
                    ]
                f_row.append(tuple(kept))
            filtered_rows.append(f_row)
        filtered = tuple(tuple(row) for row in filtered_rows)

    return TwoFreqResult(
        freq1=freq1,
        freq2=freq2,
        phi_min=lo,
        phi_max=hi,
        candidates=tuple(tuple(row) for row in cand_rows),
        counts=tuple(tuple(row) for row in count_rows),
        solved_pixels=solved,
        ambiguous_pixels=ambiguous,
        empty_pixels=empty,
        filtered=filtered,
    )
