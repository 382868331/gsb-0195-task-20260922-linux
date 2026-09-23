"""Core algorithm: residual branch-cut phase unwrapping on small masked grids.

Discrete model (see README for the prose specification):

* Pixels are primal vertices. A *face* is a 2x2 pixel loop, indexed by the
  coordinate of its top-left pixel ``(r, c)``; it carries the integer
  residue ``q``. The grid border is one additional *outer face*.
* Two faces sharing a pixel edge are dual neighbours; routing a dual path
  between two charged faces crosses one primal pixel edge per dual step and
  every crossed edge becomes a branch cut (an edge the flood fill may not
  cross). A leash leaves the last border face through one of its grid-border
  pixel edges. This dual routing is what makes a cut actually discharge the
  charged loop: the first/last crossed edge lies on that loop's perimeter.
* Charged faces are paired, in lexicographic order, with the lexicographically
  first shortest Manhattan dual path, never reusing a cut edge and never
  passing through another charged face. Unpaired leftovers are leashed to the
  outer face; a charge that is provably unreachable from both an opposite
  charge and the border is returned in ``unsatisfied`` instead of being
  hidden behind a smooth-looking field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

__all__ = [
    "MAX_SIDE",
    "TAU",
    "CONTRADICTION_TOL",
    "Edge",
    "Residue",
    "Dipole",
    "BoundaryLeash",
    "Component",
    "UnsatisfiedResidue",
    "UnwrapResult",
    "PhaseCutError",
    "InputValidationError",
    "UnwrapInfeasibleError",
    "SearchBudgetExhaustedError",
    "wrapped_difference",
    "residue_map",
    "unwrap",
]

MAX_SIDE = 32
TAU = 2.0 * math.pi
# Edge contradictions (unwrapped difference minus wrapped difference) above
# this absolute magnitude, in radians, are reported instead of being ignored.
# Well above double-precision accumulation error on a <= 32x32 grid (<~1e-13)
# and six orders of magnitude below a single 2*pi branch jump.
CONTRADICTION_TOL = 1e-9

# Fixed move order used by every BFS (up, left, right, down).
_MOVES = ((-1, 0), (0, -1), (0, 1), (1, 0))
_SIDE_NAMES = ("up", "left", "right", "down")


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class PhaseCutError(Exception):
    """Base class for all phasecut errors."""


class InputValidationError(PhaseCutError, ValueError):
    """An input violates the contract; the message pinpoints the location."""

    def __init__(self, message: str, loc: Optional[str] = None) -> None:
        self.loc = loc
        super().__init__(f"{loc}: {message}" if loc else message)


class UnwrapInfeasibleError(PhaseCutError):
    """Exhaustive search proved no pairing/leash route exists.

    Distinct from a budget cutoff: reachability really was disproved.
    """

    def __init__(self, message: str, unreachable: Sequence[tuple[int, int]] = ()) -> None:
        self.unreachable = tuple(unreachable)
        super().__init__(message)


class SearchBudgetExhaustedError(PhaseCutError):
    """The BFS expansion budget ran out before reachability was decided.

    No claim is made about whether a route exists.
    """


# --------------------------------------------------------------------------- #
# Public data structures
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Edge:
    """An undirected pixel-pixel edge with lexicographically ordered ends.

    Both ends are always valid pixel coordinates; a boundary leash "exits"
    the grid by cutting a pixel edge that lies on the grid border (e.g. the
    edge between the two pixels of the top row of a top-row face).
    """

    a: tuple[int, int]
    b: tuple[int, int]

    def __post_init__(self) -> None:
        if self.b < self.a:
            lo, hi = self.b, self.a
            object.__setattr__(self, "a", lo)
            object.__setattr__(self, "b", hi)

    def as_key(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return (self.a, self.b)


@dataclass(frozen=True)
class Residue:
    """A nonzero loop charge at the 2x2 face with top-left (row, col)."""

    row: int
    col: int
    charge: int
    wrapped_sum: float


@dataclass(frozen=True)
class Dipole:
    """A paired +1/-1 residue pair and the cut path joining their faces.

    ``path`` are the dual faces from positive to negative face; ``edges`` are
    the primal pixel edges crossed by that path (length ``len(path) - 1``).
    """

    positive: tuple[int, int]
    negative: tuple[int, int]
    path: tuple[tuple[int, int], ...]
    edges: tuple[Edge, ...]


@dataclass(frozen=True)
class BoundaryLeash:
    """A residual charge connected to the outer face across the grid border.

    ``path`` are the dual faces from the charged face to the border face;
    ``exit`` is the grid-border pixel edge that finally leaves the grid
    (both endpoints are real border pixels). ``edges`` are the interior cut
    edges followed by ``exit``.
    """

    source: tuple[int, int]
    charge: int
    path: tuple[tuple[int, int], ...]
    edges: tuple[Edge, ...]
    exit: Edge


@dataclass(frozen=True)
class Component:
    """A connected valid-pixel region after cuts.

    The seed is the lexicographically smallest pixel and its integer-sheet
    offset is 0; every other offset is relative to that seed. ``cells`` are
    sorted lexicographically and line up with ``offsets``/``unwrapped``.
    """

    seed: tuple[int, int]
    cells: tuple[tuple[int, int], ...]
    offsets: tuple[float, ...]
    unwrapped: tuple[float, ...]


@dataclass(frozen=True)
class UnsatisfiedResidue:
    """A charged face with no cut edge on its perimeter.

    The integrated field around such a vortex is internally contradictory:
    ``edge_residuals`` (unwrapped difference minus wrapped difference, in
    top/right/bottom/left loop order) sum to ``-2*pi*charge``; one entry hides
    a full 2*pi jump. The contradiction is reported, never smoothed over.
    """

    row: int
    col: int
    charge: int
    edges: tuple[Edge, ...]
    edge_residuals: tuple[float, ...]


@dataclass(frozen=True)
class EdgeResidual:
    """A present, non-cut pixel edge whose integrated residual is nonzero.

    ``residual`` = unwrapped difference minus wrapped difference; for a
    consistent sheet it is 0, while a hidden branch jump is an integer
    multiple of 2*pi. These are the "unsatisfied edge residuals": edges whose
    contradiction cannot be charged to a valid 2x2 face (typically because the
    winding hides behind masked-out pixels) or that sit on an unsatisfied
    charged face's perimeter.
    """

    edge: Edge
    residual: float


@dataclass(frozen=True)
class UnwrapResult:
    rows: int
    cols: int
    wrapped: tuple[tuple[float, ...], ...]
    mask: tuple[tuple[bool, ...], ...]
    residues: tuple[Residue, ...]
    residue_grid: tuple[tuple[int, ...], ...]
    dipoles: tuple[Dipole, ...]
    boundary_leashes: tuple[BoundaryLeash, ...]
    cuts: tuple[Edge, ...]
    components: tuple[Component, ...]
    offsets_grid: tuple[tuple[Optional[float], ...], ...]
    unwrapped_grid: tuple[tuple[Optional[float], ...], ...]
    unsatisfied: tuple[UnsatisfiedResidue, ...]
    edge_residuals: tuple[EdgeResidual, ...]


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def wrapped_difference(delta: float) -> float:
    """Wrap a phase difference into the half-open interval ``[-pi, pi)``."""
    return (delta + math.pi) % TAU - math.pi


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _is_real_number(value: object) -> bool:
    # bool is a subclass of int; a boolean is not a phase sample.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_phase(
    phase: Sequence[Sequence[object]],
) -> tuple[int, int, list[list[float]]]:
    if not isinstance(phase, (list, tuple)):
        raise InputValidationError(
            "phase must be a list/tuple of rows, got %s"
            % type(phase).__name__,
            "phase",
        )
    if len(phase) == 0:
        raise InputValidationError("phase must have at least one row", "phase")
    if len(phase) > MAX_SIDE:
        raise InputValidationError(
            "phase has %d rows, at most %d allowed" % (len(phase), MAX_SIDE),
            "phase",
        )

    width: Optional[int] = None
    out: list[list[float]] = []
    for r, row in enumerate(phase):
        loc = "phase[%d]" % r
        if not isinstance(row, (list, tuple)):
            raise InputValidationError(
                "row must be a list/tuple, got %s" % type(row).__name__, loc
            )
        if len(row) == 0:
            raise InputValidationError("row must not be empty", loc)
        if width is None:
            width = len(row)
            if width > MAX_SIDE:
                raise InputValidationError(
                    "phase has %d columns, at most %d allowed"
                    % (width, MAX_SIDE),
                    "phase[%d]" % r,
                )
        elif len(row) != width:
            raise InputValidationError(
                "row length %d != %d" % (len(row), width), loc
            )
        out_row: list[float] = []
        for c, value in enumerate(row):
            vloc = "phase[%d][%d]" % (r, c)
            if not _is_real_number(value):
                raise InputValidationError(
                    "value must be a finite real number (int/float; bool, NaN "
                    "and Infinity are rejected), got %r" % (value,),
                    vloc,
                )
            fv = float(value)  # type: ignore[arg-type]
            if not math.isfinite(fv):
                raise InputValidationError(
                    "value must be finite, got %r" % (value,), vloc
                )
            if fv < -math.pi or fv >= math.pi:
                raise InputValidationError(
                    "wrapped phase must lie in [-pi, pi), got %.17g" % fv,
                    vloc,
                )
            out_row.append(fv)
        out.append(out_row)
    assert width is not None
    return len(out), width, out


def _validate_mask(
    mask: Optional[Sequence[Sequence[object]]],
    rows: int,
    cols: int,
) -> list[list[bool]]:
    if mask is None:
        return [[True] * cols for _ in range(rows)]
    if not isinstance(mask, (list, tuple)):
        raise InputValidationError(
            "mask must be a list/tuple of rows, got %s" % type(mask).__name__,
            "mask",
        )
    if len(mask) != rows:
        raise InputValidationError(
            "mask has %d rows, phase has %d" % (len(mask), rows), "mask"
        )
    out: list[list[bool]] = []
    for r, row in enumerate(mask):
        loc = "mask[%d]" % r
        if not isinstance(row, (list, tuple)):
            raise InputValidationError(
                "row must be a list/tuple, got %s" % type(row).__name__, loc
            )
        if len(row) != cols:
            raise InputValidationError(
                "mask row length %d != phase width %d" % (len(row), cols), loc
            )
        out_row: list[bool] = []
        for c, value in enumerate(row):
            vloc = "mask[%d][%d]" % (r, c)
            if isinstance(value, bool):
                out_row.append(value)
            elif isinstance(value, int) and not isinstance(value, bool) and value in (0, 1):
                out_row.append(bool(value))
            else:
                raise InputValidationError(
                    "mask entries must be bool or 0/1, got %r" % (value,),
                    vloc,
                )
        out.append(out_row)
    return out


def _validate_budget(max_expansions: object) -> int:
    if isinstance(max_expansions, bool) or not isinstance(max_expansions, int):
        raise InputValidationError(
            "max_expansions must be an int (bool rejected), got %r"
            % (max_expansions,),
            "max_expansions",
        )
    if max_expansions < 1:
        raise InputValidationError(
            "max_expansions must be >= 1, got %d" % max_expansions,
            "max_expansions",
        )
    return max_expansions


def _validate_on_infeasible(value: object) -> str:
    if value not in ("report", "raise"):
        raise InputValidationError(
            'on_infeasible must be "report" or "raise", got %r' % (value,),
            "on_infeasible",
        )
    return value  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Residues
# --------------------------------------------------------------------------- #
def _face_perimeter(
    r: int, c: int
) -> tuple[tuple[int, int], ...]:
    """Primal loop order around face (r,c): top, right, bottom, left."""
    a, b = (r, c), (r, c + 1)
    d, e = (r + 1, c), (r + 1, c + 1)
    return (a, b, e, d)


def _compute_residues(
    phase: list[list[float]], valid: list[list[bool]]
) -> tuple[list[list[int]], list[Residue], set[tuple[int, int]]]:
    rows = len(phase)
    cols = len(phase[0])
    grid = [[0] * cols for _ in range(rows)]
    residues: list[Residue] = []
    faces: set[tuple[int, int]] = set()
    for r in range(rows - 1):
        for c in range(cols - 1):
            corners = _face_perimeter(r, c)
            if not all(valid[i][j] for (i, j) in corners):
                continue
            faces.add((r, c))
            (a, b, e, d) = corners
            s = (
                wrapped_difference(phase[b[0]][b[1]] - phase[a[0]][a[1]])
                + wrapped_difference(phase[e[0]][e[1]] - phase[b[0]][b[1]])
                + wrapped_difference(phase[d[0]][d[1]] - phase[e[0]][e[1]])
                + wrapped_difference(phase[a[0]][a[1]] - phase[d[0]][d[1]])
            )
            q = int(math.floor(s / TAU + 0.5))
            if q != 0:
                grid[r][c] = q
                residues.append(Residue(r, c, q, s))
    return grid, residues, faces


def residue_map(
    phase: Sequence[Sequence[object]],
    mask: Optional[Sequence[Sequence[object]]] = None,
) -> tuple[tuple[int, ...], ...]:
    """Return the integer residue of every 2x2 face.

    Faces that are invalid (not all four pixels valid) or non-charged are
    reported as 0. Raises :class:`InputValidationError` on bad input.
    """
    rows, cols, fphase = _validate_phase(phase)
    valid = _validate_mask(mask, rows, cols)
    grid, _residues, _faces = _compute_residues(fphase, valid)
    return tuple(tuple(row) for row in grid)


# --------------------------------------------------------------------------- #
# Dual graph: faces are vertices, crossed primal edges are cut edges
# --------------------------------------------------------------------------- #
def _crossed_edge(
    face: tuple[int, int], move: tuple[int, int]
) -> Edge:
    """Primal pixel edge crossed when stepping ``move`` away from ``face``."""
    r, c = face
    dr, dc = move
    if dr == -1:  # walk up: cross face's top side
        return Edge((r, c), (r, c + 1))
    if dr == 1:  # walk down: cross the bottom side
        return Edge((r + 1, c), (r + 1, c + 1))
    if dc == -1:  # walk left
        return Edge((r, c), (r + 1, c))
    return Edge((r, c + 1), (r + 1, c + 1))  # walk right


def _border_exit(face: tuple[int, int], rows: int, cols: int, side: str) -> Edge:
    r, c = face
    if side == "up":
        return Edge((0, c), (0, c + 1))
    if side == "down":
        return Edge((rows - 1, c), (rows - 1, c + 1))
    if side == "left":
        return Edge((r, 0), (r + 1, 0))
    return Edge((r, cols - 1), (r + 1, cols - 1))  # right


class _DualGraph:
    def __init__(
        self,
        rows: int,
        cols: int,
        faces: set[tuple[int, int]],
        charged: set[tuple[int, int]],
        max_expansions: int,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.faces = faces
        self.charged = charged
        self.max_expansions = max_expansions
        self.expansions = 0
        self.blocked_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    def add_cut(self, edge: Edge) -> None:
        self.blocked_edges.add(edge.as_key())

    def _step(
        self,
        face: tuple[int, int],
        move: tuple[int, int],
    ) -> Optional[tuple[tuple[int, int], Edge]]:
        edge = _crossed_edge(face, move)
        if edge.as_key() in self.blocked_edges:
            return None
        nxt = (face[0] + move[0], face[1] + move[1])
        if nxt not in self.faces:
            return None
        return nxt, edge

    def _expand(
        self,
        frontier: list[tuple[int, int]],
        parent: dict[tuple[int, int], tuple[int, int]],
        blocked_cells: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """One BFS layer. First writer wins; cells are claimed lexicographically."""
        claimed: dict[tuple[int, int], tuple[int, int]] = {}
        for cell in sorted(frontier):
            self.expansions += 1
            if self.expansions > self.max_expansions:
                raise SearchBudgetExhaustedError(
                    "BFS expansion budget (%d) exhausted before reachability "
                    "could be decided; no conclusion about feasibility follows"
                    % self.max_expansions
                )
            for move in _MOVES:
                step = self._step(cell, move)
                if step is None:
                    continue
                nxt, _edge = step
                if nxt in parent or nxt in claimed or nxt in blocked_cells:
                    continue
                claimed[nxt] = cell
        parent.update(claimed)
        return sorted(claimed)

    @staticmethod
    def _reconstruct(
        parent: dict[tuple[int, int], tuple[int, int]],
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> tuple[tuple[int, int], ...]:
        path = [goal]
        while path[-1] != start:
            path.append(parent[path[-1]])
        path.reverse()
        return tuple(path)

    def shortest_pair(
        self,
        start: tuple[int, int],
        goals: set[tuple[int, int]],
    ) -> Optional[tuple[tuple[int, int], tuple[tuple[int, int], ...]]]:
        """Shortest dual route to any goal face (goals may be entered).

        Other charged faces cannot be used as transit. Among equal-length
        routes the lexicographically smallest (goal, path) wins. ``None``
        means every reachable face was examined and no goal was found.
        """
        if start in goals:
            return start, (start,)
        if not goals:
            return None
        blocked = set(self.charged)
        blocked.discard(start)
        blocked -= goals
        parent: dict[tuple[int, int], tuple[int, int]] = {start: start}
        frontier = [start]
        while frontier:
            frontier = self._expand(frontier, parent, blocked)
            for cell in frontier:
                if cell in goals:
                    return cell, self._reconstruct(parent, start, cell)
        return None

    def shortest_leash(
        self, start: tuple[int, int]
    ) -> Optional[tuple[tuple[int, int], ..., Edge]]:
        """Shortest dual route leaving the grid through a border edge.

        Returns (faces path including the border face, exit edge). Ties are
        broken by the smallest border face and then up/left/right/down.
        ``None`` means the outer face was proved unreachable.
        """
        blocked = set(self.charged)
        blocked.discard(start)
        parent: dict[tuple[int, int], tuple[int, int]] = {start: start}
        frontier = [start]
        while frontier:
            best: Optional[tuple[tuple[int, int], str]] = None
            for cell in sorted(frontier):
                for side in _SIDE_NAMES:
                    move = {
                        "up": (-1, 0),
                        "left": (0, -1),
                        "right": (0, 1),
                        "down": (1, 0),
                    }[side]
                    r, c = cell
                    on_border = (
                        (side == "up" and r == 0)
                        or (side == "down" and r == self.rows - 2)
                        or (side == "left" and c == 0)
                        or (side == "right" and c == self.cols - 2)
                    )
                    if not on_border:
                        continue
                    edge = _border_exit(cell, self.rows, self.cols, side)
                    if edge.as_key() in self.blocked_edges:
                        continue
                    if best is None or (cell, _SIDE_NAMES.index(side)) < (
                        best[0],
                        _SIDE_NAMES.index(best[1]),
                    ):
                        best = (cell, side)
            if best is not None:
                cell, side = best
                return (
                    self._reconstruct(parent, start, cell),
                    _border_exit(cell, self.rows, self.cols, side),
                )
            frontier = self._expand(frontier, parent, blocked)
        return None


def _path_edges(
    path: Sequence[tuple[int, int]],
) -> tuple[Edge, ...]:
    return tuple(
        _crossed_edge(path[i], (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]))
        for i in range(len(path) - 1)
    )


# --------------------------------------------------------------------------- #
# Cut placement
# --------------------------------------------------------------------------- #
def _lay_cuts(
    residues: list[Residue],
    graph: _DualGraph,
    on_infeasible: str,
) -> tuple[list[Dipole], list[BoundaryLeash], set[Edge], list[tuple[int, int]]]:
    paired: set[tuple[int, int]] = set()
    dipoles: list[Dipole] = []
    leashes: list[BoundaryLeash] = []
    cut_edges: set[Edge] = set()
    unreachable: list[tuple[int, int]] = []

    def commit_path_edges(edges: Sequence[Edge]) -> None:
        for edge in edges:
            graph.add_cut(edge)
            cut_edges.add(edge)

    # ---- pairing pass, lexicographic residue order ----------------------- #
    for res in residues:
        start = (res.row, res.col)
        if start in paired:
            continue
        goals = {
            (other.row, other.col)
            for other in residues
            if other.charge == -res.charge and (other.row, other.col) not in paired
        }
        found = graph.shortest_pair(start, goals)
        if found is None:
            continue  # try the border later
        goal, path = found
        edges = _path_edges(path)
        commit_path_edges(edges)
        paired.add(start)
        paired.add(goal)
        if res.charge > 0:
            dipoles.append(Dipole(start, goal, path, edges))
        else:
            dipoles.append(Dipole(goal, start, tuple(reversed(path)), edges))

    # ---- boundary pass for leftovers ------------------------------------- #
    for res in residues:
        start = (res.row, res.col)
        if start in paired:
            continue
        found = graph.shortest_leash(start)
        if found is None:
            unreachable.append(start)  # proved: no opposite charge, no border
            continue
        path, exit_edge = found
        edges = _path_edges(path)
        commit_path_edges(edges)
        graph.add_cut(exit_edge)
        cut_edges.add(exit_edge)
        paired.add(start)
        leashes.append(
            BoundaryLeash(
                start, res.charge, path, edges + (exit_edge,), exit_edge
            )
        )

    if on_infeasible == "raise" and unreachable:
        raise UnwrapInfeasibleError(
            "residue face(s) %s cannot be paired with an opposite charge or "
            "leashed to the outer border through valid faces without crossing "
            "an existing cut" % (unreachable,),
            unreachable,
        )
    return dipoles, leashes, cut_edges, unreachable


# --------------------------------------------------------------------------- #
# Flood unwrap on the primal pixel graph
# --------------------------------------------------------------------------- #
def _flood_components(
    phase: list[list[float]],
    valid: list[list[bool]],
    graph: _DualGraph,
) -> tuple[list[Component], list[list[Optional[float]]], list[list[Optional[float]]]]:
    rows, cols = graph.rows, graph.cols
    blocked = graph.blocked_edges
    seen = [[False] * cols for _ in range(rows)]
    offsets_grid: list[list[Optional[float]]] = [[None] * cols for _ in range(rows)]
    unwrapped_grid: list[list[Optional[float]]] = [
        [None] * cols for _ in range(rows)
    ]
    components: list[Component] = []

    for sr in range(rows):
        for sc in range(cols):
            if not valid[sr][sc] or seen[sr][sc]:
                continue
            seed = (sr, sc)  # lexicographic scan == smallest pixel in component
            seen[sr][sc] = True
            offsets_grid[sr][sc] = 0.0
            unwrapped_grid[sr][sc] = phase[sr][sc]
            frontier = [seed]
            cells: list[tuple[int, int]] = [seed]
            while frontier:
                nxt: list[tuple[int, int]] = []
                for cell in sorted(frontier):
                    r, c = cell
                    for dr, dc in _MOVES:
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < rows and 0 <= nc < cols):
                            continue
                        if not valid[nr][nc] or seen[nr][nc]:
                            continue
                        if Edge(cell, (nr, nc)).as_key() in blocked:
                            continue
                        seen[nr][nc] = True
                        raw = phase[nr][nc] - phase[r][c]
                        off = offsets_grid[r][c] + wrapped_difference(raw) - raw
                        offsets_grid[nr][nc] = off
                        unwrapped_grid[nr][nc] = phase[nr][nc] + off
                        child = (nr, nc)
                        cells.append(child)
                        nxt.append(child)
                frontier = nxt
            cells.sort()
            components.append(
                Component(
                    seed=seed,
                    cells=tuple(cells),
                    offsets=tuple(offsets_grid[r][c] for (r, c) in cells),
                    unwrapped=tuple(unwrapped_grid[r][c] for (r, c) in cells),
                )
            )
    return components, offsets_grid, unwrapped_grid


# --------------------------------------------------------------------------- #
# Unsatisfied residues and unsatisfied edge residuals
# --------------------------------------------------------------------------- #
def _find_contradictions(
    phase: list[list[float]],
    valid: list[list[bool]],
    residue_grid: list[list[int]],
    cut_keys: set[tuple[tuple[int, int], tuple[int, int]]],
    unwrapped_grid: list[list[Optional[float]]],
) -> tuple[list[UnsatisfiedResidue], list[EdgeResidual]]:
    rows = len(phase)
    cols = len(phase[0])
    unsat_faces: list[UnsatisfiedResidue] = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            q = residue_grid[r][c]
            if q == 0:
                continue
            corners = _face_perimeter(r, c)  # a, b, e, d
            edges = tuple(
                Edge(corners[i], corners[(i + 1) % 4]) for i in range(4)
            )
            if any(edge.as_key() in cut_keys for edge in edges):
                continue  # a cut on the loop perimeter discharges the charge
            residuals = []
            for i in range(4):
                x = corners[i]
                y = corners[(i + 1) % 4]
                raw_wrap = wrapped_difference(
                    phase[y[0]][y[1]] - phase[x[0]][x[1]]
                )
                residuals.append(
                    unwrapped_grid[y[0]][y[1]]  # type: ignore[operator]
                    - unwrapped_grid[x[0]][x[1]]  # type: ignore[operator]
                    - raw_wrap
                )
            unsat_faces.append(
                UnsatisfiedResidue(
                    row=r,
                    col=c,
                    charge=q,
                    edges=edges,
                    edge_residuals=tuple(residuals),
                )
            )

    # Every present, non-cut edge must be locally sheet-consistent. A nonzero
    # residual here is an unchargeable contradiction (winding hidden behind a
    # mask hole, or the perimeter of an unsatisfied face); report it rather
    # than letting the unwrapped field look silently smooth.
    edge_residuals: list[EdgeResidual] = []
    for r in range(rows):
        for c in range(cols):
            if not valid[r][c]:
                continue
            for dr, dc in ((0, 1), (1, 0)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if not valid[nr][nc]:
                    continue
                edge = Edge((r, c), (nr, nc))
                if edge.as_key() in cut_keys:
                    continue
                residual = (
                    unwrapped_grid[nr][nc]  # type: ignore[operator]
                    - unwrapped_grid[r][c]  # type: ignore[operator]
                    - wrapped_difference(phase[nr][nc] - phase[r][c])
                )
                if abs(residual) > CONTRADICTION_TOL:
                    edge_residuals.append(EdgeResidual(edge, residual))
    edge_residuals.sort(key=lambda er: (er.edge.a, er.edge.b))
    return unsat_faces, edge_residuals


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def unwrap(
    phase: Sequence[Sequence[object]],
    mask: Optional[Sequence[Sequence[object]]] = None,
    *,
    max_expansions: object = 10_000_000,
    on_infeasible: object = "report",
) -> UnwrapResult:
    """Run residual branch-cut unwrapping and return an :class:`UnwrapResult`.

    ``max_expansions`` bounds the total number of BFS node expansions;
    exceeding it raises :class:`SearchBudgetExhaustedError` (nothing is
    returned and inputs are never mutated). With ``on_infeasible="report"``
    (default) a charge proved unreachable from both an opposite charge and
    the outer face is returned in ``unsatisfied``; with ``"raise"`` it raises
    :class:`UnwrapInfeasibleError` instead. Bad input raises
    :class:`InputValidationError`.
    """
    budget = _validate_budget(max_expansions)
    mode = _validate_on_infeasible(on_infeasible)
    rows, cols, fphase = _validate_phase(phase)
    valid = _validate_mask(mask, rows, cols)

    residue_grid, residues, faces = _compute_residues(fphase, valid)
    charged = {(res.row, res.col) for res in residues}
    graph = _DualGraph(rows, cols, faces, charged, budget)
    dipoles, leashes, cut_edges, _unreachable = _lay_cuts(
        residues, graph, mode
    )
    components, offsets_grid, unwrapped_grid = _flood_components(
        fphase, valid, graph
    )
    unsatisfied, edge_residuals = _find_contradictions(
        fphase,
        valid,
        residue_grid,
        {edge.as_key() for edge in cut_edges},
        unwrapped_grid,
    )

    return UnwrapResult(
        rows=rows,
        cols=cols,
        wrapped=tuple(tuple(row) for row in fphase),
        mask=tuple(tuple(row) for row in valid),
        residues=tuple(residues),
        residue_grid=tuple(tuple(row) for row in residue_grid),
        dipoles=tuple(dipoles),
        boundary_leashes=tuple(leashes),
        cuts=tuple(sorted(cut_edges, key=lambda e: (e.a, e.b))),
        components=tuple(components),
        offsets_grid=tuple(tuple(row) for row in offsets_grid),
        unwrapped_grid=tuple(tuple(row) for row in unwrapped_grid),
        unsatisfied=tuple(unsatisfied),
        edge_residuals=tuple(edge_residuals),
    )
