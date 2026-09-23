"""Independent fixtures and reference helpers for the phasecut tests.

Nothing here imports the package under test. Expected charges come from the
known analytic winding of the constructed fields and are re-derived with an
independent loop formula (``independent_residues``) written for these tests.
Round-trip checks against the package are used only as supplementary checks
in the test files, never as the source of expected values.
"""

from __future__ import annotations

import math
from typing import Optional

TAU = 2.0 * math.pi


def wrap(x: float) -> float:
    """Independent wrap into [-pi, pi)."""
    y = math.fmod(x, TAU)
    if y >= math.pi:
        y -= TAU
    elif y < -math.pi:
        y += TAU
    return y


def independent_residues(phase, mask):
    """Plain nearest-integer loop-sum / 2pi residue grid (standalone)."""
    rows = len(phase)
    cols = len(phase[0])
    out = [[0] * cols for _ in range(rows)]
    for r in range(rows - 1):
        for c in range(cols - 1):
            corners = ((r, c), (r, c + 1), (r + 1, c + 1), (r + 1, c))
            if not all(mask[i][j] for (i, j) in corners):
                continue
            s = 0.0
            for k in range(4):
                ar, ac = corners[k]
                br, bc = corners[(k + 1) % 4]
                s += wrap(phase[br][bc] - phase[ar][ac])
            out[r][c] = int(math.floor(s / TAU + 0.5))
    return out


def full_mask(rows: int, cols: int):
    return [[True] * cols for _ in range(rows)]


def analytic_ramp(rows, cols, ar, ac, phase0=0.0):
    """Linear phase ar*r + ac*c + phase0, wrapped; no vortices anywhere."""
    return [
        [wrap(ar * r + ac * c + phase0) for c in range(cols)]
        for r in range(rows)
    ]


def true_ramp(rows, cols, ar, ac, phase0=0.0):
    return [
        [ar * r + ac * c + phase0 for c in range(cols)] for r in range(rows)
    ]


def analytic_vortex(rows, cols, face, charge=1):
    """Principal-argument field with one unit winding around ``face``.

    The branch centre is the centre of the 2x2 face ``face``; the integer
    winding is ``charge`` (typically +/-1).
    """
    cr = face[0] + 0.5
    cc = face[1] + 0.5
    return [
        [wrap(charge * math.atan2(r - cr, c - cc)) for c in range(cols)]
        for r in range(rows)
    ]


def analytic_fields_sum(rows, cols, terms):
    """Sum of ramps and unit vortices, wrapped.

    Each term is ("ramp", ar, ac) or ("vortex", face, charge). Well-separated
    vortex centres give winding equal to the sum of the individual windings.
    """
    grid = [[0.0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            v = 0.0
            for term in terms:
                if term[0] == "ramp":
                    v += term[1] * r + term[2] * c
                else:
                    _, face, charge = term
                    v += charge * math.atan2(
                        r - (face[0] + 0.5), c - (face[1] + 0.5)
                    )
            grid[r][c] = wrap(v)
    return grid


def vortex_2x2(charge=1, shift=-0.1):
    """Four pixel values whose wrapped loop sums to charge * 2*pi.

    Values are chosen away from the +/-pi exclusion boundary:
    a = shift, b = pi/2 + shift, e = pi + shift (valid: < pi),
    d = -pi/2 + shift (valid: > -pi).
    """
    s = shift
    a = s
    b = math.pi / 2.0 + s
    e = math.pi + s
    d = -math.pi / 2.0 + s
    return [[a, b], [d, e]] if charge == 1 else [
        [a, d],
        [b, e],
    ]


def place_block(rows, cols, block, top, left):
    """Embed ``block`` (small wrapped-phase rows) into an all-zero grid."""
    grid = [[0.0] * cols for _ in range(rows)]
    for r, row in enumerate(block):
        for c, value in enumerate(row):
            grid[top + r][left + c] = value
    return grid


def block_mask(rows, cols, block_rows, block_cols, top, left):
    mask = [[False] * cols for _ in range(rows)]
    for r in range(top, top + block_rows):
        for c in range(left, left + block_cols):
            mask[r][c] = True
    return mask


def naive_flood(phase, mask):
    """Cut-free integration: the 'before cuts' picture.

    Independent of the package; independent BFS order from a lexicographic
    scan so the 2*pi contradiction at an uncut vortex shows up explicitly.
    """
    rows = len(phase)
    cols = len(phase[0])
    u: list[list[Optional[float]]] = [[None] * cols for _ in range(rows)]
    for sr in range(rows):
        for sc in range(cols):
            if not mask[sr][sc] or u[sr][sc] is not None:
                continue
            u[sr][sc] = phase[sr][sc]
            frontier = [(sr, sc)]
            while frontier:
                nxt = []
                for r, c in frontier:
                    for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)):
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < rows and 0 <= nc < cols):
                            continue
                        if not mask[nr][nc] or u[nr][nc] is not None:
                            continue
                        u[nr][nc] = u[r][c] + wrap(phase[nr][nc] - phase[r][c])
                        nxt.append((nr, nc))
                frontier = nxt
    return u


def loop_errors_from_unwrapped(unwrapped, phase, mask):
    """Per-face integrated loop error using an unwrapped grid and wraps."""
    rows = len(phase)
    cols = len(phase[0])
    out = [[None] * cols for _ in range(rows)]
    for r in range(rows - 1):
        for c in range(cols - 1):
            corners = ((r, c), (r, c + 1), (r + 1, c + 1), (r + 1, c))
            if not all(mask[i][j] for (i, j) in corners):
                continue
            if any(unwrapped[i][j] is None for (i, j) in corners):
                continue
            s = 0.0
            for k in range(4):
                ar, ac = corners[k]
                br, bc = corners[(k + 1) % 4]
                s += unwrapped[br][bc] - unwrapped[ar][ac]
                s -= wrap(phase[br][bc] - phase[ar][ac])
            out[r][c] = s
    return out
