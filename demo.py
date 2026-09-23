"""Fixed-input demonstration for the phasecut package.

Run with ``python demo.py`` (standard library only, offline, deterministic).

It computes, for real (no pre-recorded values):

1. a residue-free wrapped ramp crossing +/-pi many times -- unwrapped exactly;
2. a single-vortex reference field -- paired/leashed cuts are laid and the
   loop error before vs after cutting is printed;
3. a genuinely triggered contract rejection -- NaN is refused with a
   locatable error.
"""

from __future__ import annotations

import math
import sys

from phasecut import (
    InputValidationError,
    SearchBudgetExhaustedError,
    TAU,
    Edge,
    unwrap,
)

# The demo ships its own tiny deterministic field generators instead of
# importing the test-only reference helpers, so it stays runnable on its own.


def wrap(x: float) -> float:
    return (x + math.pi) % TAU - math.pi


def ramp(rows, cols, ar, ac):
    return [[wrap(ar * r + ac * c) for c in range(cols)] for r in range(rows)]


def vortex(rows, cols, face, charge=1):
    cr, cc = face[0] + 0.5, face[1] + 0.5
    return [
        [wrap(charge * math.atan2(r - cr, c - cc)) for c in range(cols)]
        for r in range(rows)
    ]


def cut_free_flood(phase):
    """Plain integration with no cuts (the 'before cuts' picture)."""
    rows, cols = len(phase), len(phase[0])
    u = [[None] * cols for _ in range(rows)]
    for sr in range(rows):
        for sc in range(cols):
            if u[sr][sc] is not None:
                continue
            u[sr][sc] = phase[sr][sc]
            stack = [(sr, sc)]
            while stack:
                r, c = stack.pop()
                for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and u[nr][nc] is None:
                        u[nr][nc] = u[r][c] + wrap(phase[nr][nc] - phase[r][c])
                        stack.append((nr, nc))
    return u


def face_loop_error(unwrapped, phase, r, c):
    corners = ((r, c), (r, c + 1), (r + 1, c + 1), (r + 1, c))
    s = 0.0
    for k in range(4):
        ar, ac = corners[k]
        br, bc = corners[(k + 1) % 4]
        s += unwrapped[br][bc] - unwrapped[ar][ac]
        s -= wrap(phase[br][bc] - phase[ar][ac])
    return s


def hr(title):
    print()
    print("=" * 68)
    print(title)
    print("=" * 68)


def main() -> int:
    # ------------------------------------------------------------------ #
    hr("1. Residue-free wrapped ramp (0.83*r + 1.27*c, 8x10)")
    phase = ramp(8, 10, 0.83, 1.27)
    result = unwrap(phase)
    print(f"residues: {len(result.residues)}   cuts: {len(result.cuts)}   "
          f"components: {len(result.components)}")
    worst = 0.0
    for r in range(8):
        for c in range(10):
            worst = max(
                worst,
                abs(result.unwrapped_grid[r][c] - (0.83 * r + 1.27 * c)),
            )
    print(f"max |unwrapped - true linear phase| over all 80 pixels: "
          f"{worst:.3e} rad")
    print("seed (smallest pixel) offset basis:",
          result.components[0].seed, "-> 0")

    # ------------------------------------------------------------------ #
    hr("2. Single +1 vortex reference (8x8, charge centred on face (3,3))")
    phase = vortex(8, 8, (3, 3), +1)
    before = cut_free_flood(phase)
    err_before = face_loop_error(before, phase, 3, 3)
    print(f"loop error at face (3,3) BEFORE cuts: {err_before:+.6f} rad "
          f"(= {err_before / TAU:+.2f} * 2*pi)  <- hidden branch jump")

    result = unwrap(phase)
    print(f"residues: {[(res.row, res.col, res.charge) for res in result.residues]}")
    print(f"opposite-charge pairs: {len(result.dipoles)}")
    for leash in result.boundary_leashes:
        print(
            f"boundary leash: face {leash.source} charge {leash.charge:+d} "
            f"path {list(leash.path)} exit {leash.exit.a}-{leash.exit.b}"
        )
    print("cut edges (uncrossable by the flood fill):")
    for edge in result.cuts:
        print(f"  {edge.a} -- {edge.b}")
    print(f"components after cutting: {len(result.components)}; "
          f"unsatisfied charged faces: {len(result.unsatisfied)}; "
          f"unsatisfied edge residuals: {len(result.edge_residuals)}")

    cut_keys = {e.as_key() for e in result.cuts}
    corners = ((3, 3), (3, 4), (4, 4), (4, 3))
    for k in range(4):
        x, y = corners[k], corners[(k + 1) % 4]
        edge = Edge(x, y)
        tagged = "CUT" if edge.as_key() in cut_keys else "   "
        residual = (
            result.unwrapped_grid[y[0]][y[1]]
            - result.unwrapped_grid[x[0]][x[1]]
            - wrap(phase[y[0]][y[1]] - phase[x[0]][x[1]])
        )
        print(f"  loop edge {x} -> {y} [{tagged}]  sheet residual "
              f"{residual / TAU:+.3f} * 2*pi")
    worst_uncut = 0.0
    for r in range(7):
        for c in range(7):
            loop = ((r, c), (r, c + 1), (r + 1, c + 1), (r + 1, c))
            if any(
                Edge(loop[i], loop[(i + 1) % 4]).as_key() in cut_keys
                for i in range(4)
            ):
                continue
            worst_uncut = max(
                worst_uncut,
                abs(face_loop_error(
                    [[v for v in row] for row in result.unwrapped_grid],
                    phase, r, c,
                )),
            )
    print(f"worst loop error on every face NOT touched by a cut: "
          f"{worst_uncut:.3e} rad  (AFTER cuts; the 2*pi jump now sits on "
          f"the cut edge instead of being hidden)")

    # ------------------------------------------------------------------ #
    hr("3. Contract rejection that is actually triggered (NaN input)")
    bad = ramp(4, 4, 0.5, 0.5)
    bad[2][1] = math.nan
    try:
        unwrap(bad)
    except InputValidationError as exc:
        print(f"InputValidationError raised as required: {exc}")
        print(f"locatable field: {exc.loc!r}")
    else:  # pragma: no cover - the rejection must happen
        print("ERROR: NaN was not rejected", file=sys.stderr)
        return 1

    # A budget cutoff is also demonstrated, kept cheap and distinct from a
    # proof of infeasibility.
    try:
        unwrap(vortex(6, 6, (2, 2), +1), max_expansions=1)
    except SearchBudgetExhaustedError as exc:
        print(f"budget cutoff (distinct from infeasibility): "
              f"{type(exc).__name__}")
    else:  # pragma: no cover
        print("ERROR: budget was not enforced", file=sys.stderr)
        return 1

    print()
    print("demo finished normally.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
