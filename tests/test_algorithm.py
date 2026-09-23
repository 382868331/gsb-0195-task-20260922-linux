"""Algorithm tests: residues, pairing, leashes, masks, ties, cross-pi.

Expected charges come from the known analytic winding of the fixtures and are
cross-checked with the standalone loop formula in tests/_reference.py, which
never imports the package under test.
"""

from __future__ import annotations

import math
import unittest

from phasecut import (
    Edge,
    InputValidationError,
    UnwrapInfeasibleError,
    unwrap,
)
from tests._reference import (
    TAU,
    analytic_fields_sum,
    analytic_ramp,
    analytic_vortex,
    block_mask,
    full_mask,
    independent_residues,
    loop_errors_from_unwrapped,
    naive_flood,
    place_block,
    true_ramp,
    vortex_2x2,
    wrap,
)


def only_charge_locations(residue_grid):
    return [
        (r, c, q)
        for r, row in enumerate(residue_grid)
        for c, q in enumerate(row)
        if q
    ]


def cut_key_set(result):
    return {edge.as_key() for edge in result.cuts}


def face_touches_cut(r, c, cut_keys):
    corners = ((r, c), (r, c + 1), (r + 1, c + 1), (r + 1, c))
    edges = [
        (corners[i], corners[(i + 1) % 4]) for i in range(4)
    ]
    return any(
        (tuple(sorted((x, y)))) in cut_keys for x, y in edges
    )


def assert_integrity(self, result, phase, mask):
    """Post-cut invariants expected of every well-formed result.

    * every valid face WITHOUT a cut edge on its perimeter integrates to zero
      (corners then share one component);
    * every interior cut edge carries an exact non-zero integer*2pi jump
      between the two component-local sheets;
    * every charged face is either touched by a cut or reported unsatisfied.
    """
    rows, cols = result.rows, result.cols
    cut_keys = cut_key_set(result)
    # Non-cut face loops vanish.
    for r in range(rows - 1):
        for c in range(cols - 1):
            corners = ((r, c), (r, c + 1), (r + 1, c + 1), (r + 1, c))
            if not all(mask[i][j] for (i, j) in corners):
                continue
            if face_touches_cut(r, c, cut_keys):
                continue
            s = 0.0
            for k in range(4):
                ar, ac = corners[k]
                br, bc = corners[(k + 1) % 4]
                s += result.unwrapped_grid[br][bc] - result.unwrapped_grid[ar][ac]
                s -= wrap(phase[br][bc] - phase[ar][ac])
            self.assertAlmostEqual(s, 0.0, delta=1e-9, msg=f"face {(r,c)}")
    # Interior cut edges: integer 2pi sheet jumps.
    for edge in result.cuts:
        (ar, ac), (br, bc) = edge.a, edge.b
        if not (0 <= ar < rows and 0 <= ac < cols):
            continue  # virtual exit edge
        if not (0 <= br < rows and 0 <= bc < cols):
            continue
        residual = (
            result.unwrapped_grid[br][bc]
            - result.unwrapped_grid[ar][ac]
            - wrap(phase[br][bc] - phase[ar][ac])
        )
        k = round(residual / TAU)
        self.assertNotEqual(k, 0, f"cut edge {edge} carries no sheet jump")
        self.assertAlmostEqual(
            residual, k * TAU, delta=1e-9, msg=f"cut edge {edge}"
        )
    # Charged faces: discharged or explicitly unsatisfied.
    unsat = {(u.row, u.col) for u in result.unsatisfied}
    for res in result.residues:
        self.assertTrue(
            face_touches_cut(res.row, res.col, cut_keys)
            or (res.row, res.col) in unsat
        )


class RampTests(unittest.TestCase):
    def test_wrapped_ramp_unwraps_exactly_without_cuts(self):
        rows, cols = 9, 11
        ar, ac = 0.73, 0.51  # crosses +/-pi many times, residue-free
        phase = analytic_ramp(rows, cols, ar, ac)
        truth = true_ramp(rows, cols, ar, ac)

        # Independent reference: no vortices anywhere.
        self.assertEqual(
            only_charge_locations(independent_residues(phase, full_mask(rows, cols))),
            [],
        )

        result = unwrap(phase)
        self.assertEqual(result.residues, ())
        self.assertEqual(result.dipoles, ())
        self.assertEqual(result.boundary_leashes, ())
        self.assertEqual(result.cuts, ())
        self.assertEqual(len(result.components), 1)
        self.assertEqual(result.components[0].seed, (0, 0))

        for r in range(rows):
            for c in range(cols):
                got = result.unwrapped_grid[r][c]
                self.assertAlmostEqual(
                    got,
                    truth[r][c],
                    delta=1e-12,
                    msg=f"unwrapped mismatch at ({r},{c})",
                )
                # offset is exactly an integer multiple of 2pi.
                off = result.offsets_grid[r][c]
                self.assertAlmostEqual(off / TAU, round(off / TAU), delta=1e-12)

    def test_ramp_with_phase_offset(self):
        rows, cols = 6, 6
        phase = analytic_ramp(rows, cols, 1.1, -0.9, phase0=1.7)
        truth = true_ramp(rows, cols, 1.1, -0.9, phase0=1.7)
        result = unwrap(phase)
        for r in range(rows):
            for c in range(cols):
                self.assertAlmostEqual(
                    result.unwrapped_grid[r][c], truth[r][c], delta=1e-12
                )


class SingleVortexTests(unittest.TestCase):
    def test_single_vortex_leashes_straight_up(self):
        rows = cols = 8
        face = (3, 3)
        phase = analytic_vortex(rows, cols, face, +1)
        self.assertEqual(
            only_charge_locations(independent_residues(phase, full_mask(rows, cols))),
            [(3, 3, 1)],
        )

        result = unwrap(phase)
        self.assertEqual(len(result.residues), 1)
        self.assertEqual(result.residues[0].charge, 1)
        self.assertEqual(result.dipoles, ())
        self.assertEqual(result.unsatisfied, ())
        self.assertEqual(len(result.boundary_leashes), 1)

        leash = result.boundary_leashes[0]
        self.assertEqual(leash.source, (3, 3))
        self.assertEqual(leash.charge, 1)
        # Distance to border is 3 both upward and leftward; the
        # lexicographically smallest border face (0,3) wins.
        self.assertEqual(
            leash.path, ((3, 3), (2, 3), (1, 3), (0, 3))
        )
        self.assertEqual(
            leash.exit, Edge((0, 3), (0, 4))
        )
        # 3 crossed interior edges plus the exit edge.
        self.assertEqual(len(leash.edges), 4)
        self.assertEqual(
            set(result.cuts),
            {
                Edge((3, 3), (3, 4)),
                Edge((2, 3), (2, 4)),
                Edge((1, 3), (1, 4)),
                Edge((0, 3), (0, 4)),
            },
        )

    def test_vortex_loop_error_before_and_zero_after(self):
        rows = cols = 8
        face = (3, 3)
        phase = analytic_vortex(rows, cols, face, +1)
        mask = full_mask(rows, cols)

        before = loop_errors_from_unwrapped(naive_flood(phase, mask), phase, mask)
        self.assertAlmostEqual(before[3][3], -TAU, delta=1e-12)

        result = unwrap(phase)
        # After cuts the vortex face is split across components, so its
        # geometric loop no longer sits on one sheet; the 2pi contradiction
        # is now carried explicitly by the cut edge instead of being hidden.
        self.assertEqual(result.unsatisfied, ())
        assert_integrity(self, result, phase, mask)
        cut = Edge((3, 3), (3, 4))
        self.assertIn(cut.as_key(), cut_key_set(result))
        jump = (
            result.unwrapped_grid[3][4]
            - result.unwrapped_grid[3][3]
            - wrap(phase[3][4] - phase[3][3])
        )
        self.assertAlmostEqual(jump, -TAU, delta=1e-12)
        # Every face NOT touched by a cut integrates to zero loop error.
        after = loop_errors_from_unwrapped(
            [[v for v in row] for row in result.unwrapped_grid], phase, mask
        )
        for r in range(rows - 1):
            for c in range(cols - 1):
                if not face_touches_cut(r, c, cut_key_set(result)):
                    self.assertAlmostEqual(after[r][c], 0.0, delta=1e-10)

    def test_negative_vortex(self):
        rows = cols = 7
        phase = analytic_vortex(rows, cols, (2, 2), -1)
        self.assertEqual(
            only_charge_locations(independent_residues(phase, full_mask(rows, cols))),
            [(2, 2, -1)],
        )
        result = unwrap(phase)
        self.assertEqual(len(result.boundary_leashes), 1)
        self.assertEqual(result.boundary_leashes[0].charge, -1)
        self.assertEqual(result.unsatisfied, ())
        # Up and left are both distance 2; smallest border face (0,2) wins.
        self.assertEqual(
            result.boundary_leashes[0].path, ((2, 2), (1, 2), (0, 2))
        )
        self.assertEqual(
            result.boundary_leashes[0].exit, Edge((0, 2), (0, 3))
        )


class DipoleTests(unittest.TestCase):
    def test_opposite_pair_takes_straight_shortest_path(self):
        rows = cols = 8
        phase = analytic_fields_sum(
            rows,
            cols,
            [("vortex", (2, 2), +1), ("vortex", (2, 5), -1)],
        )
        charges = only_charge_locations(
            independent_residues(phase, full_mask(rows, cols))
        )
        self.assertEqual(sorted(charges), [(2, 2, 1), (2, 5, -1)])

        result = unwrap(phase)
        self.assertEqual(result.boundary_leashes, ())
        self.assertEqual(result.unsatisfied, ())
        self.assertEqual(len(result.dipoles), 1)
        dipole = result.dipoles[0]
        self.assertEqual(dipole.positive, (2, 2))
        self.assertEqual(dipole.negative, (2, 5))
        self.assertEqual(
            dipole.path, ((2, 2), (2, 3), (2, 4), (2, 5))
        )
        self.assertEqual(
            set(dipole.edges),
            {
                Edge((2, 3), (3, 3)),
                Edge((2, 4), (3, 4)),
                Edge((2, 5), (3, 5)),
            },
        )
        self.assertEqual(len(result.components), 1)
        # Across each cut edge the two sides differ by exactly one 2pi sheet.
        grid = result.unwrapped_grid
        for edge in dipole.edges:
            (ar, ac), (br, bc) = edge.a, edge.b
            jump = grid[br][bc] - grid[ar][ac]
            wrapped_jump = wrap(phase[br][bc] - phase[ar][ac])
            self.assertAlmostEqual(abs(jump - wrapped_jump), TAU, delta=1e-12)

    def test_equidistant_choices_are_lexicographic(self):
        # + at (2,2); negatives equidistant at (1,2) [up] and (2,1) [left].
        rows = cols = 7
        phase = analytic_fields_sum(
            rows,
            cols,
            [
                ("vortex", (2, 2), +1),
                ("vortex", (1, 2), -1),
                ("vortex", (2, 1), -1),
            ],
        )
        charges = only_charge_locations(
            independent_residues(phase, full_mask(rows, cols))
        )
        self.assertEqual(
            sorted(charges), [(1, 2, -1), (2, 1, -1), (2, 2, 1)]
        )

        result = unwrap(phase)
        # Lexicographic processing pairs the (1,2) negative first, with the
        # + at (2,2); the (2,1) negative then leashes one step to the left.
        self.assertEqual(len(result.dipoles), 1)
        self.assertEqual(result.dipoles[0].positive, (2, 2))
        self.assertEqual(result.dipoles[0].negative, (1, 2))
        self.assertEqual(result.dipoles[0].path, ((2, 2), (1, 2)))

        self.assertEqual(len(result.boundary_leashes), 1)
        leash = result.boundary_leashes[0]
        self.assertEqual(leash.source, (2, 1))
        self.assertEqual(leash.charge, -1)
        self.assertEqual(leash.path, ((2, 1), (2, 0)))
        self.assertEqual(leash.exit, Edge((2, 0), (3, 0)))
        self.assertEqual(result.unsatisfied, ())

    def test_pair_then_leash_respects_existing_cuts(self):
        # Two positives on the same column and one negative to their right.
        # The first (lexicographic) positive consumes the negative; the other
        # positive must reach the border without crossing that dipole.
        rows = cols = 7
        phase = analytic_fields_sum(
            rows,
            cols,
            [
                ("vortex", (1, 3), +1),
                ("vortex", (3, 3), +1),
                ("vortex", (1, 5), -1),
            ],
        )
        charges = only_charge_locations(
            independent_residues(phase, full_mask(rows, cols))
        )
        self.assertEqual(
            sorted(charges), [(1, 3, 1), (1, 5, -1), (3, 3, 1)]
        )
        result = unwrap(phase)
        self.assertEqual(result.unsatisfied, ())
        self.assertEqual(len(result.dipoles), 1)
        self.assertEqual(
            (result.dipoles[0].positive, result.dipoles[0].negative),
            ((1, 3), (1, 5)),
        )
        self.assertEqual(len(result.boundary_leashes), 1)
        leash = result.boundary_leashes[0]
        self.assertEqual(leash.source, (3, 3))
        # Leash route must not reuse a dipole edge.
        dipole_keys = {e.as_key() for e in result.dipoles[0].edges}
        for edge in leash.edges:
            self.assertNotIn(edge.as_key(), dipole_keys)


class MaskTests(unittest.TestCase):
    def test_interior_vortex_island_is_reported_not_hidden(self):
        rows = cols = 8
        block = vortex_2x2(+1)
        phase = place_block(rows, cols, block, top=2, left=2)
        mask = block_mask(rows, cols, 2, 2, top=2, left=2)

        self.assertEqual(
            only_charge_locations(independent_residues(phase, mask)),
            [(2, 2, 1)],
        )

        # Default: contradiction is reported, algorithm still completes.
        result = unwrap(phase, mask)
        self.assertEqual(result.dipoles, ())
        self.assertEqual(result.boundary_leashes, ())
        self.assertEqual(len(result.unsatisfied), 1)
        bad = result.unsatisfied[0]
        self.assertEqual((bad.row, bad.col), (2, 2))
        self.assertEqual(bad.charge, 1)
        self.assertAlmostEqual(sum(bad.edge_residuals), -TAU, delta=1e-12)
        # Exactly one perimeter edge hides the full 2pi jump.
        big = [x for x in bad.edge_residuals if abs(x) > 1.0]
        self.assertEqual(len(big), 1)
        self.assertAlmostEqual(big[0], -TAU, delta=1e-12)

        # The same situation must be provably infeasible on request, and
        # that must be distinguishable from a budget cutoff.
        with self.assertRaises(UnwrapInfeasibleError) as ctx:
            unwrap(phase, mask, on_infeasible="raise")
        self.assertEqual(ctx.exception.unreachable, ((2, 2),))

    def test_separate_mask_components_get_independent_bases(self):
        rows = cols = 7
        phase = analytic_ramp(rows, cols, 0.4, 0.4)
        mask = full_mask(rows, cols)
        # Remove a full column so the right-hand strip is an isolated region.
        for r in range(rows):
            mask[r][3] = False
        result = unwrap(phase, mask)
        seeds = sorted(comp.seed for comp in result.components)
        self.assertEqual(seeds, [(0, 0), (0, 4)])
        for comp in result.components:
            self.assertEqual(comp.seed, min(comp.cells))
            seed_offset = comp.offsets[comp.cells.index(comp.seed)]
            self.assertEqual(seed_offset, 0.0)
        # Invalid pixels carry no result.
        for r in range(rows):
            self.assertIsNone(result.unwrapped_grid[r][3])
            self.assertIsNone(result.offsets_grid[r][3])

    def test_single_pixel_island(self):
        rows = cols = 4
        phase = analytic_ramp(rows, cols, 0.3, 0.3)
        mask = [[False] * cols for _ in range(rows)]
        mask[1][1] = True
        mask[0][0] = True
        result = unwrap(phase, mask)
        self.assertEqual(len(result.components), 2)
        cells = {comp.seed: comp.cells for comp in result.components}
        self.assertEqual(cells[(1, 1)], ((1, 1),))
        self.assertEqual(cells[(0, 0)], ((0, 0),))
        self.assertEqual(result.residues, ())
        self.assertEqual(result.unsatisfied, ())

    def test_winding_hidden_behind_mask_hole_is_reported_as_edge_residual(self):
        # A unit vortex centred on face (3,3); masking one of its four
        # pixels destroys every valid 2x2 face that could carry the charge
        # (the standalone residue reference then sees no charge), yet the
        # winding still lives in the data around the hole. It must surface as
        # unsatisfied edge residuals, never as a silently smooth field.
        rows = cols = 7
        phase = analytic_vortex(rows, cols, (3, 3), +1)
        mask = full_mask(rows, cols)
        mask[4][4] = False
        self.assertEqual(
            only_charge_locations(independent_residues(phase, mask)), []
        )

        result = unwrap(phase, mask)
        self.assertEqual(result.residues, ())
        self.assertEqual(result.dipoles, ())
        self.assertEqual(result.boundary_leashes, ())
        self.assertEqual(result.unsatisfied, ())
        self.assertGreaterEqual(len(result.edge_residuals), 1)
        sheets = []
        for er in result.edge_residuals:
            k = er.residual / TAU
            self.assertAlmostEqual(k, round(k), delta=1e-9)
            self.assertNotEqual(round(k), 0)
            sheets.append(round(k))
        # The hidden +1 winding surfaces on edges bordering the hole region.
        self.assertTrue(any(k >= 1 for k in sheets))
        # Independent confirmation (no package calls): the wrapped-difference
        # cycle around the 8-pixel ring enclosing the masked pixel sums to a
        # full 2*pi, proving the winding is real and not manufactured.
        ring = (
            (3, 3), (3, 4), (3, 5), (4, 5),
            (5, 5), (5, 4), (5, 3), (4, 3),
        )
        cycle_sum = 0.0
        for i in range(len(ring)):
            ar, ac = ring[i]
            br, bc = ring[(i + 1) % len(ring)]
            cycle_sum += wrap(phase[br][bc] - phase[ar][ac])
        self.assertAlmostEqual(abs(cycle_sum), TAU, delta=1e-9)


class CrossPiTests(unittest.TestCase):
    def test_vortex_values_against_pi_boundary(self):
        phase = vortex_2x2(+1, shift=-0.1)
        # e = pi - 0.1 is the largest legal value; d > -pi.
        flat = [x for row in phase for x in row]
        self.assertLess(max(flat), math.pi)
        self.assertGreaterEqual(min(flat), -math.pi)
        self.assertEqual(
            only_charge_locations(independent_residues(phase, [[True] * 2] * 2)),
            [(0, 0, 1)],
        )

        result = unwrap(phase)
        # The only face is itself a border face: a zero-length leash whose
        # exit crosses its top edge (up wins over left).
        self.assertEqual(len(result.boundary_leashes), 1)
        leash = result.boundary_leashes[0]
        self.assertEqual(leash.path, ((0, 0),))
        self.assertEqual(leash.exit, Edge((0, 0), (0, 1)))
        self.assertEqual(result.unsatisfied, ())
        self.assertEqual(len(result.components), 1)  # cut, but walkable around

    def test_mirror_negative_vortex_against_boundary(self):
        phase = vortex_2x2(-1, shift=-0.1)
        result = unwrap(phase)
        self.assertEqual(result.residues[0].charge, -1)
        self.assertEqual(result.unsatisfied, ())
        self.assertEqual(len(result.boundary_leashes), 1)

    def test_pi_exactly_is_rejected(self):
        phase = vortex_2x2(+1)
        phase[1][1] = math.pi  # half-open interval: illegal
        with self.assertRaises(InputValidationError) as ctx:
            unwrap(phase)
        self.assertEqual(ctx.exception.loc, "phase[1][1]")


class SteepFieldTests(unittest.TestCase):
    def test_steep_ramp_plus_pair_crosses_pi_repeatedly(self):
        rows, cols = 9, 12
        phase = analytic_fields_sum(
            rows,
            cols,
            [
                ("ramp", 2.05, 1.35),
                ("vortex", (3, 2), +1),
                ("vortex", (3, 8), -1),
            ],
        )
        mask = full_mask(rows, cols)
        # The steep smooth gradient shifts the discrete wrapped-difference
        # charges onto adjacent faces relative to the analytic centres; the
        # standalone reference determines where (deterministically).
        charges = only_charge_locations(independent_residues(phase, mask))
        self.assertEqual(sorted(charges), [(3, 3, 1), (3, 7, -1)])

        result = unwrap(phase)
        self.assertEqual(
            only_charge_locations(result.residue_grid),
            sorted(charges),
        )
        self.assertEqual(result.unsatisfied, ())
        self.assertEqual(len(result.dipoles), 1)
        self.assertEqual(
            (result.dipoles[0].positive, result.dipoles[0].negative),
            ((3, 3), (3, 7)),
        )
        self.assertEqual(result.unsatisfied, ())
        # Global integrity: every uncut face loop closes, every cut edge
        # carries an integer 2pi jump, every charge is discharged.
        assert_integrity(self, result, phase, mask)


if __name__ == "__main__":
    unittest.main()
