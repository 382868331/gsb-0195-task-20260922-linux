"""Structural invariants and deterministic randomized cross-checks.

Expected charges always come from the standalone loop formula; the package
result is checked against invariants computed directly here (sheet offsets,
cut topology, charge bookkeeping).
"""

from __future__ import annotations

import math
import random
import unittest

from phasecut import Edge, unwrap
from tests._reference import (
    TAU,
    analytic_fields_sum,
    analytic_ramp,
    analytic_vortex,
    full_mask,
    independent_residues,
    wrap,
)


def loop_corners(r, c):
    return ((r, c), (r, c + 1), (r + 1, c + 1), (r + 1, c))


class EdgeNormalizationTests(unittest.TestCase):
    def test_endpoints_are_ordered_without_aliasing(self):
        # Regression: the normaliser once overwrote `a` before reading it for
        # `b`, collapsing reversed edges into a doubled endpoint.
        edge = Edge((3, 4), (1, 2))
        self.assertEqual(edge.a, (1, 2))
        self.assertEqual(edge.b, (3, 4))
        self.assertEqual(Edge((0, 0), (0, 1)).as_key(), ((0, 0), (0, 1)))


class InvariantTests(unittest.TestCase):
    def check_result(self, phase, mask, result):
        rows = len(phase)
        cols = len(phase[0])
        cut_keys = {e.as_key() for e in result.cuts}

        # Shape and mask bookkeeping.
        self.assertEqual((result.rows, result.cols), (rows, cols))
        for r in range(rows):
            for c in range(cols):
                if mask[r][c]:
                    self.assertIsNotNone(result.unwrapped_grid[r][c])
                    self.assertIsNotNone(result.offsets_grid[r][c])
                    self.assertTrue(
                        -math.pi - 1e-12
                        <= result.wrapped[r][c]
                        < math.pi
                    )
                else:
                    self.assertIsNone(result.unwrapped_grid[r][c])
                    self.assertIsNone(result.offsets_grid[r][c])

        # Residue grid agrees with the standalone reference formula.
        ref = independent_residues(phase, mask)
        for r in range(rows):
            for c in range(cols):
                self.assertEqual(result.residue_grid[r][c], ref[r][c])

        # Edge accounting: scan every present, non-cut edge. Zero residual
        # means both ends share one sheet; a nonzero (integer 2pi) residual
        # must be reported verbatim in result.edge_residuals.
        reported = {
            er.edge.as_key(): er.residual for er in result.edge_residuals
        }
        owner: dict = {}
        for idx, comp in enumerate(result.components):
            self.assertEqual(comp.seed, min(comp.cells))
            seed_idx = comp.cells.index(comp.seed)
            self.assertEqual(comp.offsets[seed_idx], 0.0)
            for cell in comp.cells:
                owner[cell] = idx
        grid = result.offsets_grid
        for (r, c) in sorted(owner):
            for dr, dc in ((0, 1), (1, 0)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if not mask[nr][nc]:
                    continue
                edge_key = Edge((r, c), (nr, nc)).as_key()
                if edge_key in cut_keys:
                    continue
                residual = (
                    result.unwrapped_grid[nr][nc]
                    - result.unwrapped_grid[r][c]
                    - wrap(phase[nr][nc] - phase[r][c])
                )
                if abs(residual) > 1e-9:
                    self.assertIn(edge_key, reported)
                    self.assertAlmostEqual(
                        reported[edge_key], residual, delta=1e-12
                    )
                    k = round(residual / TAU)
                    self.assertNotEqual(k, 0)
                    self.assertAlmostEqual(residual, k * TAU, delta=1e-9)
                else:
                    self.assertNotIn(edge_key, reported)
                    self.assertEqual(owner.get((nr, nc)), owner[(r, c)])
                    self.assertAlmostEqual(
                        grid[nr][nc] - grid[r][c],
                        wrap(phase[nr][nc] - phase[r][c])
                        - (phase[nr][nc] - phase[r][c]),
                        delta=1e-11,
                    )
            # offset is a sheet multiple of 2pi.
            off = result.offsets_grid[r][c]
            self.assertAlmostEqual(off / TAU, round(off / TAU), delta=1e-11)

        # Components partition the valid pixels; seeds are flood minima.
        all_cells = [cell for comp in result.components for cell in comp.cells]
        valid_cells = [
            (r, c)
            for r in range(rows)
            for c in range(cols)
            if mask[r][c]
        ]
        self.assertEqual(sorted(all_cells), sorted(valid_cells))

        # Every cut edge is reported once and belongs to a dipole or leash.
        path_edge_keys: set = set()
        for d in result.dipoles:
            for e in d.edges:
                self.assertNotIn(e.as_key(), path_edge_keys)
                path_edge_keys.add(e.as_key())
            self.assertEqual(len(d.edges), len(d.path) - 1)
        for leash in result.boundary_leashes:
            for e in leash.edges:
                self.assertNotIn(e.as_key(), path_edge_keys)
                path_edge_keys.add(e.as_key())
        self.assertEqual(path_edge_keys, cut_keys)

        # Charge bookkeeping: each charged face is paired, leashed or
        # reported unsatisfied; counts balance.
        charged = {
            (r, c): q
            for r in range(rows - 1)
            for c in range(cols - 1)
            for q in [result.residue_grid[r][c]]
            if q
        }
        accounted: dict = {}
        for d in result.dipoles:
            accounted[d.positive] = accounted.get(d.positive, 0) + 1
            accounted[d.negative] = accounted.get(d.negative, 0) - 1
        for leash in result.boundary_leashes:
            accounted[leash.source] = accounted.get(leash.source, 0) + (
                1 if leash.charge > 0 else -1
            )
        for u in result.unsatisfied:
            accounted[(u.row, u.col)] = accounted.get((u.row, u.col), 0) + (
                1 if u.charge > 0 else -1
            )
            # Sign convention of the reported contradiction.
            self.assertAlmostEqual(
                sum(u.edge_residuals), -TAU * u.charge, delta=1e-9
            )
        for cell, q in charged.items():
            self.assertEqual(accounted.get(cell, 0), 1 if q > 0 else -1, cell)

        # Unsatisfied faces really have no cut on their own perimeter.
        for u in result.unsatisfied:
            corners = loop_corners(u.row, u.col)
            for i in range(4):
                key = Edge(corners[i], corners[(i + 1) % 4]).as_key()
                self.assertNotIn(key, cut_keys)

    def test_ramp_variants(self):
        for seed in range(40):
            rng = random.Random(seed)
            rows = rng.randint(2, 16)
            cols = rng.randint(2, 16)
            ar = rng.uniform(-1.8, 1.8)
            ac = rng.uniform(-1.8, 1.8)
            phase = analytic_ramp(rows, cols, ar, ac)
            result = unwrap(phase)
            self.assertEqual(result.residues, (), seed)
            self.check_result(phase, full_mask(rows, cols), result)

    def test_random_vortex_fields_with_random_masks(self):
        for seed in range(60):
            rng = random.Random(1000 + seed)
            rows = rng.randint(5, 20)
            cols = rng.randint(5, 20)
            terms = [("ramp", rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8))]
            n = rng.randint(0, 5)
            for _ in range(n):
                fr = rng.randint(0, rows - 2)
                fc = rng.randint(0, cols - 2)
                terms.append(("vortex", (fr, fc), rng.choice((1, -1))))
            phase = analytic_fields_sum(rows, cols, terms)
            mask = [
                [rng.random() > 0.18 for _ in range(cols)] for _ in range(rows)
            ]
            result = unwrap(phase, mask)
            self.check_result(phase, mask, result)

    def test_single_vortex_many_positions(self):
        for rows, cols in ((5, 5), (8, 6), (9, 9)):
            for fr in range(rows - 1):
                for fc in range(cols - 1):
                    phase = analytic_vortex(rows, cols, (fr, fc), 1)
                    result = unwrap(phase)
                    self.check_result(
                        phase, full_mask(rows, cols), result
                    )
                    self.assertEqual(len(result.boundary_leashes), 1)
                    self.assertEqual(result.unsatisfied, ())

    def test_repeated_calls_are_deterministic(self):
        phase = analytic_fields_sum(
            10,
            10,
            [
                ("ramp", 0.3, -0.2),
                ("vortex", (2, 2), 1),
                ("vortex", (5, 6), -1),
                ("vortex", (4, 2), 1),
            ],
        )
        first = unwrap(phase)
        second = unwrap(phase)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
