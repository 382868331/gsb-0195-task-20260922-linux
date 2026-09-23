"""Tests for input validation and error semantics."""

from __future__ import annotations

import math
import unittest

from phasecut import (
    InputValidationError,
    SearchBudgetExhaustedError,
    residue_map,
    unwrap,
)


def _grid(value, rows=4, cols=4):
    return [[value] * cols for _ in range(rows)]


class ValidationTests(unittest.TestCase):
    def test_rejects_nan_with_location(self):
        phase = _grid(0.0)
        phase[1][2] = math.nan
        with self.assertRaises(InputValidationError) as ctx:
            unwrap(phase)
        self.assertEqual(ctx.exception.loc, "phase[1][2]")

    def test_rejects_infinities(self):
        for bad in (math.inf, -math.inf):
            phase = _grid(0.0)
            phase[0][0] = bad
            with self.assertRaises(InputValidationError) as ctx:
                unwrap(phase)
            self.assertEqual(ctx.exception.loc, "phase[0][0]")

    def test_rejects_boolean_phase_value(self):
        phase = _grid(0.0)
        phase[0][1] = True  # bool must not sneak in as int 1
        with self.assertRaises(InputValidationError) as ctx:
            unwrap(phase)
        self.assertEqual(ctx.exception.loc, "phase[0][1]")

    def test_rejects_non_numeric(self):
        phase = _grid(0.0)
        phase[2][0] = "0.0"
        with self.assertRaises(InputValidationError) as ctx:
            unwrap(phase)
        self.assertEqual(ctx.exception.loc, "phase[2][0]")

    def test_wrapped_range_is_half_open(self):
        # +pi is excluded ...
        with self.assertRaises(InputValidationError):
            unwrap([[math.pi]])
        # ... -pi is included and anything below it is rejected.
        unwrap([[-math.pi]])
        with self.assertRaises(InputValidationError):
            unwrap([[-math.pi - 1e-12]])

    def test_rejects_wrong_container_and_shapes(self):
        with self.assertRaises(InputValidationError):
            unwrap("not a grid")  # type: ignore[arg-type]
        with self.assertRaises(InputValidationError):
            unwrap([])
        with self.assertRaises(InputValidationError):
            unwrap([[]])
        with self.assertRaises(InputValidationError):
            unwrap([[0.0], [0.0, 0.0]])
        with self.assertRaises(InputValidationError):
            unwrap([["x"] for _ in range(2)])  # type: ignore[arg-type]

    def test_rejects_oversize_grid(self):
        phase = [[0.0] * 32 for _ in range(33)]
        with self.assertRaises(InputValidationError) as ctx:
            unwrap(phase)
        self.assertEqual(ctx.exception.loc, "phase")
        # 32x32 is the boundary and must be accepted (algorithm runs).
        result = unwrap([[0.0] * 32 for _ in range(32)])
        self.assertEqual((result.rows, result.cols), (32, 32))

    def test_mask_validation(self):
        phase = _grid(0.0, 3, 3)
        with self.assertRaises(InputValidationError):
            unwrap(phase, [[True] * 3 for _ in range(2)])
        bad_mask = [[True] * 3 for _ in range(3)]
        bad_mask[1][1] = 2  # type: ignore[typeddict-item]
        with self.assertRaises(InputValidationError) as ctx:
            unwrap(phase, bad_mask)
        self.assertEqual(ctx.exception.loc, "mask[1][1]")
        # bool and plain 0/1 ints are accepted.
        result = unwrap(phase, [[1, 0, True] for _ in range(3)])
        self.assertFalse(result.mask[0][1])
        self.assertTrue(result.mask[0][0])
        self.assertTrue(result.mask[0][2])

    def test_integer_arguments_reject_bool(self):
        phase = _grid(0.0)
        with self.assertRaises(InputValidationError) as ctx:
            unwrap(phase, max_expansions=True)  # type: ignore[arg-type]
        self.assertEqual(ctx.exception.loc, "max_expansions")
        with self.assertRaises(InputValidationError):
            unwrap(phase, max_expansions=1.5)
        with self.assertRaises(InputValidationError):
            unwrap(phase, max_expansions=0)

    def test_on_infeasible_enumerated(self):
        with self.assertRaises(InputValidationError):
            unwrap(_grid(0.0), on_infeasible="silent")  # type: ignore[arg-type]

    def test_budget_exhaustion_is_distinct_from_infeasibility(self):
        # A real vortex needs a multi-layer search; one expansion is not
        # enough to decide anything, so this must be a budget error, not an
        # infeasibility claim.
        from tests._reference import analytic_vortex

        phase = analytic_vortex(6, 6, (2, 2), +1)
        with self.assertRaises(SearchBudgetExhaustedError):
            unwrap(phase, max_expansions=1)

    def test_inputs_are_not_mutated(self):
        phase = _grid(0.5)
        mask = [[True] * 4 for _ in range(4)]
        phase_copy = [row[:] for row in phase]
        mask_copy = [row[:] for row in mask]
        residue_map(phase, mask)
        unwrap(phase, mask)
        self.assertEqual(phase, phase_copy)
        self.assertEqual(mask, mask_copy)

    def test_minimal_one_by_one_grid(self):
        result = unwrap([[1.25]])
        self.assertEqual(result.components[0].cells, ((0, 0),))
        self.assertEqual(result.components[0].offsets, (0.0,))
        self.assertEqual(result.unwrapped_grid[0][0], 1.25)
        self.assertEqual(result.residues, ())


if __name__ == "__main__":
    unittest.main()
