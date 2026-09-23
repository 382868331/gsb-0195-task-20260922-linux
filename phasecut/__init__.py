"""phasecut: deterministic residual branch-cut phase unwrapping.

Public API (see module ``phasecut.core`` and README):

* :func:`unwrap`            -- main entry point, returns :class:`UnwrapResult`
* :func:`residue_map`       -- integer residue of every valid 2x2 loop
* :func:`wrapped_difference`-- wrap an edge difference into [-pi, pi)
* data classes: :class:`Edge`, :class:`Residue`, :class:`Dipole`,
  :class:`BoundaryLeash`, :class:`Component`, :class:`UnsatisfiedResidue`,
  :class:`EdgeResidual`, :class:`UnwrapResult`
* errors: :class:`PhaseCutError`, :class:`InputValidationError`,
  :class:`UnwrapInfeasibleError`, :class:`SearchBudgetExhaustedError`
"""

from .core import (
    BoundaryLeash,
    Component,
    CONTRADICTION_TOL,
    Dipole,
    Edge,
    EdgeResidual,
    InputValidationError,
    MAX_SIDE,
    PhaseCutError,
    Residue,
    SearchBudgetExhaustedError,
    TAU,
    UnsatisfiedResidue,
    UnwrapInfeasibleError,
    UnwrapResult,
    residue_map,
    unwrap,
    wrapped_difference,
)

__all__ = [
    "BoundaryLeash",
    "Component",
    "CONTRADICTION_TOL",
    "Dipole",
    "Edge",
    "EdgeResidual",
    "InputValidationError",
    "MAX_SIDE",
    "PhaseCutError",
    "Residue",
    "SearchBudgetExhaustedError",
    "TAU",
    "UnsatisfiedResidue",
    "UnwrapInfeasibleError",
    "UnwrapResult",
    "residue_map",
    "unwrap",
    "wrapped_difference",
]
