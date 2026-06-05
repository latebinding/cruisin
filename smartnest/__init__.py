"""SmartNest — Python reimplementation of the required-duration / minimum-income model.

This package reconstructs the computational core that was lost from the original
MATLAB project (``default_scenario.m`` + ``parameters.m`` survive as reference; the
8 helper functions they called were gone). Interfaces (array shapes, indexing, how
each result is consumed) are pinned by the surviving call sites; the internal
formulas are standard, documented financial-math choices and therefore *approximate*
the original model's intent rather than reproducing its exact output.

See the project README for the full list of modeling caveats.
"""

__all__ = [
    "Parameters",
    "rates",
    "contributions",
    "liability",
    "dataio",
    "scenario",
]
