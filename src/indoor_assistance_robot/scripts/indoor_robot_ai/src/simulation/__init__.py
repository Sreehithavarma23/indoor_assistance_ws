"""Simulation dry-run adapter for verified ROS 2 interfaces only."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.simulation.dry_run import DryRunAdapter, DryRunResult

__all__ = ["DryRunAdapter", "DryRunResult"]


def __getattr__(name: str):
    if name in {"DryRunAdapter", "DryRunResult"}:
        from src.simulation import dry_run as _dry_run

        return getattr(_dry_run, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
