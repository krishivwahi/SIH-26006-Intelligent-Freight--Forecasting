"""
src/solver/__init__.py
Solver package for the AI Maritime Chartering Decision Engine.
Exposes the public API: solve() and AssignmentResult.
"""
from src.solver.solver import solve, AssignmentResult

__all__ = ["solve", "AssignmentResult"]
