"""
__init__.py — Public API for the compressor_observability package.

Exposes run() so callers never need to import from agent.py directly:
    from compressor_observability import run
"""

from compressor_observability.agent import run

__all__ = ["run"]