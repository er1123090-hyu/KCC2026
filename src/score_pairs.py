"""Backward-compatible wrapper around the pairwise evaluation entrypoint."""

from __future__ import annotations

from .evaluate_pairs import main


if __name__ == "__main__":
    main()
