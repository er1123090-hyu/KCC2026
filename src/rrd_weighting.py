"""Evaluation-time weighting helpers for KCC2026 RRD runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .schemas import Criterion


@dataclass(frozen=True)
class RrdWeightResult:
    mode_requested: str
    mode_used: str
    weights_by_id: dict[str, float]
    fallback_used: bool
    diagnostics: dict[str, Any]


def _project_simplex(weights: np.ndarray) -> np.ndarray:
    if weights.size == 0:
        return weights
    sorted_weights = np.sort(weights)[::-1]
    cumulative = np.cumsum(sorted_weights)
    rho_candidates = np.where(sorted_weights - (cumulative - 1) / (np.arange(len(weights)) + 1) > 0)[0]
    if rho_candidates.size == 0:
        return np.full_like(weights, 1.0 / len(weights))
    rho = rho_candidates[-1]
    theta = (cumulative[rho] - 1) / (rho + 1)
    return np.maximum(weights - theta, 0.0)


def _normalize_positive_weights(criteria: list[Criterion]) -> tuple[np.ndarray, bool]:
    raw = np.asarray([max(float(criterion.weight), 0.0) for criterion in criteria], dtype=float)
    total = float(raw.sum())
    if total <= 0:
        if not criteria:
            return np.asarray([], dtype=float), True
        return np.full(len(criteria), 1.0 / len(criteria), dtype=float), True
    return raw / total, False


def compute_llm_weights(criteria: list[Criterion]) -> RrdWeightResult:
    normalized, fallback_used = _normalize_positive_weights(criteria)
    mode_used = "uniform" if fallback_used and len(criteria) > 0 else "llm"
    return RrdWeightResult(
        mode_requested="llm",
        mode_used=mode_used,
        weights_by_id={criterion.id: float(weight) for criterion, weight in zip(criteria, normalized)},
        fallback_used=fallback_used,
        diagnostics={"rubric_count": len(criteria)},
    )


def compute_uniform_weights(criteria: list[Criterion], *, requested_mode: str = "uniform") -> RrdWeightResult:
    if not criteria:
        weights = np.asarray([], dtype=float)
    else:
        weights = np.full(len(criteria), 1.0 / len(criteria), dtype=float)
    return RrdWeightResult(
        mode_requested=requested_mode,
        mode_used="uniform",
        weights_by_id={criterion.id: float(weight) for criterion, weight in zip(criteria, weights)},
        fallback_used=False,
        diagnostics={"rubric_count": len(criteria)},
    )


def compute_wu_weights(
    criteria: list[Criterion],
    matrix: np.ndarray,
    *,
    covariance_ridge: float,
    min_covariance_samples: int,
    negative_weight_handling: str,
) -> RrdWeightResult:
    sample_count, rubric_count = matrix.shape if matrix.ndim == 2 else (0, 0)
    if rubric_count == 0:
        return compute_uniform_weights(criteria, requested_mode="wu")
    if rubric_count == 1:
        return RrdWeightResult(
            mode_requested="wu",
            mode_used="wu",
            weights_by_id={criteria[0].id: 1.0},
            fallback_used=False,
            diagnostics={"sample_count": sample_count, "rubric_count": rubric_count, "eigenvalues": [1.0]},
        )
    if sample_count < min_covariance_samples:
        result = compute_uniform_weights(criteria, requested_mode="wu")
        return RrdWeightResult(
            mode_requested="wu",
            mode_used="uniform",
            weights_by_id=result.weights_by_id,
            fallback_used=True,
            diagnostics={
                "reason": "too_few_samples",
                "sample_count": sample_count,
                "rubric_count": rubric_count,
                "min_covariance_samples": min_covariance_samples,
            },
        )

    try:
        Xf = matrix.astype(float)
        Xc = Xf - Xf.mean(axis=0, keepdims=True)
        sigma = (Xc.T @ Xc) / max(sample_count - 1, 1)
        sigma_reg = sigma + covariance_ridge * np.eye(rubric_count)
        evals, evecs = np.linalg.eigh(sigma_reg)
        safe_evals = np.maximum(evals, 1e-12)
        inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(safe_evals)) @ evecs.T
        raw = inv_sqrt @ np.ones(rubric_count)
        negative_count = int((raw < 0).sum())
        if negative_weight_handling == "simplex_project":
            projected = _project_simplex(raw)
        else:
            projected = np.clip(raw, 0.0, None)
        if projected.sum() <= 0:
            result = compute_uniform_weights(criteria, requested_mode="wu")
            return RrdWeightResult(
                mode_requested="wu",
                mode_used="uniform",
                weights_by_id=result.weights_by_id,
                fallback_used=True,
                diagnostics={
                    "reason": "degenerate_projected_weights",
                    "sample_count": sample_count,
                    "rubric_count": rubric_count,
                    "eigenvalues": safe_evals.tolist(),
                    "negative_count_before_projection": negative_count,
                    "ridge": covariance_ridge,
                },
            )
        weights = projected / projected.sum()
        return RrdWeightResult(
            mode_requested="wu",
            mode_used="wu",
            weights_by_id={criterion.id: float(weight) for criterion, weight in zip(criteria, weights)},
            fallback_used=False,
            diagnostics={
                "sample_count": sample_count,
                "rubric_count": rubric_count,
                "eigenvalues": safe_evals.tolist(),
                "negative_count_before_projection": negative_count,
                "ridge": covariance_ridge,
            },
        )
    except Exception as exc:
        result = compute_uniform_weights(criteria, requested_mode="wu")
        return RrdWeightResult(
            mode_requested="wu",
            mode_used="uniform",
            weights_by_id=result.weights_by_id,
            fallback_used=True,
            diagnostics={
                "reason": "numerical_failure",
                "error": repr(exc),
                "sample_count": sample_count,
                "rubric_count": rubric_count,
            },
        )


def apply_weight_result(criteria: list[Criterion], weight_result: RrdWeightResult) -> list[Criterion]:
    return [
        criterion.model_copy(update={"weight": float(weight_result.weights_by_id.get(criterion.id, 0.0))})
        for criterion in criteria
    ]
