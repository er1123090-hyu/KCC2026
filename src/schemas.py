"""Pydantic schemas for the Korean rubric grounding main experiment."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictBaseModel(BaseModel):
    """Base model with strict extra-field rejection."""

    model_config = ConfigDict(extra="forbid")


class MetaEvalPair(StrictBaseModel):
    pair_id: str
    prompt_id: str
    dataset: str
    subset: str | None
    prompt: str
    response_a: str
    response_b: str
    gold_preference: Literal["A", "B", "tie"]
    metadata: dict[str, Any] = Field(default_factory=dict)


class PreferenceExemplar(StrictBaseModel):
    exemplar_id: str
    prompt_id: str
    dataset: str
    prompt: str
    chosen: str
    rejected: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AuxiliarySample(StrictBaseModel):
    prompt_id: str
    sample_id: str
    response: str
    generator_model: str
    sampling_params: dict[str, Any] = Field(default_factory=dict)


class Criterion(StrictBaseModel):
    id: str
    axis: str
    importance: Literal["essential", "important", "optional"]
    weight: float
    polarity: Literal["positive", "negative"]
    text_ko: str
    self_contained: bool


class RaRRubricItem(StrictBaseModel):
    title: str
    description: str
    weight: int


class Rubric(StrictBaseModel):
    prompt_id: str
    pair_id: str | None = None
    method: str
    generator_family: str
    generator_model: str
    criteria: list[Criterion] = Field(default_factory=list)
    rar_items: list[RaRRubricItem] = Field(default_factory=list)
    generation_metadata: dict[str, Any] = Field(default_factory=dict)


class JudgeScore(StrictBaseModel):
    pair_id: str
    prompt_id: str
    response_side: Literal["A", "B"]
    method: str
    generator_family: str | None
    generator_model: str | None
    evaluator_model: str
    rating: int
    normalized_score: float
    reason: str
    parse_failure: bool
    raw_output: str


class PairPrediction(StrictBaseModel):
    pair_id: str
    method: str
    generator_family: str | None
    generator_model: str | None
    eval_protocol: Literal["pairwise_judge", "binary_rubric_aggregation"]
    pred_preference: Literal["A", "B"]
    gold_preference: Literal["A", "B"]
    score_a: float | None = None
    score_b: float | None = None
    justification: str | None = None
    parse_failure: bool | None = None
    raw_output: str | None = None
    weighting_mode_requested: str | None = None
    weighting_mode_used: str | None = None
    weighting_fallback: bool | None = None
