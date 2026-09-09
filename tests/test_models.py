from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models import FilterConfig, JobCreateRequest, LLMConfig, LLMExpansionConfig


class TestJobCreateRequestValidation:
    def _base_kwargs(self, **overrides):
        kwargs = dict(queries=["cats"], output_folder="C:/out")
        kwargs.update(overrides)
        return kwargs

    def test_defaults_are_accepted(self):
        req = JobCreateRequest(**self._base_kwargs())
        assert req.n_per_query == 10
        assert req.concurrency == 8

    def test_zero_n_per_query_is_rejected(self):
        with pytest.raises(ValidationError):
            JobCreateRequest(**self._base_kwargs(n_per_query=0))

    def test_negative_n_per_query_is_rejected(self):
        with pytest.raises(ValidationError):
            JobCreateRequest(**self._base_kwargs(n_per_query=-5))

    def test_zero_concurrency_is_rejected(self):
        with pytest.raises(ValidationError):
            JobCreateRequest(**self._base_kwargs(concurrency=0))

    def test_positive_concurrency_is_accepted(self):
        req = JobCreateRequest(**self._base_kwargs(concurrency=1))
        assert req.concurrency == 1


class TestLLMExpansionConfig:
    def test_negative_variations_is_rejected(self):
        with pytest.raises(ValidationError):
            LLMExpansionConfig(variations_per_query=-1)

    def test_zero_variations_is_allowed(self):
        # 0 is a legitimate "don't expand" value, distinct from disabling entirely.
        cfg = LLMExpansionConfig(variations_per_query=0)
        assert cfg.variations_per_query == 0


class TestLLMConfig:
    def test_non_positive_timeout_is_rejected(self):
        with pytest.raises(ValidationError):
            LLMConfig(timeout_seconds=0)
        with pytest.raises(ValidationError):
            LLMConfig(timeout_seconds=-1)

    def test_positive_timeout_is_accepted(self):
        assert LLMConfig(timeout_seconds=30).timeout_seconds == 30


class TestFilterConfig:
    def test_negative_min_dimensions_are_rejected(self):
        with pytest.raises(ValidationError):
            FilterConfig(min_width=-1)
        with pytest.raises(ValidationError):
            FilterConfig(min_height=-1)

    def test_zero_min_dimensions_are_allowed(self):
        cfg = FilterConfig(min_width=0, min_height=0)
        assert cfg.min_width == 0
        assert cfg.min_height == 0
