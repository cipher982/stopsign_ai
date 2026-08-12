"""Tests for vehicle_labels display formatting and confidence gating."""

from stopsign.database import MAKE_MODEL_CONFIDENCE_MIN
from stopsign.database import _format_make_model


def test_format_make_model_composes_title_cased_make_and_preserves_model():
    assert _format_make_model("toyota", "Camry") == "Toyota Camry"
    assert _format_make_model("ford", "F-150") == "Ford F-150"
    assert _format_make_model("honda", "CR-V") == "Honda CR-V"


def test_format_make_model_handles_snake_case_and_acronym_makes():
    assert _format_make_model("mercedes_benz", "C-Class") == "Mercedes Benz C-Class"
    assert _format_make_model("bmw", "X5") == "BMW X5"
    assert _format_make_model("gmc", None) == "GMC"


def test_format_make_model_drops_noop_buckets():
    assert _format_make_model("unknown", None) is None
    assert _format_make_model("other", "foo") is None
    assert _format_make_model(None, None) is None


def test_format_make_model_make_only_without_model():
    assert _format_make_model("toyota", None) == "Toyota"


def test_format_make_model_confidence_gate():
    below = MAKE_MODEL_CONFIDENCE_MIN - 0.01
    above = MAKE_MODEL_CONFIDENCE_MIN + 0.01
    assert _format_make_model("toyota", "Camry", confidence=below) is None
    assert _format_make_model("toyota", "Camry", confidence=above) == "Toyota Camry"
    # No confidence passed -> no gating (aggregate views filter in SQL).
    assert _format_make_model("toyota", "Camry") == "Toyota Camry"
