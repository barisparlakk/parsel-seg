import pytest

from parsel.utils.config import apply_overrides


def _cfg():
    return {"data": {"batch_size": 8, "num_workers": 4}, "train": {"lr": 0.001}}


def test_overrides_values_with_yaml_types():
    cfg = apply_overrides(_cfg(), ["data.batch_size=16", "train.lr=3e-4", "data.num_workers=null"])
    assert cfg["data"]["batch_size"] == 16 and isinstance(cfg["data"]["batch_size"], int)
    assert cfg["train"]["lr"] == pytest.approx(3e-4)
    assert cfg["data"]["num_workers"] is None


def test_new_leaf_key_is_allowed():
    assert apply_overrides(_cfg(), ["train.max_train_batches=5"])["train"]["max_train_batches"] == 5


def test_unknown_section_raises():
    with pytest.raises(KeyError, match="no section"):
        apply_overrides(_cfg(), ["dta.batch_size=1"])


def test_leaf_value_is_not_a_section():
    with pytest.raises(KeyError, match="not a section"):
        apply_overrides(_cfg(), ["train.lr.x=1"])


def test_missing_equals_raises():
    with pytest.raises(ValueError):
        apply_overrides(_cfg(), ["data.batch_size"])


def test_string_value_and_equals_inside_value():
    cfg = apply_overrides(_cfg(), ["data.manifest=a=b"])
    assert cfg["data"]["manifest"] == "a=b"
