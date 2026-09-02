"""Package-boundary tests for co-installed FA2 and FA4 distributions."""

import importlib
from types import SimpleNamespace

import pytest

import flash_attn


def test_fa4_child_package_does_not_eagerly_import_legacy_fa2(monkeypatch):
    def fail_legacy_import(*args, **kwargs):
        raise AssertionError("FA2 interface must stay lazy while importing FA4")

    monkeypatch.setattr(flash_attn, "import_module", fail_legacy_import)

    cute = importlib.import_module("flash_attn.cute")

    assert cute.__name__ == "flash_attn.cute"


def test_legacy_top_level_api_loads_and_caches_when_present(monkeypatch):
    name = "flash_attn_func"
    expected = object()
    legacy_interface = SimpleNamespace(**{name: expected})

    monkeypatch.delitem(flash_attn.__dict__, name, raising=False)
    monkeypatch.setattr(flash_attn, "import_module", lambda *args: legacy_interface)

    assert getattr(flash_attn, name) is expected
    assert getattr(flash_attn, name) is expected


def test_missing_legacy_extension_has_an_explicit_error(monkeypatch):
    def missing_legacy_extension(*args, **kwargs):
        raise ModuleNotFoundError(
            "No module named 'flash_attn_2_cuda'", name="flash_attn_2_cuda"
        )

    monkeypatch.delitem(flash_attn.__dict__, "flash_attn_func", raising=False)
    monkeypatch.setattr(flash_attn, "import_module", missing_legacy_extension)

    with pytest.raises(ModuleNotFoundError, match="requires the FA2 extension"):
        flash_attn.flash_attn_func


def test_non_legacy_import_error_is_not_rewritten(monkeypatch):
    expected = ModuleNotFoundError(
        "No module named 'another_dependency'", name="another_dependency"
    )

    def missing_other_dependency(*args, **kwargs):
        raise expected

    monkeypatch.delitem(flash_attn.__dict__, "flash_attn_func", raising=False)
    monkeypatch.setattr(flash_attn, "import_module", missing_other_dependency)

    with pytest.raises(ModuleNotFoundError) as raised:
        flash_attn.flash_attn_func

    assert raised.value is expected
