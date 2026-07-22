from pkgutil import extend_path
from importlib import import_module

# look for every subdir with flash_attn base name such that fa2 and fa4 can be co-installed
__path__ = extend_path(__path__, __name__)

__version__ = "2.8.4"

_LEGACY_FA2_EXPORTS = (
    "flash_attn_func",
    "flash_attn_kvpacked_func",
    "flash_attn_qkvpacked_func",
    "flash_attn_varlen_func",
    "flash_attn_varlen_kvpacked_func",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_with_kvcache",
)

__all__ = ["__version__", *_LEGACY_FA2_EXPORTS]


def _load_legacy_fa2_interface():
    """Import the FA2 compatibility API only when a legacy symbol is used.

    FA4 is distributed as a child package of the co-installable ``flash_attn``
    namespace.  It must not require the separately-built FA2 extension merely
    to import ``flash_attn.cute``.  Keep the legacy API lazy, while retaining
    its import-time error when a caller actually requests a legacy symbol.
    """
    try:
        return import_module(".flash_attn_interface", __name__)
    except ModuleNotFoundError as error:
        if error.name != "flash_attn_2_cuda":
            raise
        raise ModuleNotFoundError(
            "The legacy flash_attn top-level API requires the FA2 extension "
            "'flash_attn_2_cuda'. Install a package that provides it, or use "
            "flash_attn.cute for FA4."
        ) from error


def __getattr__(name):
    if name not in _LEGACY_FA2_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    legacy_interface = _load_legacy_fa2_interface()
    exported = getattr(legacy_interface, name)
    globals()[name] = exported
    return exported


def __dir__():
    return sorted({*globals(), *_LEGACY_FA2_EXPORTS})
