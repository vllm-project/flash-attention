__version__ = "2.6.2"

# Use relative import to support build-from-source installation in vLLM
from .flash_attn_interface import (
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)
