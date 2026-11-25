# Copyright (c) 2024-2026, FlashAttention Authors and Contributors.
# Verification test ensuring MSVC preprocessor conformance (/Zc:preprocessor)
# Ref: ISO C++ §19.3 / ISO C99 §6.10.3 (no directives inside macro arguments)

import re
from pathlib import Path
import pytest


def test_no_preprocessor_directives_in_macro_arguments():
    """Verify that no preprocessor directives are embedded within function-like macro invocations."""
    repo_root = Path(__file__).resolve().parent.parent
    hopper_dir = repo_root / "hopper"

    macro_names = [
        "BOOL_SWITCH",
        "ARCH_SWITCH",
        "SPLIT_SWITCH",
        "PAGEDKV_SWITCH",
        "PACKGQA_SWITCH",
        "SOFTCAP_SWITCH",
        "FP16_SWITCH",
        "HEADDIM_SWITCH",
        "VARLEN_SWITCH",
        "CLUSTER_SWITCH",
        "VCOLMAJOR_SWITCH",
        "APPENDKV_SWITCH",
        "CAUSAL_LOCAL_SWITCH",
    ]

    violations = []

    for src_file in hopper_dir.glob("*.[ch]*"):
        if src_file.suffix not in (".cpp", ".cu", ".h", ".cuh"):
            continue

        with open(src_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        in_macro = False
        macro_paren_depth = 0
        curr_macro = ""
        macro_start_line = 0

        for line_no, line in enumerate(lines, 1):
            stripped = line.strip()

            if not in_macro:
                for m in macro_names:
                    if re.search(r"\b" + m + r"\s*\(", line) and not stripped.startswith("#define"):
                        in_macro = True
                        curr_macro = m
                        macro_start_line = line_no
                        macro_paren_depth = 0
                        break

            if in_macro:
                if stripped.startswith("#") and not stripped.startswith("#include"):
                    violations.append(
                        f"{src_file.name}:{line_no} inside {curr_macro} (line {macro_start_line}): {stripped}"
                    )

                macro_paren_depth += line.count("(") - line.count(")")
                if macro_paren_depth <= 0 and line_no > macro_start_line:
                    in_macro = False

    assert not violations, (
        f"Found {len(violations)} preprocessor directives inside macro arguments (breaks MSVC /Zc:preprocessor):\n"
        + "\n".join(violations)
    )


def test_required_build_guards_preserved():
    """Verify that all compile-time feature guards are preserved in hopper/flash_api.cpp."""
    repo_root = Path(__file__).resolve().parent.parent
    flash_api_path = repo_root / "hopper" / "flash_api.cpp"

    content = flash_api_path.read_text(encoding="utf-8")

    expected_guards = [
        "FLASHATTENTION_PACKGQA_ONLY",
        "FLASHATTENTION_DISABLE_FP16",
        "FLASHATTENTION_DISABLE_FP8",
        "FLASHATTENTION_DISABLE_HDIM64",
        "FLASHATTENTION_DISABLE_HDIM96",
        "FLASHATTENTION_DISABLE_HDIM128",
        "FLASHATTENTION_DISABLE_HDIM192",
        "FLASHATTENTION_DISABLE_HDIM256",
        "FLASHATTENTION_DISABLE_HDIMDIFF64",
        "FLASHATTENTION_DISABLE_HDIMDIFF192",
        "FLASHATTENTION_DISABLE_BACKWARD",
    ]

    for guard in expected_guards:
        assert guard in content, f"Expected guard {guard} was not found in hopper/flash_api.cpp"
