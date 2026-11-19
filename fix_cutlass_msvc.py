"""Apply the CUTLASS MSVC fixes used by vortex-captain/vllm-windows."""

from pathlib import Path
import sys


def patch_cutlass(cutlass_dir: Path) -> None:
    replacements = (
        (
            "platform/platform.h",
            "#if (201703L <=__cplusplus)",
            "#if defined(_MSC_VER) || (201703L <=__cplusplus)",
        ),
        (
            "cuda_host_adapter.hpp",
            "CUTLASS_HOST_DEVICE\n  Status memsetDevice",
            "CUTLASS_HOST\n  Status memsetDevice",
        ),
    )
    for relative_path, old, new in replacements:
        path = cutlass_dir / "include" / "cutlass" / relative_path
        content = path.read_text(encoding="utf-8")
        patched = content.replace(old, new)
        if patched != content:
            path.write_text(patched, encoding="utf-8")


if __name__ == "__main__":
    patch_cutlass(Path(sys.argv[1]))
