#!/usr/bin/env python3
import argparse
import subprocess
import sys
import re
from pathlib import Path

# Filters tailored for tests/test_flash_attn.py::test_flash_attn_kvcache
# Param order per user: [seqlen_q, seqlen_k, d, has_batch_idx, has_leftpad,
# paged_kv_block_size, rotary_fraction, rotary_interleaved, seqlen_new_eq_seqlen_q,
# causal, local, alibi, new_kv, mha_type, num_splits, dtype]


def collect_nodeids(test_file: Path) -> list[str]:
    cmd = ["pytest", "-q", "--collect-only", str(test_file)]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        print("[collect] pytest collection failed:\n" + e.output, file=sys.stderr)
        sys.exit(e.returncode)
    nodeids = []
    test_file_name = Path(test_file).name
    for line in out.splitlines():
        s = line.strip()
        # Accept either relative or absolute nodeids; filter by function name and file name
        if "test_flash_attn_kvcache[" not in s:
            continue
        file_part = s.split("::", 1)[0]
        if Path(file_part).name != test_file_name:
            continue
        nodeids.append(s)
    return nodeids


def parse_params(nodeid: str) -> list[str] | None:
    try:
        l = nodeid.index("[")
        r = nodeid.rindex("]")
    except ValueError:
        return None
    inside = nodeid[l + 1 : r]
    # split by '-', strip whitespace around tokens
    parts = [p.strip() for p in inside.split("-")]
    if len(parts) != 16:
        return None
    return parts


_num_re = re.compile(r"^-?\d+$")


def _is_int(s: str) -> bool:
    return bool(_num_re.match(s))


def keep(parts: list[str], fast_only: bool = False) -> bool:
    (
        seqlen_q,
        seqlen_k,
        d,
        has_batch_idx,
        has_leftpad,
        block,
        rot_frac,
        rot_int,
        seqlen_new,
        causal,
        local,
        alibi,
        new_kv,
        mha,
        num_splits,
        dtype,
    ) = parts

    # 轻量约束，尽量匹配更多官方小用例
    if seqlen_q not in {"1", "16", "32", "64", "128"}:
        return False
    if not _is_int(seqlen_k):
        return False
    k = int(seqlen_k)
    if fast_only:
        if k > 1024:
            return False
    else:
        if k > 8192:
            return False
    if d not in {"32", "64"}:
        return False
    if has_batch_idx != "False" or has_leftpad != "False":
        return False
    # paged_kv_block_size: 允许 None 或 256 的倍数（常见 256/512/1024）
    if block != "None":
        if not _is_int(block):
            return False
        if int(block) % 256 != 0:
            return False
    # 旋转、局部注意力、alibi、新增kv：保持关闭以降低复杂度
    if rot_frac != "0.0" or rot_int != "False":
        return False
    if local != "False" or alibi != "False" or new_kv != "False":
        return False
    # seqlen_new 与 causal 放宽（两者均可），以提升命中率
    if mha not in {"mha", "gqa", "mqa"}:
        return False
    if num_splits not in {"0", "1"}:
        return False
    if dtype != "dtype0":  # 仅跑 fp16
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a small, filtered subset of flash_attn_kvcache tests with official numeric diffs printed.")
    parser.add_argument("--max", type=int, default=8, help="Max number of tests to run (default: 8)")
    parser.add_argument("--fast", action="store_true", help="Faster: restrict seqlen_k to 1024 only")
    parser.add_argument("--file", default="tests/test_flash_attn.py", help="Test file to collect from (default: tests/test_flash_attn.py)")
    parser.add_argument("--dump", type=int, default=0, help="Debug: print first N collected nodeids and parsed params, then exit")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    test_file = (root / args.file).resolve()
    if not test_file.exists():
        print(f"Test file not found: {test_file}", file=sys.stderr)
        return 2

    nodeids = collect_nodeids(test_file)
    if not nodeids:
        print("No nodes collected. Make sure pytest and the project build are OK.", file=sys.stderr)
        return 3

    if args.dump:
        print(f"Dumping first {args.dump} collected nodeids:")
        for nid in nodeids[: args.dump]:
            parts = parse_params(nid)
            print("-", nid)
            if parts:
                print("  params:", parts)
        return 0

    picked: list[str] = []
    for nid in nodeids:
        parts = parse_params(nid)
        if parts is None:
            continue
        if keep(parts, fast_only=args.fast):
            picked.append(nid)
        if len(picked) >= args.max:
            break

    if not picked:
        print("No nodes matched the filter. Try --dump 20 to inspect params, or relax filters (e.g., --fast, increase --max).", file=sys.stderr)
        return 4

    print("Selected nodes (will run with -s to show official diffs):")
    for i, nid in enumerate(picked, 1):
        print(f"  {i:2d}. {nid}")

    def rewrite_nodeid(nid: str, desired_path: Path) -> str:
        # Replace file part in nodeid with the --file path so pytest can locate it from repo root
        if "::" not in nid:
            return nid
        file_part, rest = nid.split("::", 1)
        try:
            file_name = Path(file_part).name
        except Exception:
            file_name = file_part
        if file_name == desired_path.name:
            return f"{args.file}::{rest}"  # keep the same relative path as provided in --file
        return nid

    # Run each node id one by one to keep output readable and ensure prints show up
    all_ok = True
    for nid in picked:
        fixed = rewrite_nodeid(nid, test_file)
        print("\n=== Running:", fixed)
        cmd = ["pytest", "-q", "-s", fixed, "--maxfail=1"]
        try:
            subprocess.check_call(cmd, cwd=root)
        except subprocess.CalledProcessError as e:
            all_ok = False
            print(f"Test failed (exit {e.returncode}): {fixed}", file=sys.stderr)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
