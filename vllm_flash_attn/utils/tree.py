# Copyright (c) 2025, Samsung SDSA.

import math
import torch
from einops import rearrange


def create_mask(tree: list[int]):
    out = []
    for node in tree:
        out = [o << 1 for o in out]
        mask = 1
        if node != -1:
            mask |= out[node]
        out.append(mask)
    return out


def simple_mask(spec_len, num_seq):
    mask = []
    for seq in range(num_seq):
        pos = -1
        for s in range(spec_len):
            mask.append(pos)
            pos = len(mask)-1
    return mask


def create_tree_mask(speclens, device):
    outputs: list[torch.Tensor] = []
    for spec_len, spec_branchs in speclens:
        outputs.append(torch.tensor(create_mask(simple_mask(spec_len, spec_branchs)), dtype=torch.uint64, device=device))
    return torch.cat(outputs, dim=0)


def create_spec_tree(input_base: torch.Tensor, spec_base: torch.Tensor, seqlens: list[int], speclens: list[tuple[int, int]]):
    seq_idx = 0
    spec_idx = 0
    outputs = []
    for seq_len, (spec_len, spec_branchs) in zip(seqlens, speclens):
        total_spec_len = spec_len * spec_branchs
        outputs += [input_base[seq_idx:seq_idx+seq_len], spec_base[spec_idx:spec_idx+total_spec_len]]
        seq_idx += seq_len
        spec_idx += total_spec_len
    return torch.cat(outputs, dim=0)


def create_spec_batch(input_base: torch.Tensor, spec_base: torch.Tensor, seqlens: list[int], speclens: list[tuple[int, int]]):
    seq_idx=0
    spec_idx=0
    outputs = []
    for seq_len, (spec_len, spec_branchs) in zip(seqlens, speclens):
        for j in range(spec_branchs):
            outputs += [input_base[seq_idx:seq_idx+seq_len], spec_base[spec_idx:spec_idx+spec_len]]
            spec_idx += spec_len
        seq_idx += seq_len
    return torch.cat(outputs, dim=0)


def tree_seqlens(seqlens, speclens):
    return [i+j*b for i, (j, b) in zip(seqlens, speclens)]


def batch_seqlens(seqlens, speclens):
    seqlens_batch = []
    for i, (j, s) in zip(seqlens, speclens):
        seqlens_batch += [i+j]*s
    return seqlens_batch


def unshuffle_indices(shuffled_indices):
    n = len(shuffled_indices)
    inverse_indices = [None] * n

    for original_pos, shuffled_pos in enumerate(shuffled_indices):
        inverse_indices[shuffled_pos] = original_pos
    return inverse_indices


def create_block_shuffle(seqlens, paged_kv_block_size, device):
    num_blocks = sum(math.ceil(seq / paged_kv_block_size) for seq in seqlens)
    block_table = torch.zeros((len(seqlens), math.ceil(max(seqlens)/ paged_kv_block_size)), dtype=torch.int32, device=device)
    block_shuffle = torch.randperm(num_blocks, dtype=torch.int32, device=device)
    block_unshuffle = torch.tensor(unshuffle_indices(block_shuffle), dtype=torch.int32, device=device)
    block_idx = 0
    for i, length in enumerate(seqlens):
        blocks_in_seq = math.ceil(length/paged_kv_block_size)
        block_table[i, :blocks_in_seq] = block_unshuffle[block_idx:block_idx+blocks_in_seq]
        block_idx += blocks_in_seq
    return block_table, block_shuffle


def to_paged_blocks(sequence: torch.Tensor, seqlens: list[int], block_size: int, nheads: int, d: int, block_table: torch.Tensor):
    num_blocks = sum(math.ceil(length / block_size) for length in seqlens)
    block_tensor = torch.empty(
        num_blocks*block_size, nheads, d, device=sequence.device, dtype=sequence.dtype
    )

    bt_idx = 0
    seq_idx = 0
    for seqlen in seqlens:
        block_tensor[bt_idx:bt_idx+seqlen] = sequence[seq_idx:seq_idx+seqlen]
        rem = block_size - (seqlen % block_size) if seqlen % block_size != 0 else 0
        bt_idx += seqlen + rem
        seq_idx += seqlen

    block_tensor = rearrange(block_tensor, "(num_blocks blocksize) nhead d -> num_blocks blocksize nhead d", blocksize=block_size)
    # shuffle blocks based on block table
    block_tensor = block_tensor.index_select(0, block_table)
    return block_tensor


def generate_q_and_block_kvcache(q_seqlens: list[int], k_seqlens: list[int], speclens: list[tuple[int, int]], paged_kv_block_size: int, nheads: int, d: int, device, dtype):
    # create the input base and individual spec branchs
    q_input_base = torch.randn(sum(q_seqlens), nheads, d, device=device, dtype=dtype)
    q_spec_base = torch.randn(sum(a * b for a, b in speclens), nheads, d, device=device, dtype=dtype)

    # from the bases create the q for tree attention
    q_spec_tree = create_spec_tree(q_input_base, q_spec_base, q_seqlens, speclens)
    q_seqlens_tree = tree_seqlens(q_seqlens, speclens)

    # from the bases create the q for varlen attention
    q_spec_batch = create_spec_batch(q_input_base, q_spec_base, q_seqlens, speclens)
    q_seqlens_batch = batch_seqlens(q_seqlens, speclens)

    del q_input_base
    del q_spec_base

    # create the k base and individual spec branches
    k_input_base = torch.randn(sum(k_seqlens), nheads, d, device=device, dtype=dtype)
    k_spec_base = torch.randn(sum(a * b for a, b in speclens), nheads, d, device=device, dtype=dtype)

    # from the bases create the k for tree attention
    k_tree = create_spec_tree(k_input_base, k_spec_base, k_seqlens, speclens)
    k_seqlens_tree = tree_seqlens(k_seqlens, speclens)
    tree_block_table, tree_block_shuffle = create_block_shuffle(k_seqlens_tree, paged_kv_block_size, device)
    k_spec_tree = to_paged_blocks(k_tree, k_seqlens_tree, paged_kv_block_size, nheads, d, tree_block_shuffle)

    # from the bases create the k for varlen attention
    k_batch = create_spec_batch(k_input_base, k_spec_base, k_seqlens, speclens)
    k_seqlens_batch = batch_seqlens(k_seqlens, speclens)
    batch_block_table, batch_block_shuffle = create_block_shuffle(k_seqlens_batch, paged_kv_block_size, device)
    k_spec_batch = to_paged_blocks(k_batch, k_seqlens_batch, paged_kv_block_size, nheads, d, batch_block_shuffle)

    del k_input_base
    del k_spec_base
    del k_tree
    del k_batch

    # create the v base and individual spec branches
    v_input_base = torch.randn(sum(k_seqlens), nheads, d, device=device, dtype=dtype)
    v_spec_base = torch.randn(sum(a * b for a, b in speclens), nheads, d, device=device, dtype=dtype)

    # from the bases create the v for tree attention
    v_tree = create_spec_tree(v_input_base, v_spec_base, k_seqlens, speclens)
    v_spec_tree = to_paged_blocks(v_tree, k_seqlens_tree, paged_kv_block_size, nheads, d, tree_block_shuffle)

    # from the bases create the v for varlen attention
    v_batch = create_spec_batch(v_input_base, v_spec_base, k_seqlens, speclens)
    v_spec_batch = to_paged_blocks(v_batch, k_seqlens_batch, paged_kv_block_size, nheads, d, batch_block_shuffle)

    del v_input_base
    del v_spec_base
    del v_tree
    del v_batch

    return q_spec_tree, q_seqlens_tree, q_spec_batch, q_seqlens_batch, tree_block_table, k_spec_tree, v_spec_tree, k_seqlens_tree, batch_block_table, k_spec_batch, v_spec_batch, k_seqlens_batch


def treeify_output(t: torch.Tensor, seqlens: int, speclens: int) -> torch.Tensor:
    out = torch.empty(sum(i+j*s for i, (j,s) in zip(seqlens, speclens)), t.shape[1], t.shape[2], device=t.device, dtype=t.dtype)
    input_idx = 0
    output_idx = 0
    for seq_len, (spec_len, spec_branchs) in zip(seqlens, speclens):
        out[output_idx:output_idx+seq_len] = t[input_idx:input_idx+seq_len]
        output_idx += seq_len
        for _ in range(spec_branchs):
            input_idx += seq_len
            out[output_idx:output_idx+spec_len] = t[input_idx:input_idx+spec_len]
            output_idx += spec_len
            input_idx += spec_len     
    return out


def deblockify(t:torch.Tensor, block_table: torch.Tensor, seqlens: list[int]):
    out = []
    for i, seqlen in enumerate(seqlens):
        temp = rearrange(t.index_select(0, block_table[i]), "num_blocks blocksize nhead d -> (num_blocks blocksize) nhead d")
        out.append(temp[:seqlen])
    return torch.cat(out, dim=0)