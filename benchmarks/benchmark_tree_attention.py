# Copyright (c) 2025, Samsung SDSA.

import random
import torch

from vllm_flash_attn.utils.benchmark import benchmark_forward

from vllm_flash_attn.flash_attn_interface import (
    flash_attn_varlen_func,
    tree_attention,
)
from vllm_flash_attn.utils.tree import (
    create_tree_mask,
    generate_q_and_block_kvcache,
    treeify_output,
)


def run_tree_attention_benchmark(
    seqlen_q: int = 1024,
    seqlen_k: int = 1024,
    spec_len: tuple[int] = (8,8),
    random_seq_len: bool = False,
    random_spec_len: bool = False,
    batch_size: int = 8,
    nheads: int = 16,
    head_dim: int = 128,
    paged_kv_block_size: int = 256,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
):
    """
    Benchmark tree_attention vs flash_attn_varlen_func performance.
    
    Similar to test_paged_tree_attention but focused on performance measurement.
    """
    print("Benchmarking with:")
    print(f"  seqlen_q: {seqlen_q}, seqlen_k: {seqlen_k}")
    print(f"  spec_len: {spec_len}, random_seq_len: {random_seq_len}, random_spec_len: {random_spec_len}")
    print(f"  batch_size: {batch_size}, nheads: {nheads}, head_dim: {head_dim}")
    print(f"  paged_kv_block_size: {paged_kv_block_size}, dtype: {dtype}")
    
    torch.set_default_device(device)
    torch.cuda.manual_seed_all(42)  # Fixed seed for reproducibility
    
    # Generate random sequence lengths and spec lengths similar to the test
    if random_seq_len:
        q_seqlens = [seqlen_q + random.randint(0, 20) for _ in range(batch_size)]
        k_seqlens = [seqlen_k + random.randint(0, 20) for _ in range(batch_size)]
    else:
        q_seqlens = [seqlen_q]*batch_size
        k_seqlens = [seqlen_k]*batch_size
    
    if random_spec_len:
        speclens = [(spec_len[0]+random.randint(0, 7), spec_len[1]+random.randint(1, 2)) for _ in range(batch_size)]
    else:
        speclens = [spec_len]*batch_size
    
    # Generate test data using the utility function
    (
        q_spec_tree,
        q_seqlens_tree,
        q_spec_batch,
        q_seqlens_batch,
        tree_block_table,
        k_spec_tree,
        v_spec_tree,
        k_seqlens_tree,
        batch_block_table,
        k_spec_batch,
        v_spec_batch,
        k_seqlens_batch,
    ) = generate_q_and_block_kvcache(
        q_seqlens, k_seqlens, speclens, paged_kv_block_size, nheads, head_dim, device, dtype
    )
    
    # Create tree mask and cumulative sequence lengths
    tree_mask = create_tree_mask(speclens, device)
    tree_mask_lens = torch.tensor([0] + [i*j for i,j in speclens], dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens_q_tree = torch.tensor([0] + q_seqlens_tree, dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)
    seqused_k_tree = torch.tensor(k_seqlens_tree, dtype=torch.int32)
    cu_seqlens_q_batch = torch.tensor([0] + q_seqlens_batch, dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)
    seqused_k_batch = torch.tensor(k_seqlens_batch, dtype=torch.int32)
    
    
    print("\nRunning benchmarks...")
    
    # Benchmark tree_attention
    _, tree_measurement = benchmark_forward(
        tree_attention,
        q_spec_tree,
        k_spec_tree,
        v_spec_tree,
        max(q_seqlens_tree),
        cu_seqlens_q_tree,
        max(k_seqlens_tree),
        tree_mask,
        tree_mask_lens,
        seqused_k=seqused_k_tree,
        block_table=tree_block_table,
        desc="tree_attention",
        verbose=False
    )
    tree_time = tree_measurement.mean
    print(f"tree_attention average time: {tree_time:.6f} seconds")
    
    # Benchmark flash_attn_varlen_func
    _, varlen_measurement = benchmark_forward(
        flash_attn_varlen_func,
        q_spec_batch,
        k_spec_batch,
        v_spec_batch,
        max(q_seqlens_batch),
        cu_seqlens_q_batch,
        max(k_seqlens_batch),
        seqused_k=seqused_k_batch,
        causal=True,
        block_table=batch_block_table,
        desc="flash_attn_varlen_func",
        verbose=False
    )
    varlen_time = varlen_measurement.mean
    print(f"flash_attn_varlen_func average time: {varlen_time:.6f} seconds")
    
    # Calculate speedup
    if varlen_time > 0:
        speedup = varlen_time / tree_time
        print(f"Speedup (varlen/tree): {speedup:.2f}x")
        if speedup > 1:
            print(f"tree_attention is {speedup:.2f}x faster")
        else:
            print(f"flash_attn_varlen_func is {1/speedup:.2f}x faster")
    
    # Verify correctness
    print("\nVerifying correctness...")
    tree_output = tree_attention(
        q_spec_tree,
        k_spec_tree,
        v_spec_tree,
        max(q_seqlens_tree),
        cu_seqlens_q_tree,
        max(k_seqlens_tree),
        tree_mask,
        tree_mask_lens,
        seqused_k=seqused_k_tree,
        block_table=tree_block_table,
    )
    varlen_output = flash_attn_varlen_func(
        q_spec_batch,
        k_spec_batch,
        v_spec_batch,
        max(q_seqlens_batch),
        cu_seqlens_q_batch,
        max(k_seqlens_batch),
        seqused_k=seqused_k_batch,
        causal=True,
        block_table=batch_block_table,
    )
    varlen_output_treeified = treeify_output(varlen_output, q_seqlens, speclens)
    try:
        torch.testing.assert_close(tree_output, varlen_output_treeified, atol=2e-2, rtol=1e-2)
    except AssertionError as e:
        print("✗ Outputs differ significantly!")
        print(e)
    else:
        print("✓ Outputs match within tolerance")
    finally:
        max_diff = torch.max(torch.abs(tree_output - varlen_output_treeified)).item()
        print(f"Maximum difference between outputs: {max_diff:.6f}")
    
    return {
        'tree_time': tree_time,
        'varlen_time': varlen_time,
        'speedup': varlen_time / tree_time if varlen_time > 0 else float('inf'),
        'max_diff': max_diff,
        'config': {
            'seqlen_q': seqlen_q,
            'seqlen_k': seqlen_k,
            'batch_size': batch_size,
            'nheads': nheads,
            'head_dim': head_dim,
            'paged_kv_block_size': paged_kv_block_size,
            'dtype': str(dtype),
            'q_spec_tree.shape': q_spec_tree.shape,
            'k_spec_tree.shape': k_spec_tree.shape,
            'tree_mask.shape': tree_mask.shape,
        }
    }


def run_decoding_benchmark():
    """Run benchmarks for decoding scenario with seqlen_q=0."""
    configs = [
        # Small sequences with different spec_len and block sizes
        {'seqlen_q': 0, 'seqlen_k': 128, 'batch_size': 4, 'nheads': 8, 'head_dim': 128, 'spec_len': (1, 2), 'paged_kv_block_size': 16},
        {'seqlen_q': 0, 'seqlen_k': 256, 'batch_size': 4, 'nheads': 8, 'head_dim': 128, 'spec_len': (2, 3), 'paged_kv_block_size': 16},
        
        # Medium sequences with varied spec_len and block sizes
        {'seqlen_q': 0, 'seqlen_k': 512, 'batch_size': 8, 'nheads': 16, 'head_dim': 128, 'spec_len': (1, 2), 'paged_kv_block_size': 256},
        {'seqlen_q': 0, 'seqlen_k': 1024, 'batch_size': 8, 'nheads': 16, 'head_dim': 128, 'spec_len': (3, 4), 'paged_kv_block_size': 256},
        
        # Large sequences with larger block sizes
        {'seqlen_q': 0, 'seqlen_k': 2048, 'batch_size': 4, 'nheads': 16, 'head_dim': 128, 'spec_len': (2, 3), 'paged_kv_block_size': 512},
        
        # Different head dimensions with varied block sizes
        {'seqlen_q': 0, 'seqlen_k': 1024, 'batch_size': 8, 'nheads': 16, 'head_dim': 64, 'spec_len': (1, 2), 'paged_kv_block_size': 256},
        {'seqlen_q': 0, 'seqlen_k': 1024, 'batch_size': 8, 'nheads': 16, 'head_dim': 256, 'spec_len': (2, 3), 'paged_kv_block_size': 512},
        
        # Different batch sizes with randomization and block sizes
        {'seqlen_q': 0, 'seqlen_k': 1024, 'batch_size': 2, 'nheads': 16, 'head_dim': 128, 'spec_len': (1, 2), 'random_spec_len': True, 'paged_kv_block_size': 16},
        {'seqlen_q': 0, 'seqlen_k': 1024, 'batch_size': 16, 'nheads': 16, 'head_dim': 128, 'spec_len': (2, 3), 'random_seq_len': True, 'paged_kv_block_size': 256},
        
        # High spec_len scenarios with different block sizes
        {'seqlen_q': 0, 'seqlen_k': 1024, 'batch_size': 8, 'nheads': 16, 'head_dim': 128, 'spec_len': (4, 5), 'paged_kv_block_size': 256},
        {'seqlen_q': 0, 'seqlen_k': 1024, 'batch_size': 8, 'nheads': 16, 'head_dim': 128, 'spec_len': (6, 8), 'paged_kv_block_size': 512},
        
        # Block size comparison scenarios
        {'seqlen_q': 0, 'seqlen_k': 1024, 'batch_size': 8, 'nheads': 16, 'head_dim': 128, 'spec_len': (2, 3), 'paged_kv_block_size': 16},
        {'seqlen_q': 0, 'seqlen_k': 1024, 'batch_size': 8, 'nheads': 16, 'head_dim': 128, 'spec_len': (2, 3), 'paged_kv_block_size': 256},
        {'seqlen_q': 0, 'seqlen_k': 1024, 'batch_size': 8, 'nheads': 16, 'head_dim': 128, 'spec_len': (2, 3), 'paged_kv_block_size': 512},
    ]
    
    print("=" * 80)
    print("DECODING BENCHMARK (seqlen_q=0)")
    print("=" * 80)
    print("This benchmark represents the decoding scenario where tree attention")
    print("can be compared against batch expansion for generation tasks.")
    print("=" * 80)
    
    results = []
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Decoding Configuration:")
        result = run_tree_attention_benchmark(**config)
        results.append(result)
        print("-" * 80)
    
    # Summary
    print("\n" + "=" * 80)
    print("DECODING BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Config':<18} {'Tree(ms)':<10} {'Varlen(ms)':<12} {'Speedup':<10} {'Max Diff':<12}")
    print("-" * 80)
    
    for i, result in enumerate(results):
        config = result['config']
        config_str = f"{config['seqlen_q']}:{config['seqlen_k']}:{config['tree_mask.shape'][0]}:{config['paged_kv_block_size']}"
        tree_ms = result['tree_time'] * 1000
        varlen_ms = result['varlen_time'] * 1000
        speedup = result['speedup']
        max_diff = result['max_diff']
        
        print(f"{config_str:<18} {tree_ms:<10.3f} {varlen_ms:<12.3f} {speedup:<10.2f}x {max_diff:<12.6f}")
    
    return results


def run_comprehensive_benchmark():
    """Run benchmarks across different configurations."""
    configs = [
        # Small sequences with different spec_len and block sizes
        {'seqlen_q': 128, 'seqlen_k': 128, 'batch_size': 4, 'nheads': 8, 'head_dim': 128, 'spec_len': (1, 2), 'paged_kv_block_size': 16},
        {'seqlen_q': 256, 'seqlen_k': 256, 'batch_size': 4, 'nheads': 8, 'head_dim': 128, 'spec_len': (2, 3), 'paged_kv_block_size': 16},
        
        # Medium sequences with varied spec_len and block sizes
        {'seqlen_q': 512, 'seqlen_k': 512, 'batch_size': 8, 'nheads': 16, 'head_dim': 128, 'spec_len': (1, 2), 'paged_kv_block_size': 256},
        {'seqlen_q': 1024, 'seqlen_k': 1024, 'batch_size': 8, 'nheads': 16, 'head_dim': 128, 'spec_len': (3, 4), 'paged_kv_block_size': 256},
        
        # Large sequences with larger block sizes
        {'seqlen_q': 2048, 'seqlen_k': 2048, 'batch_size': 4, 'nheads': 16, 'head_dim': 128, 'spec_len': (2, 3), 'paged_kv_block_size': 512},
        
        # Different head dimensions with varied block sizes
        {'seqlen_q': 1024, 'seqlen_k': 1024, 'batch_size': 8, 'nheads': 16, 'head_dim': 64, 'spec_len': (1, 2), 'paged_kv_block_size': 256},
        {'seqlen_q': 1024, 'seqlen_k': 1024, 'batch_size': 8, 'nheads': 16, 'head_dim': 256, 'spec_len': (2, 3), 'paged_kv_block_size': 512},
        
        # Different batch sizes with randomization and block sizes
        {'seqlen_q': 1024, 'seqlen_k': 1024, 'batch_size': 2, 'nheads': 16, 'head_dim': 128, 'spec_len': (1, 2), 'random_spec_len': True, 'paged_kv_block_size': 16},
        {'seqlen_q': 1024, 'seqlen_k': 1024, 'batch_size': 16, 'nheads': 16, 'head_dim': 128, 'spec_len': (2, 3), 'random_seq_len': True, 'paged_kv_block_size': 256},
        
        # High spec_len scenarios with different block sizes
        {'seqlen_q': 1024, 'seqlen_k': 1024, 'batch_size': 8, 'nheads': 16, 'head_dim': 128, 'spec_len': (4, 5), 'paged_kv_block_size': 256},
        {'seqlen_q': 1024, 'seqlen_k': 1024, 'batch_size': 8, 'nheads': 16, 'head_dim': 128, 'spec_len': (6, 8), 'paged_kv_block_size': 512},
        
        # Mixed randomization scenarios with block sizes
        {'seqlen_q': 512, 'seqlen_k': 1024, 'batch_size': 8, 'nheads': 16, 'head_dim': 128, 'spec_len': (2, 3), 'random_seq_len': True, 'random_spec_len': True, 'paged_kv_block_size': 256},
        
        # Block size comparison scenarios
        {'seqlen_q': 1024, 'seqlen_k': 1024, 'batch_size': 8, 'nheads': 16, 'head_dim': 128, 'spec_len': (2, 3), 'paged_kv_block_size': 16},
        {'seqlen_q': 1024, 'seqlen_k': 1024, 'batch_size': 8, 'nheads': 16, 'head_dim': 128, 'spec_len': (2, 3), 'paged_kv_block_size': 256},
        {'seqlen_q': 1024, 'seqlen_k': 1024, 'batch_size': 8, 'nheads': 16, 'head_dim': 128, 'spec_len': (2, 3), 'paged_kv_block_size': 512},
    ]
    
    print("=" * 80)
    print("COMPREHENSIVE TREE ATTENTION BENCHMARK")
    print("=" * 80)
    
    results = []
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Configuration:")
        result = run_tree_attention_benchmark(**config)
        results.append(result)
        print("-" * 80)
    
    # Summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Config':<18} {'Tree(ms)':<10} {'Varlen(ms)':<12} {'Speedup':<10} {'Max Diff':<12}")
    print("-" * 80)
    
    for i, result in enumerate(results):
        config = result['config']
        config_str = f"{config['seqlen_q']}:{config['seqlen_k']}:{config['tree_mask.shape'][0]}:{config['paged_kv_block_size']}"
        tree_ms = result['tree_time'] * 1000
        varlen_ms = result['varlen_time'] * 1000
        speedup = result['speedup']
        max_diff = result['max_diff']
        
        print(f"{config_str:<18} {tree_ms:<10.3f} {varlen_ms:<12.3f} {speedup:<10.2f}x {max_diff:<12.6f}")
    
    return results


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. This benchmark requires GPU.")
        exit(1)
    
    print("Tree Attention vs Flash Attention Varlen Benchmark")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name()}")
    
    # Run single benchmark
    print("\n" + "=" * 80)
    print("SINGLE BENCHMARK (1024x1024, batch=8)")
    print("=" * 80)
    run_tree_attention_benchmark()
    
    # Run decoding benchmark
    run_decoding_benchmark()
    
    # Run comprehensive benchmark
    run_comprehensive_benchmark()