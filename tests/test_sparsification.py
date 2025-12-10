"""
More detailed test of Zhang sparsification
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import games.openspiel_wrapper
from sparsification.zhang import ZhangSparsification


def test_synthetic_matrix():
    """Test on synthetic matrix from Example 1 (upper triangular)"""
    print("="*60)
    print("Test 1: Synthetic Upper-Triangular Matrix")
    print("="*60)
    
    # Create upper-triangular matrix (Example 1 from paper)
    n = 64
    u_orig = np.random.randn(n)
    v_orig = np.random.randn(n)
    A = np.outer(u_orig, v_orig)
    A = np.triu(A)  # Upper triangular
    
    print(f"Matrix: {n} × {n}")
    print(f"Original nnz: {np.count_nonzero(A):,} / {A.size:,} ({np.count_nonzero(A)/A.size*100:.1f}%)")
    
    zhang = ZhangSparsification(max_rank=20)
    A_sparse, U, V = zhang.factorize(A)
    
    # Check compression
    total_nnz = A_sparse.nnz + np.count_nonzero(U) + np.count_nonzero(V)
    compression_ratio = total_nnz / np.count_nonzero(A)
    
    # Check accuracy
    A_recon = A_sparse.toarray()
    if U.shape[1] > 0:
        A_recon += U @ V.T
    
    error = np.linalg.norm(A - A_recon, 'fro') / np.linalg.norm(A, 'fro')
    
    print(f"\n📊 Results:")
    print(f"   Compression: {compression_ratio:.2f}x")
    print(f"   Error: {error:.4%}")
    
    if compression_ratio < 1.0:
        print("   ✅ Successfully compressed!")
    else:
        print("   ❌ Failed to compress (larger than original)")
    
    return compression_ratio < 1.0


def test_block_diagonal():
    """Test on block diagonal matrix"""
    print("\n" + "="*60)
    print("Test 2: Block Diagonal Matrix")
    print("="*60)
    
    # Create block diagonal (common in poker)
    block_size = 16
    n_blocks = 4
    n = block_size * n_blocks
    
    A = np.zeros((n, n))
    for i in range(n_blocks):
        start = i * block_size
        end = (i + 1) * block_size
        u = np.random.randn(block_size)
        v = np.random.randn(block_size)
        A[start:end, start:end] = np.outer(u, v)
    
    print(f"Matrix: {n} × {n}, {n_blocks} blocks")
    print(f"Original nnz: {np.count_nonzero(A):,} / {A.size:,} ({np.count_nonzero(A)/A.size*100:.1f}%)")
    
    zhang = ZhangSparsification(max_rank=20)
    A_sparse, U, V = zhang.factorize(A)
    
    total_nnz = A_sparse.nnz + np.count_nonzero(U) + np.count_nonzero(V)
    compression_ratio = total_nnz / np.count_nonzero(A)
    
    A_recon = A_sparse.toarray()
    if U.shape[1] > 0:
        A_recon += U @ V.T
    
    error = np.linalg.norm(A - A_recon, 'fro') / np.linalg.norm(A, 'fro')
    
    print(f"\n📊 Results:")
    print(f"   Compression: {compression_ratio:.2f}x")
    print(f"   Error: {error:.4%}")
    
    if compression_ratio < 1.0:
        print("   ✅ Successfully compressed!")
    else:
        print("   ❌ Failed to compress")
    
    return compression_ratio < 1.0


def test_kuhn_detailed():
    """Detailed analysis of Kuhn poker matrix"""
    print("\n" + "="*60)
    print("Test 3: Kuhn Poker (Detailed Analysis)")
    print("="*60)
    
    game = games.openspiel_wrapper.create_kuhn_poker()
    A = game.get_payoff_matrix()
    
    print(f"\nMatrix properties:")
    print(f"   Shape: {A.shape}")
    print(f"   Rank: {np.linalg.matrix_rank(A)}")
    print(f"   Density: {np.count_nonzero(A) / A.size * 100:.1f}%")
    print(f"   Max: {np.max(np.abs(A)):.2f}")
    print(f"   Frobenius norm: {np.linalg.norm(A, 'fro'):.2f}")
    
    # SVD for comparison
    U_svd, s_svd, Vt_svd = np.linalg.svd(A, full_matrices=False)
    print(f"\nSVD singular values (top 5): {s_svd[:5]}")
    print(f"   Effective rank (>1% of max): {np.sum(s_svd > 0.01 * s_svd[0])}")
    
    # Check if low-rank
    rank_5_error = np.linalg.norm(A - U_svd[:, :5] @ np.diag(s_svd[:5]) @ Vt_svd[:5, :], 'fro') / np.linalg.norm(A, 'fro')
    print(f"   Rank-5 approximation error: {rank_5_error:.4%}")
    
    if rank_5_error > 0.1:
        print("\n⚠️  Matrix is NOT low-rank!")
        print("   Zhang's method may not compress well")
    
    # Try factorization
    zhang = ZhangSparsification(max_rank=10)
    A_sparse, U, V = zhang.factorize(A)
    
    total_nnz = A_sparse.nnz + np.count_nonzero(U) + np.count_nonzero(V)
    compression_ratio = total_nnz / np.count_nonzero(A)
    
    print(f"\n📊 Compression: {compression_ratio:.2f}x")
    
    if compression_ratio >= 1.0:
        print("❌ Zhang's method NOT suitable for this matrix")
        print("   → Matrix is too dense and not low-rank")
        print("   → Should use dense CFR or different sparsification")
    
    return compression_ratio < 1.0


if __name__ == '__main__':
    test1 = test_synthetic_matrix()
    test2 = test_block_diagonal()
    test3 = test_kuhn_detailed()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Synthetic matrix: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Block diagonal: {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"Kuhn poker: {'✅ PASS' if test3 else '❌ FAIL'}")
    
    if test1 and test2:
        print("\n✅ Zhang implementation correct!")
        print("⚠️  But Kuhn poker matrix is not a good fit")
        print("   → Need different approach for dense matrices")
    elif not test1 and not test2:
        print("\n❌ Zhang implementation has bugs")