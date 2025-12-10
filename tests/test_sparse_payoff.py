"""
Test SparsePayoffMatrix with all sparsification methods
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from games.kuhn_structure import KuhnPokerStructureExtractor
from games.leduc_structure import LeducPokerStructureExtractor
from sparsification.sparse_payoff_matrix import SparsePayoffMatrix
from sparsification import (
    ZhangSparsification,
    FarinaTechniqueA,
    FarinaTechniqueB,
    ThresholdSparsification,
    SVDSparsification,
    HybridSVDThreshold
)


def test_sparse_payoff_matrix():
    """Test SparsePayoffMatrix with all methods"""
    
    print("="*60)
    print("TESTING SPARSE PAYOFF MATRIX")
    print("="*60)
    
    # All sparsification methods to test
    sparsifiers = [
        (None, "Baseline (Dense Kronecker)"),
        (FarinaTechniqueA(), "Farina Technique A"),
        (FarinaTechniqueB(), "Farina Technique B"),
        (ZhangSparsification(max_rank=3, factor_threshold=0.2), "Zhang"),
        (ThresholdSparsification(threshold=0.1), "Threshold"),
        (SVDSparsification(rank=2), "SVD"),
        (HybridSVDThreshold(rank=2, threshold=0.1), "Hybrid SVD+Threshold"),
    ]
    
    # Test on both games
    games = [
        ("Kuhn Poker", KuhnPokerStructureExtractor),
        ("Leduc Poker", LeducPokerStructureExtractor)
    ]
    
    for game_name, extractor_class in games:
        print(f"\n{'='*60}")
        print(f"{game_name}")
        print(f"{'='*60}")
        
        # Extract structure
        extractor = extractor_class()
        structure = extractor.extract_structure()
        
        # Build ground truth (dense Kronecker)
        A_true = np.kron(structure.C, structure.F) + \
                 np.kron(structure.Lambda1 @ structure.W @ structure.Lambda2, structure.S)
        
        print(f"\nGround truth matrix: {A_true.shape}")
        print(f"Nonzeros: {np.count_nonzero(A_true)}")
        print(f"Frobenius norm: {np.linalg.norm(A_true, 'fro'):.6f}")
        
        results = []
        
        for sparsifier, name in sparsifiers:
            print(f"\n{'-'*60}")
            print(f"Testing: {name}")
            print(f"{'-'*60}")
            
            try:
                # Build sparse payoff matrix
                sparse_A = SparsePayoffMatrix(structure, sparsifier)
                sparse_A.print_stats()
                
                # Test matvec correctness
                print(f"\n🧪 Testing matvec correctness...")
                
                max_error_fwd = 0.0
                max_error_bwd = 0.0
                
                num_tests = 10
                for i in range(num_tests):
                    # Forward: A @ x
                    x = np.random.randn(A_true.shape[1])
                    
                    y_true = A_true @ x
                    y_sparse = sparse_A.matvec(x)
                    
                    error_fwd = np.linalg.norm(y_true - y_sparse) / (np.linalg.norm(y_true) + 1e-10)
                    max_error_fwd = max(max_error_fwd, error_fwd)
                    
                    # Backward: A^T @ y
                    y = np.random.randn(A_true.shape[0])
                    
                    x_true = A_true.T @ y
                    x_sparse = sparse_A.rmatvec(y)
                    
                    error_bwd = np.linalg.norm(x_true - x_sparse) / (np.linalg.norm(x_true) + 1e-10)
                    max_error_bwd = max(max_error_bwd, error_bwd)
                
                print(f"   Forward (A@x):  max error = {max_error_fwd:.6e}")
                print(f"   Backward (A^T@y): max error = {max_error_bwd:.6e}")
                
                # Benchmark speed
                print(f"\n⚡ Benchmarking speed...")
                
                x = np.random.randn(A_true.shape[1])
                
                # Warmup
                _ = sparse_A.matvec(x)
                _ = A_true @ x
                
                # Benchmark
                import time
                n_trials = 100
                
                start = time.time()
                for _ in range(n_trials):
                    _ = sparse_A.matvec(x)
                time_sparse = time.time() - start
                
                start = time.time()
                for _ in range(n_trials):
                    _ = A_true @ x
                time_dense = time.time() - start
                
                speedup = time_dense / time_sparse
                
                print(f"   Sparse: {time_sparse/n_trials*1000:.3f} ms/iter")
                print(f"   Dense:  {time_dense/n_trials*1000:.3f} ms/iter")
                print(f"   Speedup: {speedup:.2f}×")
                
                # Determine pass/fail
                if max_error_fwd < 1e-6 and max_error_bwd < 1e-6:
                    status = "✅ PASS"
                elif max_error_fwd < 1e-3 and max_error_bwd < 1e-3:
                    status = "✅ PASS (approximate)"
                else:
                    status = "❌ FAIL"
                
                print(f"\n{status}")
                
                results.append({
                    'method': name,
                    'storage': sparse_A.storage_nnz,
                    'compression': sparse_A.compression_ratio,
                    'error_fwd': max_error_fwd,
                    'error_bwd': max_error_bwd,
                    'speedup': speedup,
                    'status': status
                })
                
            except Exception as e:
                print(f"\n❌ FAILED with exception: {e}")
                import traceback
                traceback.print_exc()
                
                results.append({
                    'method': name,
                    'status': f"❌ ERROR: {str(e)}"
                })
        
        # Summary table
        print(f"\n{'='*60}")
        print(f"SUMMARY: {game_name}")
        print(f"{'='*60}")
        print(f"{'Method':<30} {'Storage':>10} {'Compress':>10} {'Error':>12} {'Status':>20}")
        print("-"*60)
        
        for r in results:
            method = r['method'][:28]
            
            if 'storage' in r:
                storage = f"{r['storage']:,}"
                compress = f"{r['compression']*100:.1f}%"
                error = f"{max(r['error_fwd'], r['error_bwd']):.2e}"
                status = r['status']
            else:
                storage = "-"
                compress = "-"
                error = "-"
                status = r['status']
            
            print(f"{method:<30} {storage:>10} {compress:>10} {error:>12} {status:>20}")
    
    print(f"\n{'='*60}")
    print("✅ SPARSE PAYOFF MATRIX TESTING COMPLETE!")
    print(f"{'='*60}")


if __name__ == '__main__':
    test_sparse_payoff_matrix()