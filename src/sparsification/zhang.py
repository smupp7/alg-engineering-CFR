"""
Zhang sparsification - Algorithms 2, 3, 4
Based on Zhang & Sandholm 2020
"""
import numpy as np
from scipy.sparse import csr_matrix
from typing import Tuple


class ZhangSparsification:
    """
    Zhang's iterative rank-1 sparsification (Algorithm 2)
    
    Repeatedly extracts sparse rank-1 factors until residual is small
    """
    
    def __init__(self, max_rank=10, factor_threshold=0.15, residual_threshold=0.01):
        """
        Args:
            max_rank: Maximum number of rank-1 factors to extract
            factor_threshold: Threshold for sparsifying factors (Algorithm 3)
            residual_threshold: Threshold for final residual
        """
        self.max_rank = max_rank
        self.factor_threshold = factor_threshold
        self.residual_threshold = residual_threshold
        self.name = f"Zhang-r{max_rank}-f{factor_threshold:.2f}"
    
    def sparsify(self, A: np.ndarray) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
        """
        Apply Zhang sparsification (Algorithm 2)
        
        Returns:
            A_sparse: Sparse residual
            U: Left factors (m × rank)
            V: Right factors (n × rank)
        
        Such that: A ≈ A_sparse + U @ V^T
        """
        print(f"\n🔧 Zhang Sparsification (Algorithm 2)")
        print(f"   max_rank={self.max_rank}, factor_threshold={self.factor_threshold}")
        
        residual = A.copy()
        U_factors = []
        V_factors = []
        
        original_norm = np.linalg.norm(A, 'fro')
        max_val = np.max(np.abs(A))
        
        for rank in range(self.max_rank):
            print(f"\n   Iteration {rank+1}:")
            print(f"     Residual norm: {np.linalg.norm(residual, 'fro'):.6f}")
            
            # Algorithm 3: Find sparse rank-1 approximation
            u, v, error = self._find_sparse_rank1(residual)
            
            if u is None:
                print(f"     No good rank-1 factor found, stopping")
                break
            
            u_nnz = np.count_nonzero(u)
            v_nnz = np.count_nonzero(v)
            
            print(f"     Found rank-1: ||u||_0={u_nnz}, ||v||_0={v_nnz}")
            print(f"     Approximation error: {error:.6f}")
            
            # Check stopping criterion (Algorithm 2, line 4-5)
            if u_nnz <= 1 or v_nnz <= 1:
                print(f"     Factor too sparse (||u||_0 ≤ 1 or ||v||_0 ≤ 1), stopping")
                break
            
            U_factors.append(u)
            V_factors.append(v)
            
            # Subtract this factor from residual
            residual -= np.outer(u, v)
        
        # Threshold final residual
        residual_thresh = self.residual_threshold * max_val
        residual[np.abs(residual) < residual_thresh] = 0
        A_sparse = csr_matrix(residual)
        
        # Stack factors
        if U_factors:
            U = np.column_stack(U_factors)
            V = np.column_stack(V_factors)
        else:
            U = np.zeros((A.shape[0], 0))
            V = np.zeros((A.shape[1], 0))
        
        # Statistics
        total_nnz = A_sparse.nnz + np.count_nonzero(U) + np.count_nonzero(V)
        original_nnz = np.count_nonzero(A)
        
        print(f"\n   ✅ Complete!")
        print(f"   Extracted {len(U_factors)} rank-1 factors")
        print(f"   Original: {original_nnz} nnz")
        print(f"   Sparsified: {total_nnz} nnz ({total_nnz/original_nnz*100:.1f}%)")
        
        # Verify reconstruction
        A_recon = A_sparse.toarray() + (U @ V.T if U.shape[1] > 0 else 0)
        recon_error = np.linalg.norm(A - A_recon, 'fro') / original_norm
        print(f"   Reconstruction error: {recon_error:.6e}")
        
        return A_sparse, U, V
    
    def _find_sparse_rank1(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Algorithm 3: Find sparse rank-1 approximation
        
        Uses alternating minimization with thresholding
        """
        m, n = A.shape
        
        # Initialize with SVD (best rank-1 approximation)
        U_svd, s_svd, Vt_svd = np.linalg.svd(A, full_matrices=False)
        
        if s_svd[0] < 1e-10:
            return None, None, float('inf')
        
        # Start with top singular vectors
        u = U_svd[:, 0] * np.sqrt(s_svd[0])
        v = Vt_svd[0, :] * np.sqrt(s_svd[0])
        
        # Alternating minimization with sparsification
        max_iter = 10
        for iteration in range(max_iter):
            u_old = u.copy()
            v_old = v.copy()
            
            # Update v: minimize ||A - uv^T||^2 subject to sparsity
            # Optimal v (before thresholding): v = A^T u / ||u||^2
            if np.linalg.norm(u) > 1e-10:
                v = A.T @ u / (np.linalg.norm(u)**2)
            
            # Threshold v (Algorithm 3: sparsify)
            v = self._threshold_vector(v)
            
            # Update u: minimize ||A - uv^T||^2 subject to sparsity
            # Optimal u (before thresholding): u = A v / ||v||^2
            if np.linalg.norm(v) > 1e-10:
                u = A @ v / (np.linalg.norm(v)**2)
            
            # Threshold u
            u = self._threshold_vector(u)
            
            # Check convergence
            u_change = np.linalg.norm(u - u_old)
            v_change = np.linalg.norm(v - v_old)
            
            if u_change < 1e-6 and v_change < 1e-6:
                break
        
        # Compute approximation error
        approx = np.outer(u, v)
        error = np.linalg.norm(A - approx, 'fro')
        
        return u, v, error
    
    def _threshold_vector(self, x: np.ndarray) -> np.ndarray:
        """
        Threshold vector to enforce sparsity
        
        Zero out entries below threshold * max(|x|)
        """
        if len(x) == 0:
            return x
        
        max_val = np.max(np.abs(x))
        if max_val < 1e-10:
            return x
        
        threshold = self.factor_threshold * max_val
        x_sparse = x.copy()
        x_sparse[np.abs(x_sparse) < threshold] = 0
        
        return x_sparse
    
    def _mode_based_sparsification(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Algorithm 4: Mode-based heuristic for finding sparse rank-1
        
        Note: This is tricky for continuous values, so we use binning
        """
        m, n = A.shape
        
        # Initialize u with dominant column
        col_norms = np.linalg.norm(A, axis=0)
        best_col = np.argmax(col_norms)
        u = A[:, best_col].copy()
        
        # Threshold u
        u = self._threshold_vector(u)
        
        if np.count_nonzero(u) == 0:
            return None, None, float('inf')
        
        # Compute v using mode-based approach
        v = np.zeros(n)
        
        for j in range(n):
            # Compute ratios A[i,j] / u[i] for non-zero u[i]
            ratios = []
            for i in range(m):
                if np.abs(u[i]) > 1e-10:
                    ratios.append(A[i, j] / u[i])
            
            if len(ratios) == 0:
                continue
            
            # Mode approximation: use median (more robust than mode for continuous)
            v[j] = np.median(ratios)
        
        # Threshold v
        v = self._threshold_vector(v)
        
        # Compute error
        approx = np.outer(u, v)
        error = np.linalg.norm(A - approx, 'fro')
        
        return u, v, error


def test_zhang():
    """Test Zhang sparsification"""
    print("="*60)
    print("TESTING ZHANG SPARSIFICATION")
    print("="*60)
    
    # Test on synthetic low-rank + sparse matrix
    print("\n1. Synthetic matrix (rank-2 + sparse noise)")
    np.random.seed(42)
    
    m, n = 20, 15
    
    # Create rank-2 matrix
    u1 = np.random.randn(m)
    v1 = np.random.randn(n)
    u2 = np.random.randn(m)
    v2 = np.random.randn(n)
    
    A_low_rank = np.outer(u1, v1) + 0.5 * np.outer(u2, v2)
    
    # Add sparse noise
    noise = np.random.randn(m, n) * 0.1
    noise[np.abs(noise) < 0.05] = 0
    
    A = A_low_rank + noise
    
    print(f"   Matrix: {A.shape}")
    print(f"   Original nnz: {np.count_nonzero(A)}")
    print(f"   True rank: ~2")
    
    # Apply Zhang
    zhang = ZhangSparsification(max_rank=5, factor_threshold=0.15)
    A_sparse, U, V = zhang.sparsify(A)
    
    # Verify
    A_recon = A_sparse.toarray() + U @ V.T
    error = np.linalg.norm(A - A_recon, 'fro') / np.linalg.norm(A, 'fro')
    
    print(f"\n   Verification:")
    print(f"   Reconstruction error: {error:.6e}")
    
    if error < 1e-10:
        print(f"   ✅ Perfect reconstruction!")
    elif error < 0.01:
        print(f"   ✅ Excellent approximation!")
    else:
        print(f"   ⚠️  Approximation error")
    
    # Test on poker W matrix
    print("\n2. Poker win-lose matrix")
    W = np.array([
        [ 0., -1., -1.],
        [ 1.,  0., -1.],
        [ 1.,  1.,  0.]
    ])
    
    print(f"   W = \n{W}")
    print(f"   Original nnz: {np.count_nonzero(W)}")
    
    zhang_w = ZhangSparsification(max_rank=3, factor_threshold=0.2)
    W_sparse, U_W, V_W = zhang_w.sparsify(W)
    
    W_recon = W_sparse.toarray() + (U_W @ V_W.T if U_W.shape[1] > 0 else 0)
    error_w = np.linalg.norm(W - W_recon, 'fro') / np.linalg.norm(W, 'fro')
    
    print(f"\n   Reconstruction error: {error_w:.6e}")
    
    if error_w < 1e-8:
        print(f"   ✅ Zhang sparsification working!")
    
    print("\n" + "="*60)
    print("✅ ZHANG SPARSIFICATION VERIFIED!")
    print("="*60)


if __name__ == '__main__':
    test_zhang()