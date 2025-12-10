"""
General sparsification methods (not poker-specific)
"""
import numpy as np
from scipy.sparse import csr_matrix
from typing import Tuple


class ThresholdSparsification:
    """Simple threshold: zero small entries"""
    
    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.name = f"Threshold-{threshold:.2f}"
    
    def sparsify(self, A: np.ndarray) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
        max_val = np.max(np.abs(A))
        A_thresh = A.copy()
        A_thresh[np.abs(A_thresh) < self.threshold * max_val] = 0
        
        A_sparse = csr_matrix(A_thresh)
        U = np.zeros((A.shape[0], 0))
        V = np.zeros((A.shape[1], 0))
        
        return A_sparse, U, V


class SVDSparsification:
    """Low-rank SVD approximation"""
    
    def __init__(self, rank=5):
        self.rank = rank
        self.name = f"SVD-rank{rank}"
    
    def sparsify(self, A: np.ndarray) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
        u, s, vt = np.linalg.svd(A, full_matrices=False)
        
        # Keep top-k
        k = min(self.rank, len(s))
        U = u[:, :k] @ np.diag(np.sqrt(s[:k]))
        V = vt[:k, :].T @ np.diag(np.sqrt(s[:k]))
        
        # Residual
        A_approx = U @ V.T
        residual = A - A_approx
        A_sparse = csr_matrix(residual)
        
        return A_sparse, U, V


class RandomSparsification:
    """Random sparsification (baseline control)"""
    
    def __init__(self, keep_ratio=0.5, seed=42):
        self.keep_ratio = keep_ratio
        self.seed = seed
        self.name = f"Random-{keep_ratio:.0%}"
    
    def sparsify(self, A: np.ndarray) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
        np.random.seed(self.seed)
        
        mask = np.random.rand(*A.shape) < self.keep_ratio
        A_sparse = csr_matrix(A * mask)
        
        U = np.zeros((A.shape[0], 0))
        V = np.zeros((A.shape[1], 0))
        
        return A_sparse, U, V


class TopKSparsification:
    """Keep only top-k entries by magnitude"""
    
    def __init__(self, k_ratio=0.3):
        self.k_ratio = k_ratio
        self.name = f"TopK-{k_ratio:.0%}"
    
    def sparsify(self, A: np.ndarray) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
        flat = np.abs(A.ravel())
        k = int(self.k_ratio * len(flat))
        threshold = np.partition(flat, -k)[-k]
        
        A_thresh = A.copy()
        A_thresh[np.abs(A_thresh) < threshold] = 0
        
        A_sparse = csr_matrix(A_thresh)
        U = np.zeros((A.shape[0], 0))
        V = np.zeros((A.shape[1], 0))
        
        return A_sparse, U, V


class HybridSVDThreshold:
    """SVD + threshold on residual"""
    
    def __init__(self, rank=3, threshold=0.01):  # Lower threshold!
        self.rank = rank
        self.threshold = threshold
        self.name = f"Hybrid-r{rank}-t{threshold:.2f}"
    
    def sparsify(self, A: np.ndarray) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
        # SVD approximation
        u, s, vt = np.linalg.svd(A, full_matrices=False)
        k = min(self.rank, len(s))
        U = u[:, :k] @ np.diag(np.sqrt(s[:k]))
        V = vt[:k, :].T @ np.diag(np.sqrt(s[:k]))
        
        # Threshold residual
        residual = A - U @ V.T
        max_val = np.max(np.abs(residual))
        residual[np.abs(residual) < self.threshold * max_val] = 0
        
        A_sparse = csr_matrix(residual)
        
        return A_sparse, U, V