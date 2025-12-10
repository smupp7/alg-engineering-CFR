"""
Sparsified Kronecker matrix with pluggable sparsification methods
"""
import numpy as np
from scipy.sparse import csr_matrix
import time


class SparsePayoffMatrix:
    """
    Kronecker matrix with sparsification applied
    
    Supports multiple sparsification methods:
    - None (baseline Kronecker)
    - Zhang
    - Farina A
    - Farina B
    - Threshold
    - SVD
    - etc.
    """
    
    def __init__(self, structure, sparsifier=None):
        """
        Args:
            structure: PokerStructure from extraction
            sparsifier: Sparsification method or None
        """
        self.structure = structure
        self.sparsifier = sparsifier
        
        self.F = structure.F
        self.S = structure.S
        self.W = structure.W
        self.Lambda1 = structure.Lambda1
        self.Lambda2 = structure.Lambda2
        self.C = structure.C
        
        n_hands1 = self.Lambda1.shape[0]
        n_hands2 = self.Lambda2.shape[0]
        n_seq1 = self.F.shape[0]
        n_seq2 = self.F.shape[1]
        
        self.m = n_hands1 * n_seq1
        self.n = n_hands2 * n_seq2
        
        # Apply sparsification
        self._apply_sparsification()
        
        # Statistics
        self._compute_stats()
    
    def _apply_sparsification(self):
        """Apply sparsification method"""
        
        if self.sparsifier is None:
            # No sparsification - use dense Kronecker
            self.method_name = "Kronecker-Dense"
            self.sparsified = False
            
            self.A_sparse = None
            self.U = None
            self.V = None
            self.use_solver = False
            
        elif hasattr(self.sparsifier, 'sparsify_structure'):
            # Farina-style (operates on structure)
            self.method_name = self.sparsifier.name
            result = self.sparsifier.sparsify_structure(self.structure)
            
            self.sparsified = True
            self.A_sparse = result['A_sparse']
            self.U = result['U']
            self.V = result['V']
            self.use_solver = result.get('use_solver', False)
            
            if self.use_solver:
                self.D = result['D']
                self.n_hands = result['n_hands']
                self.n_seq1 = result['n_seq1']
            
        elif hasattr(self.sparsifier, 'sparsify'):
            # Zhang-style or general (operates on full matrix)
            self.method_name = self.sparsifier.name
            
            # Build full matrix first (only for small games!)
            A_full = np.kron(self.C, self.F) + np.kron(self.Lambda1 @ self.W @ self.Lambda2, self.S)
            
            # Apply sparsification
            self.A_sparse, self.U, self.V = self.sparsifier.sparsify(A_full)
            
            self.sparsified = True
            self.use_solver = False
        
        else:
            raise ValueError(f"Unknown sparsifier type: {type(self.sparsifier)}")
    
    def _compute_stats(self):
        """Compute storage statistics"""
        
        if not self.sparsified:
            # Dense Kronecker storage
            self.storage_nnz = (
                np.count_nonzero(self.F) + 
                np.count_nonzero(self.S) + 
                np.count_nonzero(self.W)
            )
        else:
            # Sparsified storage
            self.storage_nnz = (
                self.A_sparse.nnz + 
                np.count_nonzero(self.U) + 
                np.count_nonzero(self.V)
            )
        
        # Full matrix would have this many
        self.full_size = self.m * self.n
        self.compression_ratio = self.storage_nnz / self.full_size
    
    def matvec(self, x: np.ndarray) -> np.ndarray:
        """
        Compute A @ x
        
        If sparsified: A @ x = Â @ x + U @ M^{-1} @ V^T @ x
        If not: Use Kronecker product formula
        """
        
        if not self.sparsified:
            # Dense Kronecker matvec
            X = x.reshape(self.Lambda2.shape[0], -1)
            term1_matrix = self.F @ X.T @ self.C.T
            term1 = term1_matrix.T.ravel()
            
            Lambda_W = self.Lambda1 @ self.W @ self.Lambda2
            term2_matrix = self.S @ X.T @ Lambda_W.T
            term2 = term2_matrix.T.ravel()
            
            return term1 + term2
        
        else:
            # Sparsified matvec
            result = self.A_sparse @ x
            
            if self.U.shape[1] > 0:
                y = self.V.T @ x
                
                if self.use_solver:
                    # Technique B: solve M @ z = y
                    z = self._solve_M_system(y)
                else:
                    # Technique A: z = y
                    z = y
                
                result += self.U @ z
            
            return result
    
    def rmatvec(self, y: np.ndarray) -> np.ndarray:
        """Compute A^T @ y"""
        
        if not self.sparsified:
            # Dense Kronecker
            Y = y.reshape(self.Lambda1.shape[0], -1)
            term1_matrix = self.F.T @ Y.T @ self.C
            term1 = term1_matrix.T.ravel()
            
            Lambda_W = self.Lambda1 @ self.W @ self.Lambda2
            term2_matrix = self.S.T @ Y.T @ Lambda_W
            term2 = term2_matrix.T.ravel()
            
            return term1 + term2
        
        else:
            # Sparsified
            result = self.A_sparse.T @ y
            
            if self.U.shape[1] > 0:
                # A^T @ y = Â^T @ y + V @ M^{-T} @ U^T @ y
                z = self.U.T @ y
                
                if self.use_solver:
                    # Solve M^T @ w = z
                    w = self._solve_MT_system(z)
                else:
                    w = z
                
                result += self.V @ w
            
            return result
    
    def _solve_M_system(self, y: np.ndarray) -> np.ndarray:
        """Solve M @ z = y for Technique B"""
        
        n_hands = self.n_hands
        n_seq1 = self.n_seq1
        
        # Split y
        split_point = n_hands * n_seq1
        y_top = y[:split_point].reshape(n_hands, n_seq1)
        y_bot = y[split_point:]
        
        # Solve (D⊗I) @ z_top = y_top
        z_top = np.zeros((n_hands, n_seq1))
        for s in range(n_seq1):
            z_top[:, s] = np.linalg.solve(self.D, y_top[:, s])
        
        z_bot = y_bot
        z = np.concatenate([z_top.ravel(), z_bot])
        
        return z
    
    def _solve_MT_system(self, z: np.ndarray) -> np.ndarray:
        """Solve M^T @ w = z"""
        
        n_hands = self.n_hands
        n_seq1 = self.n_seq1
        
        split_point = n_hands * n_seq1
        z_top = z[:split_point].reshape(n_hands, n_seq1)
        z_bot = z[split_point:]
        
        # Solve (D^T⊗I) @ w_top = z_top
        w_top = np.zeros((n_hands, n_seq1))
        for s in range(n_seq1):
            w_top[:, s] = np.linalg.solve(self.D.T, z_top[:, s])
        
        w_bot = z_bot
        w = np.concatenate([w_top.ravel(), w_bot])
        
        return w
    
    def print_stats(self):
        """Print statistics"""
        print(f"\n📊 Sparsified Kronecker Matrix: {self.method_name}")
        print(f"   Full size: {self.m} × {self.n} = {self.full_size:,} entries")
        print(f"   Storage: {self.storage_nnz:,} nnz")
        print(f"   Compression: {self.compression_ratio*100:.2f}%")
        
        if self.sparsified:
            print(f"   Â: {self.A_sparse.nnz:,} nnz")
            print(f"   U: {self.U.shape}, {np.count_nonzero(self.U):,} nnz")
            print(f"   V: {self.V.shape}, {np.count_nonzero(self.V):,} nnz")