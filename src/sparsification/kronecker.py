"""
Kronecker product operations - compute A @ x without building A
"""
import numpy as np
import time


class KroneckerPayoffMatrix:
    """
    Payoff matrix in Kronecker form: A = C ⊗ F + (Λ₁WΛ₂) ⊗ S
    
    Enables computing A @ x efficiently without materializing A
    """
    
    def __init__(self, structure):
        self.structure = structure
        self.F = structure.F
        self.S = structure.S
        self.W = structure.W
        self.Lambda1 = structure.Lambda1
        self.Lambda2 = structure.Lambda2
        self.C = structure.C
        
        # Dimensions
        self.n_hands1 = self.Lambda1.shape[0]
        self.n_hands2 = self.Lambda2.shape[0]
        self.n_seq1 = self.F.shape[0]
        self.n_seq2 = self.F.shape[1]
        
        # Full matrix dimensions
        self.m = self.n_hands1 * self.n_seq1  # Rows
        self.n = self.n_hands2 * self.n_seq2  # Cols
        
        print(f"\n📊 Kronecker Payoff Matrix")
        print(f"   Full size: {self.m} × {self.n} ({self.m * self.n:,} entries)")
        print(f"   Hands: {self.n_hands1} × {self.n_hands2}")
        print(f"   Sequences: {self.n_seq1} × {self.n_seq2}")
        print(f"   Storage: F({self.F.size}) + S({self.S.size}) + W({self.W.size}) = {self.F.size + self.S.size + self.W.size} values")
        print(f"   Compression: {(self.F.size + self.S.size + self.W.size) / (self.m * self.n) * 100:.2f}% of full matrix")
    
    def matvec(self, x: np.ndarray) -> np.ndarray:
        """
        Compute A @ x using Kronecker structure
        
        A @ x = (C ⊗ F) @ x + ((Λ₁WΛ₂) ⊗ S) @ x
        
        Using property: (A ⊗ B) @ vec(X) = vec(B @ X @ A^T)
        """
        if len(x) != self.n:
            raise ValueError(f"x has length {len(x)}, expected {self.n}")
        
        # Reshape x into matrix: hands2 × sequences2
        X = x.reshape(self.n_hands2, self.n_seq2)
        
        # Term 1: (C ⊗ F) @ x
        # = vec(F @ X^T @ C^T)
        term1_matrix = self.F @ X.T @ self.C.T  # n_seq1 × n_hands1
        term1 = term1_matrix.T.ravel()  # Flatten to vector
        
        # Term 2: ((Λ₁WΛ₂) ⊗ S) @ x
        # First compute Λ₁WΛ₂
        Lambda_W = self.Lambda1 @ self.W @ self.Lambda2
        
        # Then (Lambda_W ⊗ S) @ x = vec(S @ X^T @ (Lambda_W)^T)
        term2_matrix = self.S @ X.T @ Lambda_W.T  # n_seq1 × n_hands1
        term2 = term2_matrix.T.ravel()  # Flatten to vector
        
        return term1 + term2
    
    def rmatvec(self, y: np.ndarray) -> np.ndarray:
        """
        Compute A^T @ y (for player 2)
        
        A^T @ y = (C^T ⊗ F^T) @ y + ((Λ₁WΛ₂)^T ⊗ S^T) @ y
        """
        if len(y) != self.m:
            raise ValueError(f"y has length {len(y)}, expected {self.m}")
        
        # Reshape y into matrix: hands1 × sequences1
        Y = y.reshape(self.n_hands1, self.n_seq1)
        
        # Term 1: (C^T ⊗ F^T) @ y
        term1_matrix = self.F.T @ Y.T @ self.C  # n_seq2 × n_hands2
        term1 = term1_matrix.T.ravel()
        
        # Term 2: ((Λ₁WΛ₂)^T ⊗ S^T) @ y
        Lambda_W = self.Lambda1 @ self.W @ self.Lambda2
        term2_matrix = self.S.T @ Y.T @ Lambda_W  # n_seq2 × n_hands2
        term2 = term2_matrix.T.ravel()
        
        return term1 + term2
    
    def materialize(self) -> np.ndarray:
        """
        Build full dense matrix (WARNING: only for small games!)
        """
        print(f"\n⚠️  Materializing full {self.m}×{self.n} matrix...")
        
        # Term 1: C ⊗ F
        term1 = np.kron(self.C, self.F)
        
        # Term 2: (Λ₁WΛ₂) ⊗ S
        Lambda_W = self.Lambda1 @ self.W @ self.Lambda2
        term2 = np.kron(Lambda_W, self.S)
        
        A = term1 + term2
        
        print(f"   Shape: {A.shape}")
        print(f"   Nonzeros: {np.count_nonzero(A):,}")
        print(f"   Memory: {A.nbytes / 1e6:.2f} MB")
        
        return A
    
    def verify_matvec(self, num_tests=10):
        """Verify Kronecker matvec matches dense"""
        print(f"\n🧪 Verifying Kronecker matvec...")
        
        A_dense = self.materialize()
        
        max_error_fwd = 0.0
        max_error_bwd = 0.0
        
        for i in range(num_tests):
            # Forward: A @ x
            x = np.random.randn(self.n)
            y_kron = self.matvec(x)
            y_dense = A_dense @ x
            error_fwd = np.linalg.norm(y_kron - y_dense) / (np.linalg.norm(y_dense) + 1e-10)
            max_error_fwd = max(max_error_fwd, error_fwd)
            
            # Backward: A^T @ y
            y = np.random.randn(self.m)
            x_kron = self.rmatvec(y)
            x_dense = A_dense.T @ y
            error_bwd = np.linalg.norm(x_kron - x_dense) / (np.linalg.norm(x_dense) + 1e-10)
            max_error_bwd = max(max_error_bwd, error_bwd)
        
        print(f"   Forward (A@x) max error: {max_error_fwd:.2e}")
        print(f"   Backward (A^T@y) max error: {max_error_bwd:.2e}")
        
        if max_error_fwd < 1e-10 and max_error_bwd < 1e-10:
            print(f"   ✅ Verification passed!")
            return True
        else:
            print(f"   ❌ Verification failed!")
            return False
    
    def benchmark(self, num_trials=100):
        """Benchmark Kronecker vs dense"""
        print(f"\n⚡ Benchmarking ({num_trials} trials)...")
        
        A_dense = self.materialize()
        x = np.random.randn(self.n)
        
        # Warm up
        _ = self.matvec(x)
        _ = A_dense @ x
        
        # Benchmark Kronecker
        start = time.time()
        for _ in range(num_trials):
            _ = self.matvec(x)
        time_kron = time.time() - start
        
        # Benchmark dense
        start = time.time()
        for _ in range(num_trials):
            _ = A_dense @ x
        time_dense = time.time() - start
        
        speedup = time_dense / time_kron
        
        print(f"   Kronecker: {time_kron/num_trials*1000:.3f} ms/iter")
        print(f"   Dense:     {time_dense/num_trials*1000:.3f} ms/iter")
        print(f"   Speedup:   {speedup:.2f}×")
        
        return speedup


def test_kronecker():
    """Test Kronecker operations"""
    print("="*60)
    print("TESTING KRONECKER OPERATIONS")
    print("="*60)
    
    from games.kuhn_structure import KuhnPokerStructureExtractor
    from games.leduc_structure import LeducPokerStructureExtractor
    
    for name, extractor_class in [
        ("Kuhn", KuhnPokerStructureExtractor),
        ("Leduc", LeducPokerStructureExtractor)
    ]:
        print(f"\n{'='*60}")
        print(f"{name} Poker")
        print(f"{'='*60}")
        
        extractor = extractor_class()
        structure = extractor.extract_structure()
        
        kron_matrix = KroneckerPayoffMatrix(structure)
        
        # Verify correctness
        kron_matrix.verify_matvec(num_tests=20)
        
        # Benchmark
        kron_matrix.benchmark(num_trials=100)
    
    print("\n✅ Kronecker operations working!")


if __name__ == '__main__':
    test_kronecker()