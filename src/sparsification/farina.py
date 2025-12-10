"""
Farina sparsification - EXACT from paper
"""
import numpy as np
from scipy.sparse import csr_matrix
from typing import Dict
from sparsification.zhang import ZhangSparsification


class FarinaTechniqueA:
    """
    Farina Technique A - EXACT Proposition 2
    
    U := [(Λ₁U_W) ⊗ I  |  μ₁ ⊗ I]
    V := [(Λ₂V_W) ⊗ S^T  |  μ₂ ⊗ F^T]
    
    where I is n_seq × n_seq identity matrix
    """
    
    def __init__(self, zhang_params=None):
        if zhang_params is None:
            zhang_params = {'max_rank': 2, 'factor_threshold': 0.2}
        self.zhang = ZhangSparsification(**zhang_params)
        self.name = "Farina-A"
    
    def sparsify_structure(self, structure):
        """Exact implementation of Proposition 2"""
        
        # Sparsify W using Zhang
        W_sparse, U_W, V_W = self.zhang.sparsify(structure.W)
        
        Lambda1 = structure.Lambda1  # (n_hands, n_hands)
        Lambda2 = structure.Lambda2  # (n_hands, n_hands)
        F = structure.F              # (n_seq1, n_seq2)
        S = structure.S              # (n_seq1, n_seq2)
        H_incomp = structure.H_incomp
        mu1 = structure.mu1          # (n_hands,)
        mu2 = structure.mu2          # (n_hands,)
        
        n_seq1 = F.shape[0]
        n_seq2 = F.shape[1]
        
        I_seq1 = np.eye(n_seq1)  # Identity matrix!
        
        # Build Â = (Λ₁Ŵ Λ₂) ⊗ S - (Λ₁H×Λ₂) ⊗ F
        term1 = np.kron(Lambda1 @ W_sparse.toarray() @ Lambda2, S)
        term2 = np.kron(Lambda1 @ H_incomp @ Lambda2, F)
        A_hat = term1 - term2
        
        m = A_hat.shape[0]  # n_hands1 * n_seq1
        n = A_hat.shape[1]  # n_hands2 * n_seq2
        
        print(f"  Building U and V...")
        print(f"  Target: U=({m}, ?), V=({n}, ?)")
        
        # Build U = [(Λ₁U_W) ⊗ I  |  μ₁ ⊗ I]
        U_blocks = []
        
        # For each column of U_W
        if U_W.shape[1] > 0:
            for r in range(U_W.shape[1]):
                # (Λ₁ @ U_W[:, r]) ⊗ I_seq1
                # Result: (n_hands, 1) ⊗ (n_seq1, n_seq1) = (n_hands*n_seq1, n_seq1)
                u_w_col = Lambda1 @ U_W[:, r]  # (n_hands,)
                U_block = np.kron(u_w_col.reshape(-1, 1), I_seq1)  # (m, n_seq1)
                U_blocks.append(U_block)
                print(f"    U_W block {r}: {U_block.shape}")
        
        # Correction term: μ₁ ⊗ I_seq1
        # (n_hands, 1) ⊗ (n_seq1, n_seq1) = (n_hands*n_seq1, n_seq1) = (m, n_seq1)
        U_corr = np.kron(mu1.reshape(-1, 1), I_seq1)  # (m, n_seq1)
        U_blocks.append(U_corr)
        print(f"    U_corr: {U_corr.shape}")
        
        # Concatenate horizontally
        U = np.hstack(U_blocks)  # (m, total_cols)
        print(f"  Final U: {U.shape}")
        
        # Build V = [(Λ₂V_W) ⊗ S^T  |  μ₂ ⊗ F^T]
        V_blocks = []
        
        # For each column of V_W
        if V_W.shape[1] > 0:
            for r in range(V_W.shape[1]):
                # (Λ₂ @ V_W[:, r]) ⊗ S^T
                # Result: (n_hands, 1) ⊗ (n_seq2, n_seq1) = (n_hands*n_seq2, n_seq1)
                v_w_col = Lambda2 @ V_W[:, r]  # (n_hands,)
                V_block = np.kron(v_w_col.reshape(-1, 1), S.T)  # (n, n_seq1)
                V_blocks.append(V_block)
                print(f"    V_W block {r}: {V_block.shape}")
        
        # Correction term: μ₂ ⊗ F^T
        # (n_hands, 1) ⊗ (n_seq2, n_seq1) = (n_hands*n_seq2, n_seq1) = (n, n_seq1)
        V_corr = np.kron(mu2.reshape(-1, 1), F.T)  # (n, n_seq1)
        V_blocks.append(V_corr)
        print(f"    V_corr: {V_corr.shape}")
        
        # Concatenate horizontally
        V = np.hstack(V_blocks)  # (n, total_cols)
        print(f"  Final V: {V.shape}")
        
        # Verify dimensions
        assert U.shape[0] == m, f"U rows: expected {m}, got {U.shape[0]}"
        assert V.shape[0] == n, f"V rows: expected {n}, got {V.shape[0]}"
        assert U.shape[1] == V.shape[1], f"U and V column mismatch!"
        
        return {
            'A_sparse': csr_matrix(A_hat),
            'U': U,
            'V': V,
            'W_sparse': W_sparse,
            'U_W': U_W,
            'V_W': V_W
        }


class FarinaTechniqueB:
    """Farina Technique B with proper triangular solver"""
    
    def __init__(self):
        self.name = "Farina-B"
    
    def sparsify_structure(self, structure):
        """Exact Proposition 3"""
        
        W = structure.W
        Lambda1 = structure.Lambda1
        Lambda2 = structure.Lambda2
        F = structure.F
        S = structure.S
        H_incomp = structure.H_incomp
        mu1 = structure.mu1
        mu2 = structure.mu2
        
        n_hands = W.shape[0]
        n_seq1 = F.shape[0]
        n_seq2 = F.shape[1]
        
        m = n_hands * n_seq1
        n = n_hands * n_seq2
        
        I_seq1 = np.eye(n_seq1)
        
        # Build D (lower bidiagonal)
        D = np.eye(n_hands)
        for i in range(1, n_hands):
            D[i, i-1] = -1
        
        Y = D @ W
        
        # Â = -(Λ₁H×Λ₂) ⊗ F
        A_hat = -np.kron(Lambda1 @ H_incomp @ Lambda2, F)
        
        # U = [Λ₁ ⊗ I  |  μ₁ ⊗ I]
        U_blocks = []
        for h in range(n_hands):
            lambda_col = Lambda1[:, h]
            U_block = np.kron(lambda_col.reshape(-1, 1), I_seq1)
            U_blocks.append(U_block)
        
        U_corr = np.kron(mu1.reshape(-1, 1), I_seq1)
        U_blocks.append(U_corr)
        U = np.hstack(U_blocks)
        
        # V = [(Λ₂Y^T) ⊗ S^T  |  μ₂ ⊗ F^T]
        V_blocks = []
        Lambda2_Y_T = Lambda2 @ Y.T
        
        for h in range(n_hands):
            col = Lambda2_Y_T[:, h]
            V_block = np.kron(col.reshape(-1, 1), S.T)
            V_blocks.append(V_block)
        
        V_corr = np.kron(mu2.reshape(-1, 1), F.T)
        V_blocks.append(V_corr)
        V = np.hstack(V_blocks)
        
        # Store D for solving system later
        return {
            'A_sparse': csr_matrix(A_hat),
            'U': U,
            'V': V,
            'D': D,
            'Y': Y,
            'n_hands': n_hands,
            'n_seq1': n_seq1,
            'use_solver': True
        }
    
    def solve_M_system(self, D, n_seq1, y):
        """
        Solve M @ z = y where M = [D⊗I; I]
        
        This is the KEY operation for Technique B!
        """
        n_hands = D.shape[0]
        m = n_hands * n_seq1
        
        # Split y
        split_point = n_hands * n_seq1
        y_top = y[:split_point].reshape(n_hands, n_seq1)
        y_bot = y[split_point:]
        
        # Solve (D⊗I) @ z_top = y_top
        # For each column (sequence), solve D @ z_col = y_col
        z_top = np.zeros((n_hands, n_seq1))
        for s in range(n_seq1):
            # Solve D @ z[:,s] = y[:,s]
            z_top[:, s] = np.linalg.solve(D, y_top[:, s])
        
        # z_bot = y_bot (identity system)
        z_bot = y_bot
        
        # Concatenate
        z = np.concatenate([z_top.ravel(), z_bot])
        
        return z


def verify_farina_reconstruction(structure, result, method_name):
    """Verify with proper M solver for Technique B"""
    
    Lambda1 = structure.Lambda1
    Lambda2 = structure.Lambda2
    F = structure.F
    S = structure.S
    W = structure.W
    C = structure.C
    
    # Original
    A_original = np.kron(C, F) + np.kron(Lambda1 @ W @ Lambda2, S)
    
    # Reconstructed
    A_sparse = result['A_sparse'].toarray()
    U = result['U']
    V = result['V']
    
    if result.get('use_solver'):
        # Technique B: need to apply M^{-1}
        # For verification: compute A @ x for random x
        
        x = np.ones(A_original.shape[1])
        
        # Expected: A @ x
        expected = A_original @ x
        
        # Computed: Â @ x + U @ M^{-1} @ V^T @ x
        result_sparse = A_sparse @ x
        
        # Solve M @ z = V^T @ x
        y = V.T @ x
        
        # Create solver
        tech_b = FarinaTechniqueB()
        z = tech_b.solve_M_system(
            result['D'], 
            result['n_seq1'], 
            y
        )
        
        result_computed = result_sparse + U @ z
        
        error = np.linalg.norm(expected - result_computed) / np.linalg.norm(expected)
        
    else:
        # Technique A: just UV^T
        A_reconstructed = A_sparse + U @ V.T
        error = np.linalg.norm(A_original - A_reconstructed, 'fro') / np.linalg.norm(A_original, 'fro')
    
    print(f"\n{'='*60}")
    print(f"VERIFICATION: {method_name}")
    print(f"{'='*60}")
    print(f"Error: {error:.10f}")
    
    if error < 1e-8:
        print(f"✅ {method_name}: PERFECT!")
        return True
    else:
        print(f"❌ {method_name}: Error = {error:.6f}")
        return False