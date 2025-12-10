"""
Matrix CFR with Parallelized Matrix-Vector Operations
Exploits Kronecker structure and factorization for parallelism
"""
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class ParallelKroneckerMatrix:
    """
    Kronecker matrix with parallel matvec operations.
    
    A = (C ⊗ F) + ((Λ₁WΛ₂) ⊗ S)
    
    Computes both terms in parallel!
    """
    
    def __init__(self, structure):
        self.structure = structure
        self.F = structure.F
        self.S = structure.S
        self.W = structure.W
        self.Lambda1 = structure.Lambda1
        self.Lambda2 = structure.Lambda2
        self.C = structure.C
        
        # Precompute Lambda_W for term2
        self.Lambda_W = self.Lambda1 @ self.W @ self.Lambda2
        
        # Dimensions
        self.n_hands1 = self.Lambda1.shape[0]
        self.n_hands2 = self.Lambda2.shape[0]
        self.n_seq1 = self.F.shape[0]
        self.n_seq2 = self.F.shape[1]
        
        self.m = self.n_hands1 * self.n_seq1
        self.n = self.n_hands2 * self.n_seq2
        
        self.method_name = "Parallel-Kronecker"
        self.compression_ratio = (self.F.size + self.S.size + self.W.size) / (self.m * self.n)
        
        print(f"\n🔧 Parallel Kronecker Matrix")
        print(f"   Two terms computed in parallel")
    
    def _matvec_term1(self, x):
        """Compute (C ⊗ F) @ x"""
        X = x.reshape(self.n_hands2, self.n_seq2)
        result = self.F @ X.T @ self.C.T
        return result.T.ravel()
    
    def _matvec_term2(self, x):
        """Compute ((Λ₁WΛ₂) ⊗ S) @ x"""
        X = x.reshape(self.n_hands2, self.n_seq2)
        result = self.S @ X.T @ self.Lambda_W.T
        return result.T.ravel()
    
    def matvec(self, x):
        """Parallel matrix-vector product"""
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(self._matvec_term1, x)
            future2 = executor.submit(self._matvec_term2, x)
            term1 = future1.result()
            term2 = future2.result()
        return term1 + term2
    
    def _rmatvec_term1(self, y):
        """Compute (C^T ⊗ F^T) @ y"""
        Y = y.reshape(self.n_hands1, self.n_seq1)
        result = self.F.T @ Y.T @ self.C
        return result.T.ravel()
    
    def _rmatvec_term2(self, y):
        """Compute ((Λ₁WΛ₂)^T ⊗ S^T) @ y"""
        Y = y.reshape(self.n_hands1, self.n_seq1)
        result = self.S.T @ Y.T @ self.Lambda_W
        return result.T.ravel()
    
    def rmatvec(self, y):
        """Parallel reverse matrix-vector product"""
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(self._rmatvec_term1, y)
            future2 = executor.submit(self._rmatvec_term2, y)
            term1 = future1.result()
            term2 = future2.result()
        return term1 + term2


class ParallelFactorizedMatrix:
    """
    Factorized matrix with parallel factor computation.
    
    A = U @ diag(d₁, d₂, ..., dᵣ) @ V^T
    
    Each factor computed in parallel!
    """
    
    def __init__(self, U, V, factors):
        """
        Args:
            U: Left factors (m × k)
            V: Right factors (n × k)
            factors: list of (u, v, weight) tuples for each rank-1 component
        """
        self.U = U
        self.V = V
        self.factors = factors
        self.m = U.shape[0]
        self.n = V.shape[0]
        self.rank = len(factors)
        
        self.method_name = f"Parallel-Factorized-r{self.rank}"
        self.nnz = sum(len(f[0].nonzero()[0]) + len(f[1].nonzero()[0]) for f in factors)
        self.compression_ratio = self.nnz / (self.m * self.n)
        
        print(f"\n🔧 Parallel Factorized Matrix")
        print(f"   Rank {self.rank} factors computed in parallel")
    
    def _compute_factor(self, factor_idx, x):
        """Compute contribution of one factor: uᵢ (vᵢᵀ @ x)"""
        u, v, weight = self.factors[factor_idx]
        return weight * u * np.dot(v, x)
    
    def matvec(self, x):
        """Parallel factorized matvec"""
        with ThreadPoolExecutor(max_workers=self.rank) as executor:
            futures = [
                executor.submit(self._compute_factor, i, x)
                for i in range(self.rank)
            ]
            results = [f.result() for f in futures]
        return sum(results)
    
    def _compute_factor_transpose(self, factor_idx, y):
        """Compute contribution of one factor: vᵢ (uᵢᵀ @ y)"""
        u, v, weight = self.factors[factor_idx]
        return weight * v * np.dot(u, y)
    
    def rmatvec(self, y):
        """Parallel factorized rmatvec"""
        with ThreadPoolExecutor(max_workers=self.rank) as executor:
            futures = [
                executor.submit(self._compute_factor_transpose, i, y)
                for i in range(self.rank)
            ]
            results = [f.result() for f in futures]
        return sum(results)


class ParallelMatrixCFR:
    """CFR using parallel matrix operations"""
    
    def __init__(self, sparse_matrix, variant='cfr+'):
        self.A = sparse_matrix
        self.variant = variant
        self.n_seq = [sparse_matrix.m, sparse_matrix.n]
        
        self.x = [
            np.ones(self.n_seq[0]) / self.n_seq[0],
            np.ones(self.n_seq[1]) / self.n_seq[1]
        ]
        self.R = [np.zeros(self.n_seq[0]), np.zeros(self.n_seq[1])]
        self.x_sum = [np.zeros(self.n_seq[0]), np.zeros(self.n_seq[1])]
        self.t = 0
        self.total_time = 0.0
    
    def iteration(self):
        self.t += 1
        start = time.time()
        
        for player in [0, 1]:
            # PARALLEL MATVEC HERE!
            if player == 0:
                grad = self.A.matvec(self.x[1])
            else:
                grad = -self.A.rmatvec(self.x[0])
            
            ev = np.dot(self.x[player], grad)
            regrets = grad - ev
            
            if self.variant == 'cfr+':
                self.R[player] = np.maximum(self.R[player] + regrets, 0)
            else:
                self.R[player] += regrets
            
            positive = np.maximum(self.R[player], 0)
            total = np.sum(positive)
            
            if total > 0:
                self.x[player] = positive / total
            else:
                self.x[player] = np.ones(self.n_seq[player]) / self.n_seq[player]
            
            self.x_sum[player] += self.x[player]
        
        self.total_time += time.time() - start
    
    def compute_exploitability(self):
        avg = [self.x_sum[p] / max(1, self.t) for p in range(2)]
        br0 = self.A.matvec(avg[1])
        br1 = -self.A.rmatvec(avg[0])
        return np.max(br0) - np.dot(avg[0], br0) + np.max(br1) - np.dot(avg[1], br1)


def benchmark_parallel_matvec():
    """Benchmark parallel vs serial matvec on Kuhn, Leduc, and Hold'em"""
    import pyspiel
    from game_tree import GameEnv
    from games.kuhn_structure import KuhnPokerStructureExtractor
    from games.leduc_structure import LeducPokerStructureExtractor
    from games.holdem_structure import (
        TinyHoldemExtractor, SimplifiedHoldemStructureExtractor,
        LargeHoldemExtractor, MassiveHoldemExtractor,
        UltraMassiveHoldemExtractor, GiganticHoldemExtractor,
        ColossalHoldemExtractor
    )

    from sparsification.sparse_payoff_matrix import SparsePayoffMatrix
    
    print("="*70)
    print("PARALLEL MATRIX-VECTOR OPERATIONS BENCHMARK")
    print("="*70)
    
    games = [
        ("Kuhn", KuhnPokerStructureExtractor(), 1000),
        ("Leduc", LeducPokerStructureExtractor(), 500),
        ("Large Hold'em", LargeHoldemExtractor(), 100),
        ("MASSIVE Hold'em", MassiveHoldemExtractor(), 50),
        ("ULTRA-MASSIVE Hold'em", UltraMassiveHoldemExtractor(), 20),
        ("GIGANTIC Hold'em", GiganticHoldemExtractor(), 10),
        ("COLOSSAL Hold'em", ColossalHoldemExtractor(), 5),
    ]
    
    results_summary = []
    
    for game_name, extractor, n_iters in games:
        print(f"\n{'='*70}")
        print(f"{game_name}")
        print(f"{'='*70}")
        
        structure = extractor.extract_structure()
        
        print(f"\n📊 Game Size:")
        print(f"   Hands: {len(structure.hands1)}")
        print(f"   P0 sequences: {structure.F.shape[0]}")
        print(f"   P1 sequences: {structure.F.shape[1]}")
        m = len(structure.hands1) * structure.F.shape[0]
        n = len(structure.hands1) * structure.F.shape[1]
        print(f"   Matrix: {m} × {n}")
        print(f"   Total entries: {m * n:,}")
        
        # Serial
        serial_A = SparsePayoffMatrix(structure, None)
        cfr_serial = ParallelMatrixCFR(serial_A, 'cfr+')
        
        print(f"\n📊 Serial Kronecker:")
        for i in range(n_iters):
            cfr_serial.iteration()
            if (i+1) % (n_iters // 5) == 0:
                print(f"   Iter {i+1}: {cfr_serial.compute_exploitability():.6f}")
        
        serial_time = cfr_serial.total_time
        serial_throughput = n_iters / serial_time
        
        print(f"   Total time: {serial_time:.3f}s")
        print(f"   Throughput: {serial_throughput:.0f} iter/sec")
        
        # Parallel
        parallel_A = ParallelKroneckerMatrix(structure)
        cfr_parallel = ParallelMatrixCFR(parallel_A, 'cfr+')
        
        print(f"\n🚀 Parallel Kronecker:")
        for i in range(n_iters):
            cfr_parallel.iteration()
            if (i+1) % (n_iters // 5) == 0:
                print(f"   Iter {i+1}: {cfr_parallel.compute_exploitability():.6f}")
        
        parallel_time = cfr_parallel.total_time
        parallel_throughput = n_iters / parallel_time
        speedup = serial_time / parallel_time
        
        print(f"   Total time: {parallel_time:.3f}s")
        print(f"   Throughput: {parallel_throughput:.0f} iter/sec")
        
        # Summary
        print(f"\n{'='*70}")
        print(f"{game_name} Results:")
        print(f"{'='*70}")
        print(f"   Serial:   {serial_time:.3f}s ({serial_throughput:.0f} iter/sec)")
        print(f"   Parallel: {parallel_time:.3f}s ({parallel_throughput:.0f} iter/sec)")
        print(f"   Speedup:  {speedup:.2f}x")
        
        if speedup > 1.1:
            status = f"✅ Parallel is {speedup:.2f}x FASTER!"
        elif speedup > 0.9:
            status = "≈ Similar performance"
        else:
            status = "⚠️  Serial faster (overhead > computation)"
        
        print(f"   {status}")
        
        # Store results
        results_summary.append({
            'game': game_name,
            'matrix_size': m * n,
            'speedup': speedup,
            'serial_time': serial_time,
            'parallel_time': parallel_time
        })
    
    # Final summary table
    print(f"\n{'='*70}")
    print("OVERALL RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Game':<30} {'Matrix Size':>15} {'Speedup':>10} {'Status':>15}")
    print("-"*70)
    
    for r in results_summary:
        if r['speedup'] > 1.1:
            status = "✅ PARALLEL"
        elif r['speedup'] > 0.9:
            status = "≈ EVEN"
        else:
            status = "❌ SERIAL"
        
        print(f"{r['game']:<30} {r['matrix_size']:>15,} {r['speedup']:>9.2f}x {status:>15}")
    
    print(f"\n{'='*70}")
    print("CONCLUSION:")
    
    # Find crossover point
    parallel_wins = [r for r in results_summary if r['speedup'] > 1.1]
    
    if parallel_wins:
        smallest_win = min(parallel_wins, key=lambda x: x['matrix_size'])
        print(f"   ✅ Parallelization wins at {smallest_win['game']} scale and above")
        print(f"   ✅ Crossover at ~{smallest_win['matrix_size']:,} matrix entries")
    else:
        print(f"   ⚠️  Thread overhead dominates at all tested scales")
        print(f"   📝 GPU or process-level parallelism needed for Python")
    
    print(f"   📊 Largest game tested: {results_summary[-1]['game']}")
    print(f"   📊 Matrix size: {results_summary[-1]['matrix_size']:,} entries")
    print(f"{'='*70}")


if __name__ == '__main__':
    benchmark_parallel_matvec()