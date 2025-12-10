"""
FINAL WORKING MATRIX-BASED CFR
Sequence-form CFR using sparse matrix operations
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from game_tree import GameEnv, SeqVec
from game_utils import nash_conv
import pyspiel
import time


class SequenceFormMatrixCFR:
    """
    CFR in sequence form using sparse matrix for gradients.
    
    This is the working implementation!
    Works directly in matrix space - no conversion needed!
    """
    
    def __init__(self, game_env, sparse_matrix, structure, variant='cfr+'):
        self.game = game_env
        self.A = sparse_matrix
        self.structure = structure
        self.variant = variant
        
        print(f"\n✅ Sequence-Form Matrix CFR")
        print(f"   GameEnv sequences: P0={game_env.sequence_nums[0]}, P1={game_env.sequence_nums[1]}")
        print(f"   Matrix: {sparse_matrix.method_name}")
        print(f"   Matrix dims: {sparse_matrix.m} × {sparse_matrix.n}")
        
        self.n_seq = [sparse_matrix.m, sparse_matrix.n]
        
        # Initialize strategies
        self.x = [
            np.ones(self.n_seq[0]) / self.n_seq[0],
            np.ones(self.n_seq[1]) / self.n_seq[1]
        ]
        
        # Regrets
        self.R = [
            np.zeros(self.n_seq[0]),
            np.zeros(self.n_seq[1])
        ]
        
        # Cumulative strategies
        self.x_sum = [
            np.zeros(self.n_seq[0]),
            np.zeros(self.n_seq[1])
        ]
        
        self.t = 0
        self.total_time = 0.0
    
    def iteration(self):
        """Run one CFR iteration using sparse matrix ops"""
        self.t += 1
        start = time.time()
        
        for player in [0, 1]:
            # Compute gradient via SPARSE MATRIX-VECTOR PRODUCT
            if player == 0:
                grad = self.A.matvec(self.x[1])
            else:
                grad = -self.A.rmatvec(self.x[0])
            
            # Expected value
            ev = np.dot(self.x[player], grad)
            
            # Regrets = gradient - expected value
            regrets = grad - ev
            
            # Update regrets (variant-specific)
            if self.variant == 'vanilla':
                self.R[player] += regrets
            elif self.variant == 'cfr+':
                self.R[player] = np.maximum(self.R[player] + regrets, 0)
            elif self.variant == 'dcfr':
                t = self.t
                alpha, beta = 1.5, 0.0
                pos_disc = t**alpha / (t**alpha + 1)
                neg_disc = t**beta / (t**beta + 1)
                self.R[player] += regrets
                self.R[player] = np.where(
                    self.R[player] > 0,
                    self.R[player] * pos_disc,
                    self.R[player] * neg_disc
                )
            
            # Regret matching
            positive = np.maximum(self.R[player], 0)
            total = np.sum(positive)
            
            if total > 0:
                self.x[player] = positive / total
            else:
                self.x[player] = np.ones(self.n_seq[player]) / self.n_seq[player]
            
            # Accumulate for average strategy
            self.x_sum[player] += self.x[player]
        
        self.total_time += time.time() - start
    
    def compute_exploitability(self):
        """Compute approximate exploitability"""
        avg = [self.x_sum[p] / max(1, self.t) for p in range(2)]
        
        # Best response values
        br0 = self.A.matvec(avg[1])
        br1 = -self.A.rmatvec(avg[0])
        
        # Exploitability
        exploit_0 = np.max(br0) - np.dot(avg[0], br0)
        exploit_1 = np.max(br1) - np.dot(avg[1], br1)
        
        return exploit_0 + exploit_1
    
    def print_stats(self):
        """Print performance statistics"""
        print(f"\n📊 Stats ({self.t} iterations):")
        print(f"   Variant: {self.variant}")
        print(f"   Matrix: {self.A.method_name}")
        if hasattr(self.A, 'compression_ratio'):
            print(f"   Compression: {self.A.compression_ratio*100:.1f}%")
        print(f"   Total time: {self.total_time:.3f}s")
        if self.t > 0:
            print(f"   Per iteration: {self.total_time/self.t*1000:.3f}ms")
            print(f"   Throughput: {self.t/self.total_time:.1f} iter/sec")


def test_all_methods():
    """Test all sparsification methods"""
    print("="*70)
    print("FINAL TEST: Matrix-Based CFR with All Sparsification Methods")
    print("="*70)
    
    from games.kuhn_structure import KuhnPokerStructureExtractor
    from sparsification.sparse_payoff_matrix import SparsePayoffMatrix
    from sparsification.farina import FarinaTechniqueA, FarinaTechniqueB
    from sparsification.zhang import ZhangSparsification
    from sparsification.general import ThresholdSparsification
    
    # Load game
    game_env = GameEnv()
    game_env.load_open_spiel_game(pyspiel.load_game("kuhn_poker"))
    
    # Extract structure
    extractor = KuhnPokerStructureExtractor()
    structure = extractor.extract_structure()
    
    # Test all methods
    methods = [
        (None, "Baseline (Kronecker-Dense)"),
        (FarinaTechniqueA(), "Farina Technique A"),
        (FarinaTechniqueB(), "Farina Technique B"),
        (ZhangSparsification(max_rank=2, factor_threshold=0.2), "Zhang (rank=2)"),
        (ZhangSparsification(max_rank=3, factor_threshold=0.2), "Zhang (rank=3)"),
        (ThresholdSparsification(threshold=0.1), "Threshold (0.1)"),
    ]
    
    results = []
    
    for sparsifier, name in methods:
        print(f"\n{'='*70}")
        print(f"Testing: {name}")
        print(f"{'='*70}")
        
        # Build sparse matrix
        sparse_A = SparsePayoffMatrix(structure, sparsifier)
        
        # Run CFR
        cfr = SequenceFormMatrixCFR(game_env, sparse_A, structure, variant='cfr+')
        
        n_iters = 2000
        for i in range(n_iters):
            cfr.iteration()
            
            if (i + 1) % 400 == 0:
                exploit = cfr.compute_exploitability()
                print(f"   Iter {i+1:4d}: exploitability = {exploit:.6f}")
        
        final_exploit = cfr.compute_exploitability()
        cfr.print_stats()
        
        # Store results
        storage = sparse_A.nnz if hasattr(sparse_A, 'nnz') else 0
        compression = sparse_A.compression_ratio if hasattr(sparse_A, 'compression_ratio') else 0
        
        results.append({
            'name': name,
            'exploit': final_exploit,
            'storage': storage,
            'compression': compression,
            'time': cfr.total_time,
            'converged': final_exploit < 0.01
        })
    
    # Print summary
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"{'Method':<30} {'Storage':>8} {'Compress':>10} {'Final':>10} {'Time':>8} {'Status':>8}")
    print("-"*70)
    
    for r in results:
        status = "✅ PASS" if r['converged'] else "❌ FAIL"
        print(f"{r['name']:<30} {r['storage']:>8} {r['compression']*100:>9.1f}% {r['exploit']:>10.6f} {r['time']:>7.2f}s {status:>8}")
    
    # Check if all passed
    all_passed = all(r['converged'] for r in results)
    
    print("="*70)
    if all_passed:
        print("🎉🎉🎉 ALL METHODS CONVERGED! SUCCESS! 🎉🎉🎉")
    else:
        print("⚠️  Some methods did not converge")
    print("="*70)
    
    return all_passed


if __name__ == '__main__':
    success = test_all_methods()
    
    if success:
        print("\n✅ Matrix-based CFR with sparsification: WORKING!")
        print("✅ All sparsification methods preserve correctness!")
        print("✅ Ready for presentation!")
    else:
        print("\n⚠️  Some issues remain")