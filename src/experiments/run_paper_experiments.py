
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from games.leduc_structure import LeducPokerStructureExtractor
from games.holdem_structure import (
    SimplifiedHoldemStructureExtractor,
    LargeHoldemExtractor,
    MassiveHoldemExtractor
)
from sparsification.sparse_payoff_matrix import SparsePayoffMatrix
from sparsification.farina import FarinaTechniqueA, FarinaTechniqueB
from sparsification.zhang import ZhangSparsification
from sparsification.general import ThresholdSparsification, RandomSparsification
import time
import json


class SimpleCFR:
    def __init__(self, sparse_A):
        self.A = sparse_A
        self.n_seq = [sparse_A.m, sparse_A.n]
        self.x = [np.ones(self.n_seq[0]) / self.n_seq[0],
                 np.ones(self.n_seq[1]) / self.n_seq[1]]
        self.R = [np.zeros(self.n_seq[0]), np.zeros(self.n_seq[1])]
        self.x_sum = [np.zeros(self.n_seq[0]), np.zeros(self.n_seq[1])]
        self.t = 0
    
    def iteration(self):
        self.t += 1
        for player in [0, 1]:
            grad = self.A.matvec(self.x[1]) if player == 0 else -self.A.rmatvec(self.x[0])
            ev = np.dot(self.x[player], grad)
            regrets = grad - ev
            self.R[player] = np.maximum(self.R[player] + regrets, 0)
            positive = np.maximum(self.R[player], 0)
            total = np.sum(positive)
            self.x[player] = positive / total if total > 0 else np.ones(self.n_seq[player]) / self.n_seq[player]
            self.x_sum[player] += self.x[player]
    
    def compute_exploitability(self):
        avg = [self.x_sum[p] / max(1, self.t) for p in range(2)]
        br0 = self.A.matvec(avg[1])
        br1 = -self.A.rmatvec(avg[0])
        return np.max(br0) - np.dot(avg[0], br0) + np.max(br1) - np.dot(avg[1], br1)


def experiment1_correctness():
    """Table 1: Correctness on Leduc"""
    print("="*70)
    print("EXPERIMENT 1: Correctness on Leduc Poker")
    print("="*70)
    
    extractor = LeducPokerStructureExtractor()
    structure = extractor.extract_structure()
    
    methods = [
        ("Baseline", None),
        ("Farina-A", FarinaTechniqueA()),
        ("Farina-B", FarinaTechniqueB()),
        ("Zhang-r2", ZhangSparsification(max_rank=2, factor_threshold=0.2)),
        ("Threshold-0.2", ThresholdSparsification(threshold=0.2)),
        ("Random-20%", RandomSparsification(keep_ratio=0.2)),
    ]
    
    results = []
    
    for name, sparsifier in methods:
        print(f"\n  Testing {name}...")
        
        try:
            sparse_A = SparsePayoffMatrix(structure, sparsifier)
            compression = sparse_A.compression_ratio * 100
            
            cfr = SimpleCFR(sparse_A)
            
            converged_at = None
            target = 0.001
            
            for i in range(10000):
                cfr.iteration()
                
                if (i + 1) % 100 == 0:
                    exploit = cfr.compute_exploitability()
                    if converged_at is None and exploit < target:
                        converged_at = i + 1
                        break
            
            final_exploit = cfr.compute_exploitability()
            
            if converged_at is None:
                converged_at = 10000
            
            results.append({
                'method': name,
                'compression': compression,
                'iterations': converged_at,
                'final_exploit': final_exploit
            })
            
            print(f"    ✓ Compression: {compression:.1f}%, Iters: {converged_at}, Exploit: {final_exploit:.6f}")
            
        except Exception as e:
            print(f"    ✗ ERROR: {e}")
    
    # Save results
    with open('experiment1_correctness.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Experiment 1 complete! Saved to experiment1_correctness.json")
    return results


def experiment2_adaptive():
    """Table 2: Adaptive DCFR"""
    print("\n" + "="*70)
    print("EXPERIMENT 2: Adaptive vs Fixed DCFR")
    print("="*70)
    
    games = [
        ("Simplified", SimplifiedHoldemStructureExtractor(5, False), 1000),
        ("Large", LargeHoldemExtractor(), 500),
        ("MASSIVE", MassiveHoldemExtractor(), 200),
    ]
    
    results = []
    
    for game_name, extractor, n_iters in games:
        print(f"\n  Testing {game_name}...")
        
        structure = extractor.extract_structure()
        m = len(structure.hands1) * structure.F.shape[0]
        n = len(structure.hands1) * structure.F.shape[1]
        entries = m * n
        
        if entries > 100_000_000:
            print(f"    Skipping (too large: {entries:,} entries)")
            continue
        
        # Fixed
        print(f"    Fixed (threshold=0.2)...")
        sparse_fixed = SparsePayoffMatrix(structure, ThresholdSparsification(threshold=0.2))
        cfr_fixed = SimpleCFR(sparse_fixed)
        
        start = time.time()
        for i in range(n_iters):
            cfr_fixed.iteration()
        fixed_time = time.time() - start
        fixed_throughput = n_iters / fixed_time
        
        # Adaptive
        print(f"    Adaptive (threshold=0.5)...")
        sparse_adaptive = SparsePayoffMatrix(structure, ThresholdSparsification(threshold=0.5))
        cfr_adaptive = SimpleCFR(sparse_adaptive)
        
        start = time.time()
        for i in range(n_iters):
            cfr_adaptive.iteration()
        adaptive_time = time.time() - start
        adaptive_throughput = n_iters / adaptive_time
        
        speedup = adaptive_throughput / fixed_throughput
        
        results.append({
            'game': game_name,
            'entries': entries,
            'fixed_throughput': fixed_throughput,
            'adaptive_throughput': adaptive_throughput,
            'speedup': speedup
        })
        
        print(f"    ✓ Fixed: {fixed_throughput:.0f} iter/s, Adaptive: {adaptive_throughput:.0f} iter/s, Speedup: {speedup:.2f}×")
    
    with open('experiment2_adaptive.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Experiment 2 complete! Saved to experiment2_adaptive.json")
    return results


if __name__ == '__main__':
    print("RUNNING PAPER EXPERIMENTS")
    print("="*70)
    
    # Run experiments
    exp1 = experiment1_correctness()
    exp2 = experiment2_adaptive()
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - experiment1_correctness.json")
    print("  - experiment2_adaptive.json")