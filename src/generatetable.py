"""
Generate Table 1: Sparsification Method Comparison
Save as: src/cfr/generate_table1.py
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from games.leduc_structure import LeducPokerStructureExtractor
from sparsification.sparse_payoff_matrix import SparsePayoffMatrix
from sparsification.farina import FarinaTechniqueA, FarinaTechniqueB
from sparsification.zhang import ZhangSparsification
from sparsification.general import ThresholdSparsification, RandomSparsification
import time


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


def test_method(structure, sparsifier, method_name, max_iters=10000, target=0.001):
    """Test a single method"""
    
    print(f"   {method_name}...", end=' ', flush=True)
    
    try:
        sparse_A = SparsePayoffMatrix(structure, sparsifier)
    except Exception as e:
        print(f"ERROR: {e}")
        return None
    
    cfr = SimpleCFR(sparse_A)
    
    converged_at = None
    start_time = time.time()
    
    for i in range(max_iters):
        cfr.iteration()
        
        if (i + 1) % 100 == 0:
            exploit = cfr.compute_exploitability()
            if converged_at is None and exploit < target:
                converged_at = i + 1
                break
    
    total_time = time.time() - start_time
    
    if converged_at is None:
        converged_at = max_iters
        final_exploit = cfr.compute_exploitability()
    else:
        final_exploit = cfr.compute_exploitability()
    
    compression = sparse_A.compression_ratio * 100
    
    print(f"✓ {converged_at} iters, {compression:.1f}% compression")
    
    return {
        'method': method_name,
        'compression': compression,
        'iters': converged_at,
        'time': total_time,
        'final_exploit': final_exploit
    }


def generate_table():
    print("="*70)
    print("GENERATING TABLE 1: Sparsification Method Comparison")
    print("="*70)
    
    # Use Leduc
    print("\nExtracting Leduc Poker structure...")
    extractor = LeducPokerStructureExtractor()
    structure = extractor.extract_structure()
    
    # Methods to test
    methods = [
        ("Baseline", None),
        ("Farina-A", FarinaTechniqueA()),
        ("Farina-B", FarinaTechniqueB()),
        ("Zhang-r2", ZhangSparsification(max_rank=2, factor_threshold=0.2)),
        ("Threshold-0.2", ThresholdSparsification(threshold=0.2)),
        ("Random-20%", RandomSparsification(keep_ratio=0.2)),
    ]
    
    print("\nTesting methods (target: exploitability < 0.001)...\n")
    
    results = []
    for method_name, sparsifier in methods:
        result = test_method(structure, sparsifier, method_name)
        if result is not None:
            results.append(result)
    
    # PRINT TABLE
    print("\n" + "="*70)
    print("TABLE 1: Sparsification Method Comparison (Leduc Poker)")
    print("="*70)
    print(f"{'Method':<20} {'Compression':>15} {'Iterations':>12} {'Time (s)':>10} {'Final Exploit':>15}")
    print("-"*70)
    
    for r in results:
        print(f"{r['method']:<20} {r['compression']:>14.1f}% {r['iters']:>12} {r['time']:>10.3f} {r['final_exploit']:>15.6f}")
    
    print("-"*70)
    
    # Summary statistics
    structured = [r for r in results if 'Random' not in r['method'] and r['method'] != 'Baseline']
    random = [r for r in results if 'Random' in r['method']]
    
    if structured:
        print(f"\nStructured Methods (Farina, Zhang, Threshold):")
        print(f"  Average compression: {np.mean([r['compression'] for r in structured]):.1f}%")
        print(f"  Average iterations:  {np.mean([r['iters'] for r in structured]):.0f}")
        print(f"  Best method:         {min(structured, key=lambda x: x['iters'])['method']}")
    
    if random:
        print(f"\nRandom Baseline:")
        print(f"  Compression: {random[0]['compression']:.1f}%")
        print(f"  Iterations:  {random[0]['iters']}")
        print(f"  Performance: {random[0]['iters'] / np.mean([r['iters'] for r in structured]):.2f}× slower than structured")
    
    # Save to file
    with open('table1_sparsification.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("TABLE 1: Sparsification Method Comparison (Leduc Poker)\n")
        f.write("="*70 + "\n")
        f.write(f"{'Method':<20} {'Compression':>15} {'Iterations':>12} {'Time (s)':>10} {'Final Exploit':>15}\n")
        f.write("-"*70 + "\n")
        
        for r in results:
            f.write(f"{r['method']:<20} {r['compression']:>14.1f}% {r['iters']:>12} {r['time']:>10.3f} {r['final_exploit']:>15.6f}\n")
        
        f.write("-"*70 + "\n")
        
        if structured:
            f.write(f"\nStructured Methods:\n")
            f.write(f"  Average compression: {np.mean([r['compression'] for r in structured]):.1f}%\n")
            f.write(f"  Average iterations:  {np.mean([r['iters'] for r in structured]):.0f}\n")
        
        if random:
            f.write(f"\nRandom Baseline:\n")
            f.write(f"  Performance: {random[0]['iters'] / np.mean([r['iters'] for r in structured]):.2f}× slower\n")
    
    print("\n✅ Table saved to: table1_sparsification.txt")
    
    # LaTeX version
    print("\n" + "="*70)
    print("LaTeX Version:")
    print("="*70)
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("Method & Compression & Iterations & Time (s) & Final Exploit \\\\")
    print("\\midrule")
    for r in results:
        print(f"{r['method']} & {r['compression']:.1f}\\% & {r['iters']} & {r['time']:.3f} & {r['final_exploit']:.6f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Sparsification method comparison on Leduc Poker (convergence to exploitability < 0.001)}")
    print("\\label{tab:sparsification}")
    print("\\end{table}")


if __name__ == '__main__':
    generate_table()