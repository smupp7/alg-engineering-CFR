"""
Figure 1: ALL METHODS including Farina
Save as: src/cfr/figure1_all_methods.py
"""
import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from games.leduc_structure import LeducPokerStructureExtractor  # ← Use Leduc!
from sparsification.sparse_payoff_matrix import SparsePayoffMatrix
from sparsification.farina import FarinaTechniqueA, FarinaTechniqueB  # ← Add Farina!
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


def test_method_full(structure, sparsifier, method_name, n_iters=5000):
    """Track convergence over time"""
    
    print(f"\n   Testing {method_name}...")
    
    try:
        sparse_A = SparsePayoffMatrix(structure, sparsifier)
    except Exception as e:
        print(f"      ✗ ERROR: {e}")
        return None
    
    cfr = SimpleCFR(sparse_A)
    
    # Checkpoints
    checkpoints = list(range(100, n_iters+1, 100))
    iterations = []
    exploits = []
    
    checkpoint_idx = 0
    converged_at = None
    convergence_threshold = 0.001
    
    start_time = time.time()
    
    for i in range(n_iters):
        cfr.iteration()
        
        if checkpoint_idx < len(checkpoints) and (i + 1) == checkpoints[checkpoint_idx]:
            exploit = cfr.compute_exploitability()
            iterations.append(i + 1)
            exploits.append(exploit)
            
            if converged_at is None and exploit < convergence_threshold:
                converged_at = i + 1
            
            checkpoint_idx += 1
            
            if (i + 1) % 1000 == 0:
                print(f"      {i+1}/{n_iters}, exploit={exploit:.6f}")
    
    total_time = time.time() - start_time
    compression = sparse_A.compression_ratio * 100
    final_exploit = exploits[-1]
    
    if converged_at is None:
        converged_at = n_iters
    
    print(f"      ✓ comp={compression:.1f}%, converged={converged_at}, final={final_exploit:.6f}")
    
    return {
        'name': method_name,
        'iterations': iterations,
        'exploits': exploits,
        'compression': compression,
        'converged_at': converged_at,
        'final_exploit': final_exploit,
        'total_time': total_time
    }


def generate_figure1():
    print("="*70)
    print("FIGURE 1: ALL METHODS (Farina + Zhang + Random)")
    print("="*70)
    
    # Use Leduc - has H_incomp!
    extractor = LeducPokerStructureExtractor()
    structure = extractor.extract_structure()
    
    # ALL METHODS!
    methods = [
        ("Farina-A", FarinaTechniqueA(), '#A23B72', 's', '-'),
        ("Farina-B", FarinaTechniqueB(), '#F18F01', '^', '-'),
        ("Zhang-r2", ZhangSparsification(max_rank=2, factor_threshold=0.2), '#2E86AB', 'o', '-'),
        ("Random-20%", RandomSparsification(keep_ratio=0.2), '#E63946', 'X', '--'),
    ]
    
    print("\nRunning 5000 iterations per method...")
    
    results = []
    for method_name, sparsifier, color, marker, linestyle in methods:
        result = test_method_full(structure, sparsifier, method_name)
        if result is not None:
            result['color'] = color
            result['marker'] = marker
            result['linestyle'] = linestyle
            results.append(result)
    
    if len(results) == 0:
        print("\n❌ No results!")
        return
    
    # PLOT
    print("\n" + "="*70)
    print(f"PLOTTING {len(results)} methods...")
    print("="*70)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for result in results:
        linewidth = 2.5 if result['linestyle'] == '--' else 3.5
        
        print(f"   {result['name']}: {len(result['iterations'])} points")
        
        ax.plot(result['iterations'], result['exploits'],
                marker=result['marker'], linestyle=result['linestyle'],
                linewidth=linewidth, markersize=7, markevery=5,
                label=f"{result['name']} ({result['compression']:.0f}%)",
                color=result['color'], markeredgecolor='black', 
                markeredgewidth=1.2, alpha=0.85)
    
    ax.axhline(y=0.001, color='green', linestyle=':', linewidth=2.5, 
               label='Target', zorder=1)
    
    ax.set_xlabel('Iterations', fontsize=15, fontweight='bold')
    ax.set_ylabel('Exploitability', fontsize=15, fontweight='bold')
    ax.set_title('Structured Sparsification (Farina, Zhang) vs Random', 
                  fontsize=17, fontweight='bold', pad=20)
    ax.set_yscale('log')
    
    # Dynamic limits
    all_exploits = [e for r in results for e in r['exploits']]
    y_min = max(min(all_exploits) * 0.3, 0.00001)
    y_max = max(all_exploits) * 3
    ax.set_ylim([y_min, y_max])
    
    ax.set_xlim([0, 5000])
    ax.legend(fontsize=12, loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3, which='both')
    
    # Annotation
    ax.text(2500, y_max * 0.2, 'Farina & Zhang:\nFaster convergence', 
            fontsize=13, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='lightgreen', 
                     alpha=0.8, edgecolor='green', linewidth=2))
    
    plt.tight_layout()
    plt.savefig('figure1_all_methods.png', dpi=300, bbox_inches='tight')
    print("\n✅ Figure 1 saved!")
    plt.show()


if __name__ == '__main__':
    generate_figure1()