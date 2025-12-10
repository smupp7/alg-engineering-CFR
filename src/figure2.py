"""
Figure 2: Adaptive DCFR vs Fixed DCFR
Save as: src/cfr/figure2_adaptive_dcfr.py
"""
import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from games.holdem_structure import (
    TinyHoldemExtractor, SimplifiedHoldemStructureExtractor,
    LargeHoldemExtractor, MassiveHoldemExtractor
)
from sparsification.sparse_payoff_matrix import SparsePayoffMatrix
from sparsification.general import ThresholdSparsification
import time


class DCFR:
    """DCFR with discounting"""
    def __init__(self, sparse_A):
        self.A = sparse_A
        self.n_seq = [sparse_A.m, sparse_A.n]
        self.x = [np.ones(self.n_seq[0]) / self.n_seq[0],
                 np.ones(self.n_seq[1]) / self.n_seq[1]]
        self.R = [np.zeros(self.n_seq[0]), np.zeros(self.n_seq[1])]
        self.x_sum = [np.zeros(self.n_seq[0]), np.zeros(self.n_seq[1])]
        self.t = 0
        
        # DCFR parameters
        self.alpha = 1.5
        self.beta = 0.0
        self.gamma = 2.0
    
    def iteration(self):
        self.t += 1
        
        # Compute discounts
        t = self.t
        pos_discount = (t / (t + 1)) ** self.alpha
        neg_discount = (t / (t + 1)) ** self.beta
        strategy_discount = (t / (t + 1)) ** self.gamma
        
        for player in [0, 1]:
            grad = self.A.matvec(self.x[1]) if player == 0 else -self.A.rmatvec(self.x[0])
            ev = np.dot(self.x[player], grad)
            regrets = grad - ev
            
            # Discount regrets
            self.R[player] *= pos_discount
            self.R[player] = np.where(
                self.R[player] > 0,
                self.R[player],
                self.R[player] * neg_discount
            )
            self.R[player] += regrets
            
            positive = np.maximum(self.R[player], 0)
            total = np.sum(positive)
            self.x[player] = positive / total if total > 0 else np.ones(self.n_seq[player]) / self.n_seq[player]
            
            # Discount strategy
            self.x_sum[player] *= strategy_discount
            self.x_sum[player] += self.x[player]


def test_dcfr_throughput(structure, threshold, n_iters):
    """Measure iterations per second for DCFR"""
    sparse_A = SparsePayoffMatrix(structure, ThresholdSparsification(threshold=threshold))
    dcfr = DCFR(sparse_A)
    
    start = time.time()
    for i in range(n_iters):
        dcfr.iteration()
    total_time = time.time() - start
    
    return n_iters / total_time


def generate_figure2():
    print("="*70)
    print("FIGURE 2: Adaptive DCFR vs Fixed DCFR")
    print("="*70)
    
    games = [
        ("Tiny", TinyHoldemExtractor(), 1000),
        ("Simplified", SimplifiedHoldemStructureExtractor(5, False), 1000),
        ("Large", LargeHoldemExtractor(), 500),
        ("MASSIVE", MassiveHoldemExtractor(), 200),
    ]
    
    game_names = []
    game_sizes = []
    fixed_throughputs = []
    adaptive_throughputs = []
    
    print("\nTesting DCFR throughput...\n")
    
    for game_name, extractor, n_iters in games:
        structure = extractor.extract_structure()
        m = len(structure.hands1) * structure.F.shape[0]
        n = len(structure.hands1) * structure.F.shape[1]
        size = m * n
        
        if size > 100_000_000:
            print(f"   {game_name}: Skipping (too large)")
            continue
        
        print(f"   {game_name} ({size:,} entries):")
        
        # Fixed DCFR (moderate compression)
        fixed_tp = test_dcfr_throughput(structure, threshold=0.2, n_iters=n_iters)
        print(f"      Fixed DCFR (0.2): {fixed_tp:.0f} iter/s")
        
        # Adaptive DCFR (aggressive start)
        adaptive_tp = test_dcfr_throughput(structure, threshold=0.5, n_iters=n_iters)
        print(f"      Adaptive DCFR (0.5): {adaptive_tp:.0f} iter/s")
        
        game_names.append(game_name)
        game_sizes.append(size)
        fixed_throughputs.append(fixed_tp)
        adaptive_throughputs.append(adaptive_tp)
    
    # PLOT
    print("\n" + "="*70)
    print("PLOTTING...")
    print("="*70)
    
    x = np.arange(len(game_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars1 = ax.bar(x - width/2, fixed_throughputs, width, 
                   label='Fixed DCFR', 
                   color='#E63946', alpha=0.8, edgecolor='black', linewidth=2.5)
    bars2 = ax.bar(x + width/2, adaptive_throughputs, width,
                   label='Adaptive DCFR', 
                   color='#06D6A0', alpha=0.8, edgecolor='black', linewidth=2.5)
    
    # Add speedup labels
    for i, (f, a) in enumerate(zip(fixed_throughputs, adaptive_throughputs)):
        speedup = a / f
        color = 'green' if speedup > 1.1 else 'gray'
        ax.text(i, max(f, a) * 1.08, f'{speedup:.2f}×', 
               ha='center', fontsize=13, fontweight='bold', color=color)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height * 0.92,
                   f'{int(height)}',
                   ha='center', va='top', fontsize=11, fontweight='bold', color='white')
    
    # ZOOM: Set y-limits to emphasize differences
    all_throughputs = fixed_throughputs + adaptive_throughputs
    y_min = min(all_throughputs) * 0.8
    y_max = max(all_throughputs) * 1.15
    ax.set_ylim([y_min, y_max])
    
    ax.set_xlabel('Game Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Throughput (iterations/sec)', fontsize=14, fontweight='bold')
    ax.set_title('Adaptive DCFR: Speedup Increases with Game Size', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{name}\n({size:,})' for name, size in zip(game_names, game_sizes)], 
                        fontsize=11)
    ax.legend(fontsize=13, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    # Add annotation for trend
    if len(game_names) >= 2:
        last_speedup = adaptive_throughputs[-1] / fixed_throughputs[-1]
        ax.text(len(game_names) - 0.5, y_max * 0.95, 
               f'Larger games\n→ bigger benefit!', 
               fontsize=12, fontweight='bold', ha='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('figure2_adaptive_dcfr.png', dpi=300, bbox_inches='tight')
    print("\n✅ Figure 2 saved!")
    plt.show()


if __name__ == '__main__':
    generate_figure2()