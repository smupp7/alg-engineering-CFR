"""
Figure 3: Parallel Speedup - Only Matters at Scale
Save as: src/cfr/figure3_parallel_crossover.py
"""
import matplotlib.pyplot as plt
import numpy as np


def generate_figure3():
    print("="*70)
    print("FIGURE 3: Parallel Crossover Point")
    print("="*70)
    
    # Your data
    game_names = ['Kuhn', 'Leduc', 'Large', 'MASSIVE', 'ULTRA', 'GIGANTIC', 'COLOSSAL']
    matrix_sizes = [72, 5049, 230400, 50000000, 2500000000, 14400000000, 40000000000]
    speedups = [0.10, 0.11, 0.16, 0.37, 1.20, 1.23, 1.15]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot speedup curve with bigger markers
    colors = ['#E63946' if s < 1.0 else '#06D6A0' for s in speedups]
    
    ax.plot(matrix_sizes, speedups, '-', color='#1982C4', linewidth=5, 
            alpha=0.7, zorder=2)
    
    ax.scatter(matrix_sizes, speedups, s=400, c=colors, 
              edgecolors='black', linewidths=3, alpha=0.9, zorder=3)
    
    # Breakeven line
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=3, 
               label='Breakeven (1.0×)', zorder=2)
    
    # STRONG shading
    ax.axhspan(0, 1.0, alpha=0.25, color='red', zorder=1)
    ax.axhspan(1.0, 1.5, alpha=0.25, color='green', zorder=1)
    
    # BIG annotation for crossover
    crossover_idx = next(i for i, s in enumerate(speedups) if s >= 1.0)
    ax.annotate('Crossover Point:\n~2.5 Billion Entries', 
               xy=(matrix_sizes[crossover_idx], speedups[crossover_idx]),
               xytext=(-150, -80), textcoords='offset points',
               fontsize=16, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.8', facecolor='yellow', alpha=0.95, edgecolor='black', linewidth=3),
               arrowprops=dict(arrowstyle='->', lw=4, color='black'))
    
    # Add region labels
    ax.text(500, 0.4, 'Serial Faster\n(Overhead dominates)', 
           fontsize=14, ha='center', fontweight='bold', 
           bbox=dict(boxstyle='round,pad=0.6', facecolor='pink', alpha=0.7, edgecolor='red', linewidth=2))
    
    ax.text(5e9, 1.25, 'Parallel Faster\n(Computation dominates)', 
           fontsize=14, ha='center', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.6', facecolor='lightgreen', alpha=0.7, edgecolor='green', linewidth=2))
    
    # Add game labels
    for i, (size, name, speedup) in enumerate(zip(matrix_sizes, game_names, speedups)):
        ax.text(size, speedup - 0.1, name, fontsize=10, ha='center', 
                fontweight='bold', alpha=0.8)
    
    # ZOOM y-axis to emphasize the crossover
    ax.set_ylim([0, 1.4])
    
    ax.set_xlabel('Matrix Size (entries)', fontsize=15, fontweight='bold')
    ax.set_ylabel('Parallel Speedup (×)', fontsize=15, fontweight='bold')
    ax.set_title('Parallelization Benefit: Only Emerges at Scale (>2.5B entries)', 
                 fontsize=17, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=13, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('figure3_parallel_crossover.png', dpi=300, bbox_inches='tight')
    print("\n✅ Figure 3 saved!")
    plt.show()


if __name__ == '__main__':
    generate_figure3()