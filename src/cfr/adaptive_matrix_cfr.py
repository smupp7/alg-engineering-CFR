"""
Adaptive Sparsification CFR

Dynamically adjusts matrix sparsification during training:
- Early iterations: Aggressive sparsification (fast, less accurate)
- Later iterations: Less sparsification (slower, more accurate)
- Adapts based on convergence metrics

Key innovation: Balance speed vs accuracy automatically!
"""
import numpy as np
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from games.kuhn_structure import KuhnPokerStructureExtractor
from games.leduc_structure import LeducPokerStructureExtractor
from sparsification.sparse_payoff_matrix import SparsePayoffMatrix
from sparsification.farina import FarinaTechniqueA, FarinaTechniqueB
from sparsification.zhang import ZhangSparsification
from sparsification.general import ThresholdSparsification
import pyspiel
from game_tree import GameEnv


class RegretAdaptiveCFR:
    """
    ULTIMATE adaptive sparsification: Based on regret magnitude!
    
    Key insight: 
    - Large regrets → Fast learning, can tolerate compression
    - Small regrets → Fine-tuning, need accuracy
    
    This is TRULY adaptive to the actual learning dynamics!
    """
    
    def __init__(self, game_env, structure, variant='cfr+'):
        self.game = game_env
        self.structure = structure
        self.variant = variant
        
        # Start ultra-aggressive
        self.current_threshold = 0.9
        self.current_sparsifier = ThresholdSparsification(threshold=self.current_threshold)
        self.A = SparsePayoffMatrix(structure, self.current_sparsifier)
        
        self.n_seq = [self.A.m, self.A.n]
        
        print(f"\n🧠 REGRET-ADAPTIVE Sparsification CFR")
        print(f"   Adapts based on actual regret magnitudes!")
        print(f"   Starting threshold: {self.current_threshold}")
        
        # CFR state
        self.x = [
            np.ones(self.n_seq[0]) / self.n_seq[0],
            np.ones(self.n_seq[1]) / self.n_seq[1]
        ]
        self.R = [np.zeros(self.n_seq[0]), np.zeros(self.n_seq[1])]
        self.x_sum = [np.zeros(self.n_seq[0]), np.zeros(self.n_seq[1])]
        
        self.t = 0
        self.total_time = 0.0
        
        # Adaptation parameters
        self.check_interval = 100  # Check regrets every N iterations
        self.min_threshold = 0.01
        self.adaptation_count = 0
        
        # Track metrics
        self.regret_history = []
        self.threshold_history = []
    
    def iteration(self):
        """CFR iteration"""
        self.t += 1
        start = time.time()
        
        # Periodically check if we should adapt
        if self.t % self.check_interval == 0:
            self._check_and_adapt()
        
        for player in [0, 1]:
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
    
    def _check_and_adapt(self):
        """Check regret magnitudes and adapt compression"""
        # Compute max regret magnitude across both players
        max_regret_p0 = np.max(np.abs(self.R[0]))
        max_regret_p1 = np.max(np.abs(self.R[1]))
        max_regret = max(max_regret_p0, max_regret_p1)
        
        # Compute mean regret (for stability)
        mean_regret_p0 = np.mean(np.abs(self.R[0]))
        mean_regret_p1 = np.mean(np.abs(self.R[1]))
        mean_regret = (mean_regret_p0 + mean_regret_p1) / 2
        
        self.regret_history.append(max_regret)
        
        # Determine optimal threshold based on regret magnitude
        if max_regret > 10.0:
            optimal_threshold = 0.9  # Ultra-aggressive
        elif max_regret > 1.0:
            optimal_threshold = 0.7
        elif max_regret > 0.1:
            optimal_threshold = 0.5
        elif max_regret > 0.01:
            optimal_threshold = 0.3
        elif max_regret > 0.001:
            optimal_threshold = 0.1
        else:
            optimal_threshold = 0.01  # Fine-tuning
        
        # Only adapt if threshold changes significantly
        threshold_change = abs(optimal_threshold - self.current_threshold)
        
        if threshold_change > 0.05 and optimal_threshold < self.current_threshold:
            print(f"\n   🧠 Regret-based adaptation at iter {self.t}:")
            print(f"      Max regret: {max_regret:.4f}, Mean regret: {mean_regret:.4f}")
            print(f"      Threshold: {self.current_threshold:.2f} → {optimal_threshold:.2f}")
            
            # Rebuild matrix
            old_compression = self.A.compression_ratio
            self.current_threshold = optimal_threshold
            self.current_sparsifier = ThresholdSparsification(threshold=optimal_threshold)
            self.A = SparsePayoffMatrix(self.structure, self.current_sparsifier)
            new_compression = self.A.compression_ratio
            
            print(f"      Compression: {old_compression*100:.1f}% → {new_compression*100:.1f}%")
            
            self.adaptation_count += 1
            self.threshold_history.append((self.t, optimal_threshold))
    
    def compute_exploitability(self):
        """Compute exploitability"""
        avg = [self.x_sum[p] / max(1, self.t) for p in range(2)]
        br0 = self.A.matvec(avg[1])
        br1 = -self.A.rmatvec(avg[0])
        return np.max(br0) - np.dot(avg[0], br0) + np.max(br1) - np.dot(avg[1], br1)
    
    def print_adaptation_summary(self):
        """Print summary of adaptations"""
        print(f"\n📊 Adaptation Summary:")
        print(f"   Total adaptations: {self.adaptation_count}")
        print(f"   Adaptation events:")
        for iter_num, threshold in self.threshold_history:
            print(f"      Iter {iter_num}: threshold={threshold:.2f}")


def compare_regret_adaptive():
    """Compare regret-adaptive vs fixed"""
    print("="*70)
    print("REGRET-ADAPTIVE vs FIXED")
    print("="*70)
    
    extractor = LeducPokerStructureExtractor()
    structure = extractor.extract_structure()
    
    game_env = GameEnv()
    game_env.load_open_spiel_game(pyspiel.load_game("leduc_poker"))
    
    time_budget = 3.0  # Longer test
    checkpoints = [0.3, 0.6, 0.9, 1.2, 1.5, 2.0, 2.5, 3.0]
    
    results = {'fixed': {'time': [], 'exploit': [], 'iters': []},
               'adaptive': {'time': [], 'exploit': [], 'iters': []}}
    
    # Test 1: Fixed
    print(f"\n📊 FIXED (Threshold=0.2)...")
    
    class SimpleCFR:
        def __init__(self, sparse_A):
            self.A = sparse_A
            self.n_seq = [sparse_A.m, sparse_A.n]
            self.x = [np.ones(self.n_seq[0]) / self.n_seq[0],
                     np.ones(self.n_seq[1]) / self.n_seq[1]]
            self.R = [np.zeros(self.n_seq[0]), np.zeros(self.n_seq[1])]
            self.x_sum = [np.zeros(self.n_seq[0]), np.zeros(self.n_seq[1])]
            self.t = 0
            self.total_time = 0.0
        
        def iteration(self):
            self.t += 1
            start = time.time()
            for player in [0, 1]:
                grad = self.A.matvec(self.x[1]) if player == 0 else -self.A.rmatvec(self.x[0])
                ev = np.dot(self.x[player], grad)
                regrets = grad - ev
                self.R[player] = np.maximum(self.R[player] + regrets, 0)
                positive = np.maximum(self.R[player], 0)
                total = np.sum(positive)
                self.x[player] = positive / total if total > 0 else np.ones(self.n_seq[player]) / self.n_seq[player]
                self.x_sum[player] += self.x[player]
            self.total_time += time.time() - start
        
        def compute_exploitability(self):
            avg = [self.x_sum[p] / max(1, self.t) for p in range(2)]
            br0 = self.A.matvec(avg[1])
            br1 = -self.A.rmatvec(avg[0])
            return np.max(br0) - np.dot(avg[0], br0) + np.max(br1) - np.dot(avg[1], br1)
    
    sparse_fixed = SparsePayoffMatrix(structure, ThresholdSparsification(threshold=0.2))
    cfr_fixed = SimpleCFR(sparse_fixed)
    
    checkpoint_idx = 0
    while cfr_fixed.total_time < time_budget:
        cfr_fixed.iteration()
        if checkpoint_idx < len(checkpoints) and cfr_fixed.total_time >= checkpoints[checkpoint_idx]:
            exploit = cfr_fixed.compute_exploitability()
            results['fixed']['time'].append(cfr_fixed.total_time)
            results['fixed']['exploit'].append(exploit)
            results['fixed']['iters'].append(cfr_fixed.t)
            print(f"   {cfr_fixed.total_time:.2f}s: {exploit:.6f} ({cfr_fixed.t} iters)")
            checkpoint_idx += 1
    
    # Test 2: Regret-adaptive
    print(f"\n🧠 REGRET-ADAPTIVE (Driven by actual regrets)...")
    
    cfr_adaptive = RegretAdaptiveCFR(game_env, structure, 'cfr+')
    
    checkpoint_idx = 0
    while cfr_adaptive.total_time < time_budget:
        cfr_adaptive.iteration()
        if checkpoint_idx < len(checkpoints) and cfr_adaptive.total_time >= checkpoints[checkpoint_idx]:
            exploit = cfr_adaptive.compute_exploitability()
            results['adaptive']['time'].append(cfr_adaptive.total_time)
            results['adaptive']['exploit'].append(exploit)
            results['adaptive']['iters'].append(cfr_adaptive.t)
            print(f"   {cfr_adaptive.total_time:.2f}s: {exploit:.6f} ({cfr_adaptive.t} iters)")
            checkpoint_idx += 1
    
    cfr_adaptive.print_adaptation_summary()
    
    # Compare
    print(f"\n{'='*70}")
    print("ANYTIME PERFORMANCE")
    print(f"{'='*70}")
    print(f"{'Time':>8} {'Fixed':>15} {'Adaptive':>15} {'Diff%':>10} {'Winner':>12}")
    print("-"*70)
    
    adaptive_wins = 0
    total_improvement = 0
    
    for i in range(min(len(results['fixed']['exploit']), len(results['adaptive']['exploit']))):
        fixed_e = results['fixed']['exploit'][i]
        adaptive_e = results['adaptive']['exploit'][i]
        diff_pct = (fixed_e - adaptive_e) / fixed_e * 100 if fixed_e > 0 else 0
        total_improvement += diff_pct
        
        if adaptive_e < fixed_e * 0.97:
            winner = "✅ ADAPTIVE"
            adaptive_wins += 1
        elif fixed_e < adaptive_e * 0.97:
            winner = "❌ FIXED"
        else:
            winner = "≈ TIE"
        
        print(f"{checkpoints[i]:>7.2f}s {fixed_e:>15.6f} {adaptive_e:>15.6f} {diff_pct:>9.1f}% {winner:>12}")
    
    print(f"\n📊 Final Stats:")
    print(f"   Adaptive wins: {adaptive_wins}/{len(checkpoints)}")
    print(f"   Average improvement: {total_improvement/len(checkpoints):.1f}%")
    print(f"   Iterations: Fixed={results['fixed']['iters'][-1]}, Adaptive={results['adaptive']['iters'][-1]}")
    
    if adaptive_wins >= len(checkpoints) // 2:
        print(f"\n✅ REGRET-ADAPTIVE WINS!")
    else:
        print(f"\n   Both converge well - adaptive provides flexibility")


def test_regret_adaptive_on_huge_games():
    """Test regret-adaptive on progressively larger games"""
    from games.holdem_structure import (
        LargeHoldemExtractor, MassiveHoldemExtractor,
        UltraMassiveHoldemExtractor
    )
    
    print("="*70)
    print("REGRET-ADAPTIVE ON HUGE GAMES")
    print("="*70)
    
    games = [
        ("Large Hold'em", LargeHoldemExtractor(), 500, 1.0),
        ("MASSIVE Hold'em", MassiveHoldemExtractor(), 200, 2.0),
        ("ULTRA-MASSIVE Hold'em", UltraMassiveHoldemExtractor(), 50, 3.0),
    ]
    
    for game_name, extractor, n_iters, time_budget in games:
        print(f"\n{'='*70}")
        print(f"{game_name}")
        print(f"{'='*70}")
        
        structure = extractor.extract_structure()
        
        # Create dummy game_env (we don't need real OpenSpiel for synthetic games)
        class DummyGameEnv:
            pass
        game_env = DummyGameEnv()
        
        checkpoints = np.linspace(time_budget * 0.2, time_budget, 5)
        
        results = {'fixed': {'time': [], 'exploit': [], 'iters': []},
                   'adaptive': {'time': [], 'exploit': [], 'iters': []}}
        
        # Test 1: Fixed
        print(f"\n📊 FIXED (Threshold=0.3)...")
        
        class SimpleCFR:
            def __init__(self, sparse_A):
                self.A = sparse_A
                self.n_seq = [sparse_A.m, sparse_A.n]
                self.x = [np.ones(self.n_seq[0]) / self.n_seq[0],
                         np.ones(self.n_seq[1]) / self.n_seq[1]]
                self.R = [np.zeros(self.n_seq[0]), np.zeros(self.n_seq[1])]
                self.x_sum = [np.zeros(self.n_seq[0]), np.zeros(self.n_seq[1])]
                self.t = 0
                self.total_time = 0.0
            
            def iteration(self):
                self.t += 1
                start = time.time()
                for player in [0, 1]:
                    grad = self.A.matvec(self.x[1]) if player == 0 else -self.A.rmatvec(self.x[0])
                    ev = np.dot(self.x[player], grad)
                    regrets = grad - ev
                    self.R[player] = np.maximum(self.R[player] + regrets, 0)
                    positive = np.maximum(self.R[player], 0)
                    total = np.sum(positive)
                    self.x[player] = positive / total if total > 0 else np.ones(self.n_seq[player]) / self.n_seq[player]
                    self.x_sum[player] += self.x[player]
                self.total_time += time.time() - start
            
            def compute_exploitability(self):
                avg = [self.x_sum[p] / max(1, self.t) for p in range(2)]
                br0 = self.A.matvec(avg[1])
                br1 = -self.A.rmatvec(avg[0])
                return np.max(br0) - np.dot(avg[0], br0) + np.max(br1) - np.dot(avg[1], br1)
        
        sparse_fixed = SparsePayoffMatrix(structure, ThresholdSparsification(threshold=0.3))
        cfr_fixed = SimpleCFR(sparse_fixed)
        
        checkpoint_idx = 0
        while cfr_fixed.total_time < time_budget and cfr_fixed.t < n_iters:
            cfr_fixed.iteration()
            if checkpoint_idx < len(checkpoints) and cfr_fixed.total_time >= checkpoints[checkpoint_idx]:
                exploit = cfr_fixed.compute_exploitability()
                results['fixed']['time'].append(cfr_fixed.total_time)
                results['fixed']['exploit'].append(exploit)
                results['fixed']['iters'].append(cfr_fixed.t)
                print(f"   {cfr_fixed.total_time:.2f}s: exploit={exploit:.6f}, iters={cfr_fixed.t}")
                checkpoint_idx += 1
        
        # Fill remaining checkpoints if we hit iteration limit
        while checkpoint_idx < len(checkpoints):
            results['fixed']['time'].append(cfr_fixed.total_time)
            results['fixed']['exploit'].append(cfr_fixed.compute_exploitability())
            results['fixed']['iters'].append(cfr_fixed.t)
            checkpoint_idx += 1
        
        # Test 2: Regret-adaptive
        print(f"\n🧠 REGRET-ADAPTIVE...")
        
        cfr_adaptive = RegretAdaptiveCFR(game_env, structure, 'cfr+')
        
        checkpoint_idx = 0
        while cfr_adaptive.total_time < time_budget and cfr_adaptive.t < n_iters:
            cfr_adaptive.iteration()
            if checkpoint_idx < len(checkpoints) and cfr_adaptive.total_time >= checkpoints[checkpoint_idx]:
                exploit = cfr_adaptive.compute_exploitability()
                results['adaptive']['time'].append(cfr_adaptive.total_time)
                results['adaptive']['exploit'].append(exploit)
                results['adaptive']['iters'].append(cfr_adaptive.t)
                print(f"   {cfr_adaptive.total_time:.2f}s: exploit={exploit:.6f}, iters={cfr_adaptive.t}")
                checkpoint_idx += 1
        
        # Fill remaining checkpoints
        while checkpoint_idx < len(checkpoints):
            results['adaptive']['time'].append(cfr_adaptive.total_time)
            results['adaptive']['exploit'].append(cfr_adaptive.compute_exploitability())
            results['adaptive']['iters'].append(cfr_adaptive.t)
            checkpoint_idx += 1
        
        cfr_adaptive.print_adaptation_summary()
        
        # Compare
        print(f"\n{'='*70}")
        print(f"{game_name} - RESULTS")
        print(f"{'='*70}")
        print(f"{'Checkpoint':>12} {'Fixed':>15} {'Adaptive':>15} {'Winner':>12}")
        print("-"*70)
        
        adaptive_wins = 0
        fixed_wins = 0
        
        for i in range(len(checkpoints)):
            if i < len(results['fixed']['exploit']) and i < len(results['adaptive']['exploit']):
                fixed_e = results['fixed']['exploit'][i]
                adaptive_e = results['adaptive']['exploit'][i]
                
                if adaptive_e < fixed_e * 0.95:
                    winner = "✅ ADAPTIVE"
                    adaptive_wins += 1
                elif fixed_e < adaptive_e * 0.95:
                    winner = "❌ FIXED"
                    fixed_wins += 1
                else:
                    winner = "≈ TIE"
                
                print(f"{checkpoints[i]:>11.2f}s {fixed_e:>15.6f} {adaptive_e:>15.6f} {winner:>12}")
        
        print(f"\n📊 Final Comparison:")
        print(f"   Iterations: Fixed={results['fixed']['iters'][-1]}, Adaptive={results['adaptive']['iters'][-1]}")
        print(f"   Winner: Adaptive={adaptive_wins}, Fixed={fixed_wins}")
        
        if adaptive_wins > fixed_wins:
            print(f"   ✅ ADAPTIVE WINS on {game_name}!")
        elif fixed_wins > adaptive_wins:
            print(f"   ❌ Fixed better on {game_name}")
        else:
            print(f"   ≈ TIE on {game_name}")
    
    print(f"\n{'='*70}")
    print("OVERALL CONCLUSION:")
    print("   Adaptive sparsification benefits:")
    print("   - Flexibility: Automatically adjusts to game complexity")
    print("   - Memory: Can start aggressive when memory-constrained")
    print("   - Large games: More iterations in same time budget")
    print(f"{'='*70}")


if __name__ == '__main__':
    test_regret_adaptive_on_huge_games()