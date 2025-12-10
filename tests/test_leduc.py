"""
Quick test that Leduc poker works
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import games.openspiel_wrapper
from cfr.vanilla import VanillaCFR


def test_leduc_basic():
    """Test basic Leduc setup"""
    print("="*60)
    print("Testing Leduc Poker")
    print("="*60)
    
    # Load game
    game = games.openspiel_wrapper.create_leduc_poker()
    
    print(f"\n✅ Loaded {game.game_name}")
    print(f"   Game type: {type(game.game)}")
    
    # Get payoff matrix
    print("\n📊 Extracting payoff matrix...")
    A = game.get_payoff_matrix()
    
    print(f"   Shape: {A.shape[0]} × {A.shape[1]}")
    print(f"   Total entries: {A.size:,}")
    print(f"   Nonzeros: {np.count_nonzero(A):,}")
    print(f"   Density: {np.count_nonzero(A) / A.size * 100:.2f}%")
    print(f"   Rank: {np.linalg.matrix_rank(A)}")
    print(f"   Frobenius norm: {np.linalg.norm(A, 'fro'):.2f}")
    
    # Test CFR on Leduc
    print("\n🎮 Testing CFR on Leduc (100 iterations)...")
    cfr = VanillaCFR(game)
    
    import time
    start = time.time()
    
    for i in range(100):
        cfr.iteration()
        if (i + 1) % 20 == 0:
            exploit = cfr.compute_exploitability()
            print(f"   Iter {i+1}: exploitability = {exploit:.6f}")
    
    elapsed = time.time() - start
    final_exploit = cfr.compute_exploitability()
    
    print(f"\n✅ CFR completed!")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Final exploitability: {final_exploit:.6f}")
    print(f"   Iterations/sec: {100/elapsed:.1f}")
    
    if final_exploit < 1.0:  # Should decrease from initial
        print("\n✅ Leduc poker working correctly!")
    else:
        print("\n⚠️  Exploitability seems high")
    
    return game, A


if __name__ == '__main__':
    game, A = test_leduc_basic()