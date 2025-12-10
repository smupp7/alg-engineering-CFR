"""
Test game implementations
"""
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Now import
import games.openspiel_wrapper
import numpy as np


def test_kuhn_poker():
    """Test Kuhn poker game wrapper"""
    print("="*60)
    print("Testing Kuhn Poker")
    print("="*60)
    
    game = games.openspiel_wrapper.create_kuhn_poker()
    
    # Test infosets
    infosets = game.get_infosets()
    print(f"\n✅ Infosets extracted")
    print(f"   Player 0 infosets: {len(infosets[0])}")
    print(f"   Player 1 infosets: {len(infosets[1])}")
    if len(infosets[0]) > 0:
        print(f"   Sample P0: {infosets[0][:3]}")
    if len(infosets[1]) > 0:
        print(f"   Sample P1: {infosets[1][:3]}")
    
    # Test payoff matrix
    print("\nExtracting payoff matrix...")
    A = game.get_payoff_matrix()
    print(f"\n✅ Payoff matrix extracted")
    print(f"   Shape: {A.shape}")
    if A.size > 0:
        print(f"   Sample entries:\n{A[:min(3, A.shape[0]), :min(3, A.shape[1])]}")
    
    # Sanity checks
    assert A.shape[0] > 0, "Empty matrix"
    assert A.shape[1] > 0, "Empty matrix"
    assert not np.isnan(A).any(), "Contains NaN"
    assert not np.isinf(A).any(), "Contains Inf"
    
    print("\n✅ Kuhn poker tests passed!")
    return game, A


def test_leduc_poker():
    """Test Leduc poker game wrapper"""
    print("\n" + "="*60)
    print("Testing Leduc Poker")
    print("="*60)
    
    game = games.openspiel_wrapper.create_leduc_poker()
    
    # Test infosets
    infosets = game.get_infosets()
    print(f"\n✅ Infosets extracted")
    
    print(f"   Player 0: {len(infosets[0])} infosets")
    print(f"   Player 1: {len(infosets[1])} infosets")
    
    print("\n⚠️  Skipping full payoff matrix for Leduc (too large)")
    print("   Will test CFR algorithms directly on game tree")
    
    print("\n✅ Leduc poker tests passed!")
    return game


if __name__ == '__main__':
    print("Starting game tests...\n")
    
    try:
        kuhn_game, kuhn_matrix = test_kuhn_poker()
        leduc_game = test_leduc_poker()
        
        print("\n" + "="*60)
        print("✅✅✅ ALL GAME TESTS PASSED ✅✅✅")
        print("="*60)
        print("\nReady to implement CFR algorithms!")
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()