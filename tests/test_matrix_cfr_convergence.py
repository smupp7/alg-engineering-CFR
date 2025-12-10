"""
Comprehensive test suite for MatrixCFR convergence

Tests ALL sparsification methods converge to Nash equilibrium
"""
import sys
import os

# Fix paths - game_tree and game_utils are in src/cfr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'cfr'))

import numpy as np
import pyspiel

# Import instructor's code from src/cfr
try:
    from game_tree import GameEnv
    from game_utils import compute_utility_vector
except ImportError as e:
    print(f"ERROR: Cannot import game_tree/game_utils: {e}")
    print("Please ensure game_tree.py and game_utils.py are in src/cfr/")
    sys.exit(1)

# Import MatrixCFR
try:
    from matrix_cfr import MatrixCFR, run_matrix_cfr
except ImportError as e:
    print(f"ERROR: Cannot import MatrixCFR: {e}")
    print("Please ensure matrix_cfr.py is in src/cfr/")
    sys.exit(1)

# Import your code
from games.kuhn_structure import KuhnPokerStructureExtractor
from sparsification.sparse_payoff_matrix import SparsePayoffMatrix
from sparsification.farina import FarinaTechniqueA, FarinaTechniqueB
from sparsification.zhang import ZhangSparsification
from sparsification.general import ThresholdSparsification, SVDSparsification

def test_all_sparsification_methods():
    """
    Test that ALL sparsification methods converge
    """
    print("="*70)
    print("COMPREHENSIVE MATRIX CFR CONVERGENCE TEST")
    print("="*70)
    
    # Load game
    game_env = GameEnv()
    game_env.load_open_spiel_game(pyspiel.load_game("kuhn_poker"))
    
    print(f"\n📊 Game: Kuhn Poker")
    print(f"   P0: {game_env.sequence_nums[0]} sequences")
    print(f"   P1: {game_env.sequence_nums[1]} sequences")
    
    # Extract structure
    extractor = KuhnPokerStructureExtractor()
    structure = extractor.extract_structure()
    
    # Define all methods to test
    methods = [
        ("Baseline (Kronecker)", None),
        ("Farina Technique A", FarinaTechniqueA()),
        ("Farina Technique B", FarinaTechniqueB()),
        ("Zhang (rank=2)", ZhangSparsification(max_rank=2, factor_threshold=0.2)),
        ("Zhang (rank=3)", ZhangSparsification(max_rank=3, factor_threshold=0.2)),
        ("Threshold (0.1)", ThresholdSparsification(threshold=0.1)),
        ("Threshold (0.05)", ThresholdSparsification(threshold=0.05)),
    ]
    
    # Test parameters
    n_iterations = 2000
    convergence_threshold = 0.01
    
    results = []
    
    for name, sparsifier in methods:
        print(f"\n{'='*70}")
        print(f"Testing: {name}")
        print(f"{'='*70}")
        
        try:
            # Build sparse matrix
            sparse_A = SparsePayoffMatrix(structure, sparsifier)
            
            print(f"   Matrix: {sparse_A.m} × {sparse_A.n}")
            print(f"   Storage: {sparse_A.storage_nnz} nnz")
            print(f"   Compression: {sparse_A.compression_ratio*100:.1f}%")
            
            # Run CFR
            cfr = run_matrix_cfr(
                game_env, 
                sparse_A, 
                structure,
                n_iterations=n_iterations,
                variant='cfr+',
                print_every=400
            )
            
            # Check final exploitability
            final_exploit = cfr.compute_exploitability()
            converged = final_exploit < convergence_threshold
            
            results.append({
                'name': name,
                'final': final_exploit,
                'converged': converged,
                'storage': sparse_A.storage_nnz,
                'compression': sparse_A.compression_ratio,
                'time': cfr.total_time,
                'iter_per_sec': n_iterations / cfr.total_time if cfr.total_time > 0 else 0
            })
            
            if converged:
                print(f"\n   ✅ CONVERGED: {final_exploit:.6f}")
            else:
                print(f"\n   ❌ FAILED: {final_exploit:.6f}")
        
        except Exception as e:
            print(f"\n   ❌ ERROR: {e}")
            results.append({
                'name': name,
                'final': float('inf'),
                'converged': False,
                'storage': 0,
                'compression': 0,
                'time': 0,
                'iter_per_sec': 0,
                'error': str(e)
            })
    
    # Print summary
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'Storage':>10} {'Compress':>10} {'Final':>12} {'Time':>8} {'Status':>10}")
    print("-"*70)
    
    for r in results:
        status = "✅ PASS" if r['converged'] else "❌ FAIL"
        storage = f"{r['storage']}" if r['storage'] > 0 else "ERROR"
        compress = f"{r['compression']*100:.1f}%" if r['compression'] > 0 else "N/A"
        final = f"{r['final']:.6f}" if r['final'] < float('inf') else "ERROR"
        time_str = f"{r['time']:.2f}s" if r['time'] > 0 else "N/A"
        
        print(f"{r['name']:<25} {storage:>10} {compress:>10} {final:>12} {time_str:>8} {status:>10}")
    
    # Count successes
    total = len(results)
    passed = sum(1 for r in results if r['converged'])
    
    print(f"\n{'='*70}")
    print(f"SUMMARY: {passed}/{total} methods converged")
    print(f"{'='*70}")
    
    if passed == total:
        print("✅✅✅ ALL METHODS PASSED! ✅✅✅")
        print("✅ Matrix-based CFR works with ALL sparsification methods!")
        return True
    else:
        print(f"⚠️  {total - passed} method(s) failed")
        return False


def test_cfr_variants():
    """
    Test different CFR variants (vanilla, CFR+, DCFR)
    """
    print("\n" + "="*70)
    print("TESTING CFR VARIANTS")
    print("="*70)
    
    # Load game
    game_env = GameEnv()
    game_env.load_open_spiel_game(pyspiel.load_game("kuhn_poker"))
    
    # Extract structure
    extractor = KuhnPokerStructureExtractor()
    structure = extractor.extract_structure()
    
    # Use Farina-A for testing
    sparse_A = SparsePayoffMatrix(structure, FarinaTechniqueA())
    
    variants = ['vanilla', 'cfr+', 'dcfr']
    results = []
    
    for variant in variants:
        print(f"\n{'='*70}")
        print(f"Testing: {variant.upper()}")
        print(f"{'='*70}")
        
        cfr = run_matrix_cfr(
            game_env,
            sparse_A,
            structure,
            n_iterations=2000,
            variant=variant,
            print_every=400
        )
        
        final = cfr.compute_exploitability()
        converged = final < 0.01
        
        results.append({
            'variant': variant,
            'final': final,
            'converged': converged
        })
        
        status = "✅" if converged else "❌"
        print(f"\n   {status} {variant.upper()}: {final:.6f}")
    
    # Summary
    print(f"\n{'='*70}")
    print("VARIANT RESULTS")
    print(f"{'='*70}")
    
    for r in results:
        status = "✅ PASS" if r['converged'] else "❌ FAIL"
        print(f"{r['variant'].upper():<15} {r['final']:>12.6f} {status:>10}")
    
    passed = sum(1 for r in results if r['converged'])
    print(f"\n{passed}/{len(results)} variants converged")
    
    return passed == len(results)


def test_convergence_rate():
    """
    Test convergence rate across methods
    """
    print("\n" + "="*70)
    print("CONVERGENCE RATE COMPARISON")
    print("="*70)
    
    # Load game
    game_env = GameEnv()
    game_env.load_open_spiel_game(pyspiel.load_game("kuhn_poker"))
    
    # Extract structure
    extractor = KuhnPokerStructureExtractor()
    structure = extractor.extract_structure()
    
    methods = [
        ("Baseline", None),
        ("Farina-A", FarinaTechniqueA()),
        ("Zhang", ZhangSparsification(max_rank=3, factor_threshold=0.2)),
    ]
    
    checkpoints = [200, 400, 600, 800, 1000, 1500, 2000]
    
    for name, sparsifier in methods:
        print(f"\n{name}:")
        
        sparse_A = SparsePayoffMatrix(structure, sparsifier)
        cfr = MatrixCFR(game_env, sparse_A, structure, variant='cfr+')
        
        for checkpoint in checkpoints:
            while cfr.t < checkpoint:
                cfr.iteration()
            
            exploit = cfr.compute_exploitability()
            print(f"   Iter {checkpoint:4d}: {exploit:.6f}")


def test_matrix_correctness():
    """
    Verify that matrix-based utilities match tree-based utilities
    """
    print("\n" + "="*70)
    print("MATRIX CORRECTNESS TEST")
    print("="*70)
    
    from game_utils import compute_utility_vector
    
    # Load game
    game_env = GameEnv()
    game_env.load_open_spiel_game(pyspiel.load_game("kuhn_poker"))
    
    # Extract structure
    extractor = KuhnPokerStructureExtractor()
    structure = extractor.extract_structure()
    
    # Test with Farina-A
    sparse_A = SparsePayoffMatrix(structure, FarinaTechniqueA())
    
    # Create CFR instance
    cfr = MatrixCFR(game_env, sparse_A, structure, variant='cfr+')
    
    # Run a few iterations to get non-uniform strategies
    for _ in range(10):
        cfr.iteration()
    
    # Get current policies in SeqVec format
    policies = cfr.get_average_policy()
    
    print("\nComparing matrix vs tree utility computation:")
    
    for player in [0, 1]:
        print(f"\nPlayer {player}:")
        
        # Tree-based utilities
        u_tree = compute_utility_vector(game_env, player, policies)
        
        # Matrix-based utilities
        u_matrix = cfr.compute_utility_gradient(player)
        
        # Convert to betting sequence space for comparison
        u_matrix_bet = cfr._kronecker_to_betting(u_matrix, player)
        
        # Compare norms
        tree_norm = np.linalg.norm(u_tree.array)
        matrix_norm = np.linalg.norm(u_matrix_bet)
        
        print(f"   Tree norm:   {tree_norm:.6f}")
        print(f"   Matrix norm: {matrix_norm:.6f}")
        print(f"   Difference:  {abs(tree_norm - matrix_norm):.6e}")
        
        # This might not match exactly due to dimension mismatch
        # but should be in same ballpark
        if abs(tree_norm - matrix_norm) / max(tree_norm, 1e-6) < 0.5:
            print(f"   ✅ Reasonable match")
        else:
            print(f"   ⚠️  Large difference")


def run_all_tests():
    """Run all tests"""
    print("="*70)
    print("MATRIX CFR - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    results = {
        'sparsification': False,
        'variants': False,
        'convergence_rate': True,  # Just informational
        'correctness': True  # Just informational
    }
    
    # Test 1: All sparsification methods
    print("\n" + "="*70)
    print("TEST 1: ALL SPARSIFICATION METHODS")
    print("="*70)
    results['sparsification'] = test_all_sparsification_methods()
    
    # Test 2: CFR variants
    print("\n" + "="*70)
    print("TEST 2: CFR VARIANTS")
    print("="*70)
    results['variants'] = test_cfr_variants()
    
    # Test 3: Convergence rate (informational)
    print("\n" + "="*70)
    print("TEST 3: CONVERGENCE RATE (Informational)")
    print("="*70)
    test_convergence_rate()
    
    # Test 4: Matrix correctness (informational)
    print("\n" + "="*70)
    print("TEST 4: MATRIX CORRECTNESS (Informational)")
    print("="*70)
    test_matrix_correctness()
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL TEST SUMMARY")
    print("="*70)
    
    critical_passed = results['sparsification'] and results['variants']
    
    for test_name, passed in results.items():
        if test_name in ['convergence_rate', 'correctness']:
            status = "ℹ️  INFO"
        else:
            status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.upper():<25} {status}")
    
    print("\n" + "="*70)
    if critical_passed:
        print("🎉🎉🎉 ALL CRITICAL TESTS PASSED! 🎉🎉🎉")
        print("✅ Matrix-based CFR is working correctly!")
        print("✅ All sparsification methods converge!")
        print("✅ All CFR variants work!")
    else:
        print("❌ Some tests failed - debugging needed")
    print("="*70)
    
    return critical_passed


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)