"""
Definitive tests for Farina techniques
Tests reconstruction error to verify correctness
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from games.kuhn_structure import KuhnPokerStructureExtractor
from games.leduc_structure import LeducPokerStructureExtractor
from sparsification.farina import (
    FarinaTechniqueA,
    FarinaTechniqueB,
    verify_farina_reconstruction
)


def test_farina_on_game(game_name, extractor_class):
    """Test both Farina techniques on a game"""
    print(f"\n{'='*60}")
    print(f"TESTING: {game_name}")
    print(f"{'='*60}")
    
    # Extract structure
    extractor = extractor_class()
    structure = extractor.extract_structure()
    
    print(f"\nGame structure:")
    print(f"  F: {structure.F.shape}, {np.count_nonzero(structure.F)} nnz")
    print(f"  S: {structure.S.shape}, {np.count_nonzero(structure.S)} nnz")
    print(f"  W: {structure.W.shape}, {np.count_nonzero(structure.W)} nnz")
    print(f"  Win-lose matrix W:")
    print(structure.W)
    
    # Test Technique A
    print(f"\n{'='*60}")
    print("TECHNIQUE A")
    print(f"{'='*60}")
    
    tech_a = FarinaTechniqueA()
    result_a = tech_a.sparsify_structure(structure)
    
    print(f"\nW sparsification:")
    print(f"  Original W: {np.count_nonzero(structure.W)} nnz")
    print(f"  W_sparse: {result_a['W_sparse'].nnz} nnz")
    print(f"  Rank: {result_a['U_W'].shape[1]}")
    
    passed_a = verify_farina_reconstruction(structure, result_a, "Technique A")
    
    # Test Technique B
    print(f"\n{'='*60}")
    print("TECHNIQUE B")
    print(f"{'='*60}")
    
    tech_b = FarinaTechniqueB()
    result_b = tech_b.sparsify_structure(structure)
    
    print(f"\nDifference matrix Y = D @ W:")
    print(result_b['Y'])
    print(f"Y sparsity: {np.count_nonzero(result_b['Y'])}/{structure.W.size} = {np.count_nonzero(result_b['Y'])/structure.W.size*100:.1f}%")
    
    passed_b = verify_farina_reconstruction(structure, result_b, "Technique B")
    
    return passed_a, passed_b


def main():
    """Run all tests"""
    print("="*60)
    print("FARINA TECHNIQUES: DEFINITIVE VERIFICATION")
    print("="*60)
    
    results = {}
    
    # Test on Kuhn
    passed_a, passed_b = test_farina_on_game("Kuhn Poker", KuhnPokerStructureExtractor)
    results['Kuhn'] = {'A': passed_a, 'B': passed_b}
    
    # Test on Leduc
    passed_a, passed_b = test_farina_on_game("Leduc Poker", LeducPokerStructureExtractor)
    results['Leduc'] = {'A': passed_a, 'B': passed_b}
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    all_passed = True
    for game, techs in results.items():
        print(f"\n{game}:")
        for tech, passed in techs.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  Technique {tech}: {status}")
            all_passed = all_passed and passed
    
    if all_passed:
        print(f"\n{'='*60}")
        print("✅ ALL TESTS PASSED!")
        print("✅ Farina implementations are CORRECT!")
        print(f"{'='*60}")
    else:
        print(f"\n❌ SOME TESTS FAILED!")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)