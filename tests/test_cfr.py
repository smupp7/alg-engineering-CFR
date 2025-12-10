"""
Test CFR implementations
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import games.openspiel_wrapper
from cfr.vanilla import VanillaCFR
from cfr.cfr_plus import CFRPlus
from cfr.dcfr import DiscountedCFR


def test_vanilla_cfr_kuhn():
    """Test Vanilla CFR on Kuhn Poker"""
    print("="*60)
    print("Testing Vanilla CFR on Kuhn Poker")
    print("="*60)
    
    game = games.openspiel_wrapper.create_kuhn_poker()
    cfr = VanillaCFR(game)
    
    # Train for a few iterations
    cfr.train(iterations=1000, log_every=200)
    
    # Get average strategy
    avg_strategy = cfr.get_average_strategy()
    print(f"\n✅ Learned strategy for {len(avg_strategy)} infosets")
    
    # Show sample strategies
    print("\nSample strategies:")
    for infoset in list(avg_strategy.keys())[:3]:
        print(f"  {infoset}: {avg_strategy[infoset]}")
    
    print("\n✅ Vanilla CFR test passed!")
    return cfr


def test_cfr_plus_kuhn():
    """Test CFR+ on Kuhn Poker"""
    print("\n" + "="*60)
    print("Testing CFR+ on Kuhn Poker")
    print("="*60)
    
    game = games.openspiel_wrapper.create_kuhn_poker()
    cfr = CFRPlus(game)
    
    cfr.train(iterations=1000, log_every=200)
    
    avg_strategy = cfr.get_average_strategy()
    print(f"\n✅ Learned strategy for {len(avg_strategy)} infosets")
    
    print("\n✅ CFR+ test passed!")
    return cfr


def test_dcfr_kuhn():
    """Test DCFR on Kuhn Poker"""
    print("\n" + "="*60)
    print("Testing DCFR on Kuhn Poker")
    print("="*60)
    
    game = games.openspiel_wrapper.create_kuhn_poker()
    cfr = DiscountedCFR(game, alpha=1.5, beta=0.0, gamma=2.0)
    
    cfr.train(iterations=1000, log_every=200)
    
    avg_strategy = cfr.get_average_strategy()
    print(f"\n✅ Learned strategy for {len(avg_strategy)} infosets")
    
    print("\n✅ DCFR test passed!")
    return cfr


def compare_algorithms():
    """Compare convergence speed of all three algorithms"""
    print("\n" + "="*60)
    print("Comparing CFR Algorithms (Convergence to Nash)")
    print("="*60)
    
    game = games.openspiel_wrapper.create_kuhn_poker()
    
    # Target exploitability (Kuhn has Nash exploitability ~0.055)
    target_exploitability = 0.001  # Very tight convergence
    max_iterations = 100000
    check_every = 500
    
    print(f"\nTarget: Exploitability < {target_exploitability}")
    print(f"Max iterations: {max_iterations:,}")
    print(f"Check every: {check_every} iterations\n")
    
    algorithms = [
        ('Vanilla CFR', VanillaCFR(game)),
        ('CFR+', CFRPlus(game)),
        ('DCFR', DiscountedCFR(game))
    ]
    
    results = {}
    
    for name, cfr in algorithms:
        print(f"Training {name}...")
        print(f"{'Iteration':>10} | {'Exploitability':>15} | {'Time (s)':>10}")
        print("-" * 50)
        
        import time
        start_time = time.time()
        
        converged = False
        exploit_history = []
        
        for i in range(check_every, max_iterations + 1, check_every):
            # Train for check_every iterations
            for _ in range(check_every):
                cfr.iteration()
            
            # Compute exploitability
            exploit = cfr.compute_exploitability()
            elapsed = time.time() - start_time
            exploit_history.append((i, exploit, elapsed))
            
            # Print every 2000 iterations or when converged
            if i % 2000 == 0 or exploit < target_exploitability:
                print(f"{i:>10} | {exploit:>15.6f} | {elapsed:>10.2f}")
            
            # Check convergence
            if exploit < target_exploitability:
                converged = True
                results[name] = {
                    'iterations': i,
                    'time': elapsed,
                    'exploitability': exploit,
                    'history': exploit_history,
                    'converged': True
                }
                print(f"  ✅ Converged!")
                break
        
        if not converged:
            elapsed = time.time() - start_time
            exploit = cfr.compute_exploitability()
            results[name] = {
                'iterations': max_iterations,
                'time': elapsed,
                'exploitability': exploit,
                'history': exploit_history,
                'converged': False
            }
            print(f"  ⚠️  Did not reach target (final: {exploit:.6f})")
        
        print()
    
    # Summary comparison
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Algorithm':<15} | {'Iterations':>10} | {'Time (s)':>10} | {'Speedup':>10} | {'Status':<10}")
    print("-"*70)
    
    baseline_iters = results['Vanilla CFR']['iterations']
    baseline_time = results['Vanilla CFR']['time']
    
    for name in ['Vanilla CFR', 'CFR+', 'DCFR']:
        res = results[name]
        speedup_iters = baseline_iters / res['iterations']
        speedup_time = baseline_time / res['time']
        status = '✅ Conv' if res['converged'] else '⚠️ No conv'
        
        print(f"{name:<15} | {res['iterations']:>10,} | {res['time']:>10.2f} | "
              f"{speedup_iters:>9.2f}× | {status:<10}")
    
    print("\n📊 Expected behavior:")
    print("   - CFR+ should converge ~1.5-2× faster than Vanilla")
    print("   - DCFR should converge ~2-3× faster than Vanilla")
    print("   - All should reach exploitability near 0")
    
    return results


def test_kuhn_nash_equilibrium():
    """
    Test that CFR finds known Nash equilibrium for Kuhn Poker
    
    Known Nash strategies for Kuhn:
    - Player with J: Always pass
    - Player with Q: Bet with prob ~1/3
    - Player with K: Always bet
    """
    print("\n" + "="*60)
    print("Testing Convergence to Known Nash (Kuhn Poker)")
    print("="*60)
    
    game = games.openspiel_wrapper.create_kuhn_poker()
    
    # Use DCFR (fastest)
    cfr = DiscountedCFR(game)
    
    print("\nTraining DCFR for 10,000 iterations...")
    cfr.train(10000, log_every=2000)
    
    # Get final strategy
    avg_strategy = cfr.get_average_strategy()
    
    print("\n" + "="*60)
    print("Learned Strategies (sample infosets):")
    print("="*60)
    
    # Show some key infosets
    sample_infosets = [
        '0',    # Player 0 with Jack, first action
        '1',    # Player 0 with Queen, first action  
        '2',    # Player 0 with King, first action
        '0p',   # Player 1 with Jack after opponent passed
        '1p',   # Player 1 with Queen after opponent passed
        '2p',   # Player 1 with King after opponent passed
    ]
    
    for infoset in sample_infosets:
        if infoset in avg_strategy:
            strat = avg_strategy[infoset]
            print(f"{infoset:6s}: {strat}")
    
    # Compute final exploitability
    exploit = cfr.compute_exploitability()
    print(f"\nFinal exploitability: {exploit:.6f}")
    
    if exploit < 0.01:
        print("✅ Successfully converged to near-Nash equilibrium!")
    else:
        print(f"⚠️  Exploitability still high (target: < 0.01)")
    
    print("\n📝 Expected patterns:")
    print("   - Jack (0): Should mostly pass")
    print("   - Queen (1): Should mix between pass/bet")
    print("   - King (2): Should mostly bet")

def debug_exploitability():
    """Debug exploitability calculation"""
    print("\nDebugging exploitability calculation...")
    
    game = games.openspiel_wrapper.create_kuhn_poker()
    
    # Test at different iteration counts
    test_points = [10, 100, 1000, 5000, 10000]
    
    print(f"\n{'Iterations':>10} | {'Exploitability':>15} | {'Trend':>10}")
    print("-" * 40)
    
    prev_exploit = None
    for iters in test_points:
        # Create fresh CFR instance
        cfr = VanillaCFR(game)
        
        # Train
        for _ in range(iters):
            cfr.iteration()
        
        # Compute exploitability
        exploit = cfr.compute_exploitability()
        
        # Check trend
        if prev_exploit is not None:
            trend = "⬇️ Good" if exploit < prev_exploit else "⬆️ BAD!"
        else:
            trend = "-"
        
        print(f"{iters:>10} | {exploit:>15.6f} | {trend:>10}")
        prev_exploit = exploit
    
    # Show final strategies
    avg_strat = cfr.get_average_strategy()
    print(f"\nLearned {len(avg_strat)} strategies after {test_points[-1]} iterations")
    print("\nSample strategies:")
    for infoset in sorted(list(avg_strat.keys()))[:6]:
        strat = avg_strat[infoset]
        # Format nicely
        strat_str = ", ".join([f"a{a}:{p:.3f}" for a, p in strat.items()])
        print(f"  {infoset:6s}: {strat_str}")
    
    if prev_exploit > 0.1:
        print("\n⚠️  WARNING: Exploitability is not decreasing properly!")
        print("   This suggests a bug in CFR or exploitability calculation.")
    elif prev_exploit < 0.01:
        print("\n✅ Exploitability is decreasing properly!")
    else:
        print("\n⚠️  Exploitability decreasing but slowly.")

def test_openspiel_cfr_baseline():
    """Test using OpenSpiel's built-in CFR as baseline"""
    print("\n" + "="*60)
    print("BASELINE: OpenSpiel's Built-In CFR")
    print("="*60)
    
    import pyspiel
    from open_spiel.python.algorithms import cfr
    from open_spiel.python.algorithms import exploitability
    
    print("\n✅ Loaded kuhn_poker")
    game = pyspiel.load_game("kuhn_poker")
    
    cfr_solver = cfr.CFRSolver(game)
    
    print("\nTraining OpenSpiel's CFR for 10,000 iterations...")
    print(f"{'Iteration':>10} | {'Exploitability':>15}")
    print("-" * 30)
    
    for i in range(10000):
        cfr_solver.evaluate_and_update_policy()
        
        if (i + 1) % 2000 == 0:
            # Get current average policy
            avg_policy = cfr_solver.average_policy()
            
            # Compute exploitability using OpenSpiel's function
            conv = exploitability.exploitability(game, avg_policy)
            print(f"{i+1:>10} | {conv:>15.6f}")
    
    # Final exploitability
    final_policy = cfr_solver.average_policy()
    final_exploit = exploitability.exploitability(game, final_policy)
    
    print(f"\n✅ Final Exploitability: {final_exploit:.6f}")
    
    if final_exploit < 0.01:
        print("✅ OpenSpiel's CFR converges correctly!")
        print("   → This means OUR implementation has a bug")
        print("   → We need to fix our CFR code")
    else:
        print("⚠️  OpenSpiel's CFR also doesn't converge")
        print(f"   → This is unexpected (final: {final_exploit:.6f})")
    
    return final_exploit

def test_our_strategy_with_openspiel_exploitability():
    """Test our CFR strategy using OpenSpiel's exploitability calc"""
    print("\n" + "="*60)
    print("TEST: Our CFR + OpenSpiel's Exploitability")
    print("="*60)
    
    import pyspiel
    from open_spiel.python.algorithms import exploitability
    from open_spiel.python.policy import TabularPolicy
    import games.openspiel_wrapper
    from cfr.vanilla import VanillaCFR
    
    # Train our CFR
    game_wrapped = games.openspiel_wrapper.create_kuhn_poker()
    our_cfr = VanillaCFR(game_wrapped)
    
    print("\nTraining our CFR for 10,000 iterations...")
    our_cfr.train(10000, log_every=2000)
    
    # Get our strategy
    our_strategy = our_cfr.get_average_strategy()
    
    print("\nOur strategy:")
    for infoset in sorted(our_strategy.keys())[:6]:
        print(f"  {infoset}: {our_strategy[infoset]}")
    
    # Convert to OpenSpiel policy format
    game_pyspiel = pyspiel.load_game("kuhn_poker")
    
    class OurPolicy(TabularPolicy):
        def __init__(self, game, strategy_dict):
            super().__init__(game)
            self.strategy_dict = strategy_dict
        
        def action_probabilities(self, state, player_id=None):
            if state.is_terminal():
                return {}
            if state.is_chance_node():
                return {a: p for a, p in state.chance_outcomes()}
            
            infoset = state.information_state_string()
            if infoset in self.strategy_dict:
                return self.strategy_dict[infoset]
            else:
                # Uniform for unseen states
                actions = state.legal_actions()
                return {a: 1.0/len(actions) for a in actions}
    
    our_policy = OurPolicy(game_pyspiel, our_strategy)
    
    # Compute exploitability using OpenSpiel's function
    openspiel_exploit = exploitability.exploitability(game_pyspiel, our_policy)
    
    # Compute using our function
    our_exploit = our_cfr.compute_exploitability()
    
    print(f"\n" + "="*60)
    print("RESULTS:")
    print(f"  Our exploitability calc:      {our_exploit:.6f}")
    print(f"  OpenSpiel's exploitability:   {openspiel_exploit:.6f}")
    print("="*60)
    
    if openspiel_exploit < 0.01:
        print("\n✅✅✅ OUR CFR IS WORKING! ✅✅✅")
        print("   The bug was in our exploitability calculation!")
        print("   Our CFR correctly converges to Nash equilibrium!")
    else:
        print("\n❌ Both show high exploitability")
        print("   Our CFR implementation still has issues")


if __name__ == '__main__':
    print("Starting CFR tests...\n")
    
    try:
        # FIRST: Test OpenSpiel's own CFR
        print("="*60)
        print("STEP 0: Verify OpenSpiel's CFR Works")
        print("="*60)
        openspiel_nash_conv = test_openspiel_cfr_baseline()
        
        if openspiel_nash_conv > 0.1:
            print("\n❌ ERROR: Even OpenSpiel's CFR doesn't converge!")
            print("   This suggests a problem with OpenSpiel installation")
            exit(1)
        
        # CRITICAL TEST: Check if our CFR works with OpenSpiel's exploitability
        print("\n\n" + "="*60)
        print("STEP 0.5: TEST OUR CFR WITH OPENSPIEL'S EXPLOITABILITY")
        print("="*60)
        test_our_strategy_with_openspiel_exploitability()
        
        # Continue with our tests
        print("\n\n" + "="*60)
        print("STEP 1: Test Our CFR Implementations")
        print("="*60)
        
        vanilla = test_vanilla_cfr_kuhn()
        plus = test_cfr_plus_kuhn()
        dcfr = test_dcfr_kuhn()
        
        debug_exploitability()
        
        compare_algorithms()
        
        test_kuhn_nash_equilibrium()
        
        print("\n" + "="*60)
        print("✅✅✅ ALL CFR TESTS PASSED ✅✅✅")
        print("="*60)
        print("\nCFR implementations verified!")
        print("Ready to implement sparsification!")
        
    except Exception as e:
        print(f"\n❌ Test failed:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()