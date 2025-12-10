"""
Vanilla CFR - Exact replication of OpenSpiel's algorithm
"""
import pyspiel
import numpy as np
from collections import defaultdict
import time


class VanillaCFR:
    
    def __init__(self, game):
        if hasattr(game, 'game'):
            self.game = game.game
            self.game_name = game.game_name
        else:
            self.game = game
            self.game_name = str(game)
        
        # Use flat dict with (infoset, action) keys
        self.cumulative_regrets = defaultdict(float)
        self.cumulative_strategy = defaultdict(float)
        
        # Store info about infosets
        self.info_states = {}  # infoset_str -> list of legal actions
        
        self.t = 0
        print(f"✅ Initialized Vanilla CFR for {self.game_name}")
    
    def iteration(self):
        """One CFR iteration"""
        self.t += 1
        for player in range(self.game.num_players()):
            self._compute_counterfactual_regret(
                self.game.new_initial_state(),
                player,
                [1.0] * self.game.num_players()
            )
    
    def _compute_counterfactual_regret(self, state, player, reach_probs):
        """
        Compute counterfactual regrets
        
        Args:
            state: Current state
            player: Player for whom we compute regrets
            reach_probs: List of reach probabilities for each player
        
        Returns:
            Value of state for player
        """
        if state.is_terminal():
            return state.returns()[player]
        
        if state.is_chance_node():
            value = 0.0
            for action, prob in state.chance_outcomes():
                value += prob * self._compute_counterfactual_regret(
                    state.child(action), player, reach_probs
                )
            return value
        
        current_player = state.current_player()
        info_state = state.information_state_string(current_player)
        legal_actions = state.legal_actions()
        
        # Store legal actions for this infoset
        if info_state not in self.info_states:
            self.info_states[info_state] = legal_actions
        
        # Get current strategy
        strategy = self._get_infostate_policy(info_state, legal_actions)
        
        # Compute value for each action
        action_values = []
        for action in legal_actions:
            new_reach = reach_probs.copy()
            new_reach[current_player] *= strategy[action]
            
            action_values.append(
                self._compute_counterfactual_regret(
                    state.child(action), player, new_reach
                )
            )
        
        # Value of this state
        value = sum(strategy[a] * action_values[i] 
                   for i, a in enumerate(legal_actions))
        
        # Update regrets and strategy (only for player's infosets)
        if current_player == player:
            # Counterfactual reach (product of opponent reaches)
            cf_reach = 1.0
            for p in range(self.game.num_players()):
                if p != player:
                    cf_reach *= reach_probs[p]
            
            for i, action in enumerate(legal_actions):
                regret = action_values[i] - value
                self.cumulative_regrets[(info_state, action)] += cf_reach * regret
                
                # Accumulate strategy
                self.cumulative_strategy[(info_state, action)] += \
                    reach_probs[player] * strategy[action]
        
        return value
    
    def _get_infostate_policy(self, info_state, legal_actions):
        """Get current strategy using regret matching"""
        regrets = [max(0.0, self.cumulative_regrets[(info_state, a)]) 
                  for a in legal_actions]
        
        sum_regrets = sum(regrets)
        
        if sum_regrets > 0:
            policy = [r / sum_regrets for r in regrets]
        else:
            # Uniform
            policy = [1.0 / len(legal_actions)] * len(legal_actions)
        
        # Return as dict
        return {legal_actions[i]: policy[i] for i in range(len(legal_actions))}
    
    def get_average_strategy(self):
        """Get average strategy"""
        average_strategy = {}
        
        for info_state, legal_actions in self.info_states.items():
            total = sum(self.cumulative_strategy[(info_state, a)] 
                       for a in legal_actions)
            
            if total > 0:
                average_strategy[info_state] = {
                    a: self.cumulative_strategy[(info_state, a)] / total
                    for a in legal_actions
                }
            else:
                n = len(legal_actions)
                average_strategy[info_state] = {a: 1.0 / n for a in legal_actions}
        
        return average_strategy
    
    def compute_exploitability(self):
        """Compute exploitability"""
        from utils.metrics import compute_exploitability
        return compute_exploitability(self.game, self.get_average_strategy())
    
    def train(self, iterations, log_every=1000):
        """Train"""
        print(f"\nTraining Vanilla CFR for {iterations} iterations...")
        start_time = time.time()
        
        for i in range(iterations):
            self.iteration()
            
            if (i + 1) % log_every == 0:
                elapsed = time.time() - start_time
                print(f"  Iter {i+1:6d} | {elapsed:6.1f}s | {(i+1)/elapsed:6.1f} iter/s")
        
        total_time = time.time() - start_time
        print(f"✅ Complete: {total_time:.2f}s")
        return total_time