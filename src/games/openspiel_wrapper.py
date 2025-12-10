"""
OpenSpiel game wrapper with payoff matrix extraction
"""
import numpy as np
import pyspiel
from typing import Dict, List, Tuple
from collections import defaultdict


class OpenSpielGame:
    """
    Wrapper around OpenSpiel games for CFR algorithms
    """
    
    def __init__(self, game_name: str):
        self.game = pyspiel.load_game(game_name)
        self.game_name = game_name
        self.is_poker = 'poker' in game_name.lower()
        self._infosets = None
        self._payoff_matrix = None
        print(f"✅ Loaded {game_name}")
    
    def get_infosets(self) -> Dict[int, List[str]]:
        """Get all information sets for each player"""
        if self._infosets is not None:
            return self._infosets
        
        print("Building information sets...")
        infosets = {0: set(), 1: set()}
        
        def traverse(state):
            if state.is_terminal():
                return
            if state.is_chance_node():
                for action, _ in state.chance_outcomes():
                    traverse(state.child(action))
            else:
                player = state.current_player()
                infoset = state.information_state_string(player)
                infosets[player].add(infoset)
                for action in state.legal_actions():
                    traverse(state.child(action))
        
        traverse(self.game.new_initial_state())
        self._infosets = {p: sorted(list(s)) for p, s in infosets.items()}
        print(f"  Player 0: {len(self._infosets[0])} infosets")
        print(f"  Player 1: {len(self._infosets[1])} infosets")
        return self._infosets
    
    def get_payoff_matrix(self) -> np.ndarray:
        """Extract payoff matrix in sequence form"""
        if self._payoff_matrix is not None:
            return self._payoff_matrix
        
        print("Extracting payoff matrix...")
        infosets = self.get_infosets()
        pure_strategies = self._enumerate_pure_strategies()
        
        n0 = len(pure_strategies[0])
        n1 = len(pure_strategies[1])
        print(f"  P0 strategies: {n0}")
        print(f"  P1 strategies: {n1}")
        
        if n0 == 0 or n1 == 0:
            print("⚠️  Warning: No strategies found!")
            return np.zeros((1, 1))
        
        A = np.zeros((n0, n1))
        for i, strat0 in enumerate(pure_strategies[0]):
            for j, strat1 in enumerate(pure_strategies[1]):
                A[i, j] = self._compute_expected_payoff(strat0, strat1)
        
        self._payoff_matrix = A
        nnz = np.count_nonzero(A)
        print(f"  Nonzeros: {nnz:,} / {A.size:,}")
        return A
    
    def _enumerate_pure_strategies(self) -> Tuple[List[Dict], List[Dict]]:
        """Enumerate all pure strategies for both players"""
        infosets = self.get_infosets()
        
        def enumerate_for_player(player: int) -> List[Dict]:
            player_infosets = infosets[player]
            if len(player_infosets) == 0:
                return [{}]
            
            infoset_actions = self._get_infoset_actions(player)
            strategies = [{}]
            
            for infoset in player_infosets:
                # Safely get actions
                actions = infoset_actions.get(infoset, [])
                if not actions:
                    print(f"⚠️  No actions for infoset '{infoset}' (player {player})")
                    continue
                
                new_strategies = []
                for strategy in strategies:
                    for action in actions:
                        new_strat = strategy.copy()
                        new_strat[infoset] = action
                        new_strategies.append(new_strat)
                
                if new_strategies:
                    strategies = new_strategies
            
            return strategies
        
        return (enumerate_for_player(0), enumerate_for_player(1))
    
    def _get_infoset_actions(self, player: int) -> Dict[str, List[int]]:
        """Get available actions at each infoset"""
        infoset_actions = defaultdict(set)
        
        def traverse(state):
            if state.is_terminal():
                return
            if state.is_chance_node():
                for action, _ in state.chance_outcomes():
                    traverse(state.child(action))
            else:
                current_player = state.current_player()
                if current_player == player:
                    infoset = state.information_state_string(player)
                    for action in state.legal_actions():
                        infoset_actions[infoset].add(action)
                for action in state.legal_actions():
                    traverse(state.child(action))
        
        traverse(self.game.new_initial_state())
        
        # Safely convert to sorted lists
        result = {}
        for infoset, actions in infoset_actions.items():
            result[infoset] = sorted(list(actions)) if actions else []
        return result
    
    def _compute_expected_payoff(self, strategy0: Dict, strategy1: Dict) -> float:
        """Compute expected payoff for given strategies"""
        strategies = [strategy0, strategy1]
        
        def traverse(state, prob: float) -> float:
            if state.is_terminal():
                return prob * state.returns()[0]
            
            if state.is_chance_node():
                value = 0.0
                for action, action_prob in state.chance_outcomes():
                    value += traverse(state.child(action), prob * action_prob)
                return value
            else:
                player = state.current_player()
                infoset = state.information_state_string(player)
                action = strategies[player].get(infoset)
                if action is None:
                    return 0.0
                return traverse(state.child(action), prob)
        
        return traverse(self.game.new_initial_state(), 1.0)
    
    def __repr__(self):
        return f"OpenSpielGame({self.game_name})"


def create_kuhn_poker():
    """Create Kuhn Poker game"""
    return OpenSpielGame('kuhn_poker')


def create_leduc_poker():
    """Create Leduc Poker game"""
    return OpenSpielGame('leduc_poker')