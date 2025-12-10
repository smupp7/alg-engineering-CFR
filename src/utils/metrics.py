"""
Metrics for evaluating CFR strategies
"""
import pyspiel
from open_spiel.python.algorithms import exploitability as openspiel_exploitability
from open_spiel.python.policy import TabularPolicy


class DictPolicy(TabularPolicy):
    """
    Convert our dict-based strategy to OpenSpiel Policy format
    """
    
    def __init__(self, game, strategy_dict):
        """
        Args:
            game: pyspiel game object
            strategy_dict: Dict[infoset_str -> Dict[action -> prob]]
        """
        super().__init__(game)
        self.strategy_dict = strategy_dict
    
    def action_probabilities(self, state, player_id=None):
        """Get action probabilities for a state"""
        if state.is_terminal():
            return {}
        
        if state.is_chance_node():
            return {a: p for a, p in state.chance_outcomes()}
        
        infoset = state.information_state_string()
        
        if infoset in self.strategy_dict:
            return self.strategy_dict[infoset]
        else:
            # Uniform for unseen infosets
            actions = state.legal_actions()
            return {a: 1.0 / len(actions) for a in actions}


def compute_exploitability(game, avg_strategy):
    """
    Compute exploitability using OpenSpiel's proven implementation
    
    Args:
        game: pyspiel game object
        avg_strategy: Dict mapping infoset -> (action -> probability)
    
    Returns:
        Exploitability value (0 = Nash equilibrium)
    """
    # Convert our strategy to OpenSpiel policy format
    policy = DictPolicy(game, avg_strategy)
    
    # Use OpenSpiel's exploitability calculator
    return openspiel_exploitability.exploitability(game, policy)


def compute_nash_conv(game, strategy_profile):
    """
    Compute Nash Convergence (exploitability / 2)
    """
    return compute_exploitability(game, strategy_profile) / 2.0


def _compute_best_response_value(state, player, opponent_strategy):
    """
    Compute value of best response against opponent strategy
    
    This computes what player could achieve by playing optimally
    while opponent follows their strategy
    
    Args:
        state: Current game state
        player: Player computing best response (0 or 1)
        opponent_strategy: Opponent's strategy dict
    
    Returns:
        Expected value for player playing best response
    """
    if state.is_terminal():
        return state.returns()[player]
    
    if state.is_chance_node():
        # Average over chance outcomes
        value = 0.0
        for action, prob in state.chance_outcomes():
            child = state.child(action)
            value += prob * _compute_best_response_value(
                child, player, opponent_strategy
            )
        return value
    
    current_player = state.current_player()
    
    if current_player == player:
        # Player's turn: play BEST action (maximize)
        best_value = float('-inf')
        for action in state.legal_actions():
            child = state.child(action)
            value = _compute_best_response_value(
                child, player, opponent_strategy
            )
            best_value = max(best_value, value)
        return best_value if best_value > float('-inf') else 0.0
    else:
        # Opponent's turn: they play according to their strategy
        infoset = state.information_state_string(current_player)
        strategy = opponent_strategy.get(infoset, {})
        
        if not strategy:
            # Uniform if no strategy defined
            actions = state.legal_actions()
            strategy = {a: 1.0 / len(actions) for a in actions}
        
        value = 0.0
        for action in state.legal_actions():
            child = state.child(action)
            prob = strategy.get(action, 0.0)
            value += prob * _compute_best_response_value(
                child, player, opponent_strategy
            )
        return value


def _compute_strategy_value(state, player, strategy_profile):
    """
    Compute expected value when both players follow the strategy profile
    
    Args:
        state: Current game state
        player: Player whose value we're computing
        strategy_profile: Strategy for both players
    
    Returns:
        Expected value for player
    """
    if state.is_terminal():
        return state.returns()[player]
    
    if state.is_chance_node():
        value = 0.0
        for action, prob in state.chance_outcomes():
            child = state.child(action)
            value += prob * _compute_strategy_value(
                child, player, strategy_profile
            )
        return value
    
    current_player = state.current_player()
    infoset = state.information_state_string(current_player)
    strategy = strategy_profile.get(infoset, {})
    
    if not strategy:
        # Uniform if no strategy
        actions = state.legal_actions()
        strategy = {a: 1.0 / len(actions) for a in actions}
    
    value = 0.0
    for action in state.legal_actions():
        child = state.child(action)
        prob = strategy.get(action, 0.0)
        value += prob * _compute_strategy_value(
            child, player, strategy_profile
        )
    return value


