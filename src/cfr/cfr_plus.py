"""
CFR+ implementation
Tammelin et al. 2014
"""
from .vanilla import VanillaCFR


class CFRPlus(VanillaCFR):
    """
    CFR+ - Regrets are floored at 0 as they're updated
    
    Key difference: max(0, cumulative_regret + new_regret)
    instead of: max(0, cumulative_regret) + new_regret
    """
    
    def __init__(self, game):
        super().__init__(game)
        print(f"✅ Initialized CFR+ for {self.game_name}")
    
    def _compute_counterfactual_regret(self, state, player, reach_probs):
        """Override to floor regrets as they're added (not after)"""
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
        
        if info_state not in self.info_states:
            self.info_states[info_state] = legal_actions
        
        strategy = self._get_infostate_policy(info_state, legal_actions)
        
        # Compute action values
        action_values = []
        for action in legal_actions:
            new_reach = reach_probs.copy()
            new_reach[current_player] *= strategy[action]
            
            action_values.append(
                self._compute_counterfactual_regret(
                    state.child(action), player, new_reach
                )
            )
        
        value = sum(strategy[a] * action_values[i] 
                   for i, a in enumerate(legal_actions))
        
        # Update regrets (only for player's infosets)
        if current_player == player:
            cf_reach = 1.0
            for p in range(self.game.num_players()):
                if p != player:
                    cf_reach *= reach_probs[p]
            
            for i, action in enumerate(legal_actions):
                regret = action_values[i] - value
                key = (info_state, action)
                
                # CFR+: Floor regrets AS they're added (key difference!)
                self.cumulative_regrets[key] = max(
                    0.0,
                    self.cumulative_regrets[key] + cf_reach * regret
                )
                
                # Strategy: weight by t^2 (from paper)
                self.cumulative_strategy[key] += reach_probs[player] * strategy[action] * (self.t ** 2)
        
        return value