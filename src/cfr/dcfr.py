"""
Discounted CFR implementation
Based on Brown & Sandholm 2019: "Solving Imperfect-Information Games via Discounted Regret Minimization"
"""
from .vanilla import VanillaCFR


class DiscountedCFR(VanillaCFR):
    """
    Discounted CFR (DCFR)
    
    Discounts accumulated regrets before each iteration:
    - Positive regrets multiplied by: t^α / (t^α + 1)
    - Negative regrets multiplied by: t^β / (t^β + 1)  
    - Strategy sum multiplied by: (t / (t + 1))^γ
    
    Standard parameters from paper: α=1.5, β=0, γ=2
    """
    
    def __init__(self, game, alpha=1.5, beta=0.0, gamma=2.0):
        super().__init__(game)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        print(f"✅ Initialized DCFR (α={alpha}, β={beta}, γ={gamma}) for {self.game_name}")
    
    def iteration(self):
        """
        DCFR iteration with discounting
        
        From paper: "multiplying accumulated positive regrets by t^α/(t^α+1), 
        negative regrets by t^β/(t^β+1), and contributions to the average 
        strategy by (t/(t+1))^γ on each iteration t"
        """
        self.t += 1
        
        # Apply discounting BEFORE running CFR iteration
        if self.t > 1:
            # Compute discount factors following paper's formula
            t_alpha = self.t ** self.alpha
            discount_pos = t_alpha / (t_alpha + 1)
            
            if self.beta > 0:
                t_beta = self.t ** self.beta
                discount_neg = t_beta / (t_beta + 1)
            else:
                # β=0 means negative regrets go to 0
                discount_neg = 0.0
            
            # Strategy discount: (t/(t+1))^γ
            discount_strategy = (self.t / (self.t + 1)) ** self.gamma
            
            # Apply discounting to regrets
            for key in list(self.cumulative_regrets.keys()):
                regret = self.cumulative_regrets[key]
                if regret > 0:
                    # Positive regrets
                    self.cumulative_regrets[key] *= discount_pos
                else:
                    # Negative regrets
                    self.cumulative_regrets[key] *= discount_neg
            
            # Apply discounting to strategy sum
            for key in list(self.cumulative_strategy.keys()):
                self.cumulative_strategy[key] *= discount_strategy
        
        # Now run standard CFR iteration (adds new regrets to discounted old ones)
        for player in range(self.game.num_players()):
            self._compute_counterfactual_regret(
                self.game.new_initial_state(),
                player,
                [1.0] * self.game.num_players()
            )