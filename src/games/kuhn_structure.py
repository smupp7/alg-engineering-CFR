"""
Extract Kronecker structure for Kuhn poker
"""
import numpy as np
import pyspiel
from games.poker_structure import PokerStructure  # Fixed import!
from typing import List, Tuple

class KuhnPokerStructureExtractor:
    """
    Extract Kronecker structure from Kuhn poker
    
    Kuhn poker:
    - 3 cards: J, Q, K
    - 1 betting round
    - Actions: pass, bet
    """
    
    def __init__(self):
        self.game = pyspiel.load_game("kuhn_poker")
        
        # Kuhn hands
        self.hands = ['J', 'Q', 'K']  # Jack, Queen, King
        self.n_hands = 3
        
        print("🃏 Extracting Kuhn Poker Kronecker Structure")
    
    def extract_structure(self) -> PokerStructure:
        """Extract complete Kronecker structure"""
        
        print("\n1️⃣ Building betting tree skeleton...")
        F, S, sequences1, sequences2 = self._build_betting_skeleton()
        
        print(f"   Fold matrix F: {F.shape}")
        print(f"   Showdown matrix S: {S.shape}")
        print(f"   P1 sequences: {len(sequences1)}")
        print(f"   P2 sequences: {len(sequences2)}")
        
        print("\n2️⃣ Building win-lose matrix...")
        W = self._build_win_lose_matrix()
        
        print(f"   Win-lose matrix W: {W.shape}")
        print(f"   W =\n{W}")
        
        print("\n3️⃣ Building hand belief distributions...")
        mu1, mu2, Lambda1, Lambda2 = self._build_hand_beliefs()
        
        print(f"   Hand beliefs: uniform over {self.n_hands} hands")
        
        print("\n4️⃣ Building incompatibility matrix...")
        H_incomp = self._build_incompatibility_matrix()
        C = self._build_C_matrix(mu1, mu2, Lambda1, Lambda2, H_incomp)
        
        print(f"   Incompatibility matrix H: {np.count_nonzero(H_incomp)} incompatible pairs")
        
        structure = PokerStructure(
            F=F, S=S, W=W,
            Lambda1=Lambda1, Lambda2=Lambda2,
            mu1=mu1, mu2=mu2,
            C=C, H_incomp=H_incomp,
            hands1=self.hands, hands2=self.hands,
            sequences1=sequences1, sequences2=sequences2
        )
        
        print("\n✅ Structure extraction complete!")
        return structure
    
    def _build_betting_skeleton(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Build betting tree skeleton (independent of hands)
        
        Returns F (fold payoffs) and S (showdown payoffs)
        """
        # Kuhn betting tree:
        # Root -> P1 acts (pass/bet) -> P2 acts (pass/bet/call) -> Terminal
        
        # P1 sequences: ['', 'p', 'b']
        # P2 sequences: ['', 'p', 'b', 'pp', 'pb', 'bp', 'bb']
        
        # Simplified: Map to terminal nodes
        # We need to traverse game tree and extract payoff structure
        
        sequences1 = []  # P1 action sequences
        sequences2 = []  # P2 action sequences
        terminal_states = []  # List of terminal states with sequences
        
        # Build sequence mapping by traversing tree
        self._enumerate_sequences(self.game.new_initial_state(), 
                                  [], [], sequences1, sequences2, terminal_states)
        
        # Remove duplicates and sort
        sequences1 = sorted(list(set(sequences1)))
        sequences2 = sorted(list(set(sequences2)))
        
        n_seq1 = len(sequences1)
        n_seq2 = len(sequences2)
        
        # Build F and S matrices
        F = np.zeros((n_seq1, n_seq2))
        S = np.zeros((n_seq1, n_seq2))
        
        # Map sequences to indices
        seq1_to_idx = {seq: i for i, seq in enumerate(sequences1)}
        seq2_to_idx = {seq: i for i, seq in enumerate(sequences2)}
        
        # Fill in payoffs from terminal states
        for seq1, seq2, is_showdown, pot_contrib1, pot_contrib2 in terminal_states:
            i = seq1_to_idx.get(seq1, -1)
            j = seq2_to_idx.get(seq2, -1)
            
            if i >= 0 and j >= 0:
                if is_showdown:
                    # Showdown: payoff depends on hands
                    # S stores pot contributions at showdown
                    S[i, j] = pot_contrib1  # P1's contribution
                else:
                    # Fold: fixed payoff
                    F[i, j] = pot_contrib1  # P1's payoff
        
        return F, S, sequences1, sequences2
    
    def _enumerate_sequences(self, state, seq1, seq2, sequences1, sequences2, terminals):
        """Recursively enumerate action sequences"""
        
        if state.is_terminal():
            # Terminal state
            seq1_str = ''.join(seq1)
            seq2_str = ''.join(seq2)
            
            sequences1.append(seq1_str)
            sequences2.append(seq2_str)
            
            # Determine if showdown or fold
            is_showdown = self._is_showdown(seq1, seq2)
            
            # Get pot contributions (simplified for Kuhn)
            pot_contrib1, pot_contrib2 = self._get_pot_contributions(seq1, seq2)
            
            terminals.append((seq1_str, seq2_str, is_showdown, pot_contrib1, pot_contrib2))
            return
        
        if state.is_chance_node():
            # Skip chance outcomes (hand dealing)
            for action, prob in state.chance_outcomes():
                self._enumerate_sequences(state.child(action), seq1.copy(), seq2.copy(),
                                        sequences1, sequences2, terminals)
            return
        
        # Player action node
        current_player = state.current_player()
        
        for action in state.legal_actions():
            action_str = self._action_to_string(action)
            
            new_seq1 = seq1.copy()
            new_seq2 = seq2.copy()
            
            if current_player == 0:
                new_seq1.append(action_str)
            else:
                new_seq2.append(action_str)
            
            self._enumerate_sequences(state.child(action), new_seq1, new_seq2,
                                    sequences1, sequences2, terminals)
    
    def _action_to_string(self, action: int) -> str:
        """Convert action to string"""
        # In Kuhn: 0=pass, 1=bet
        return 'p' if action == 0 else 'b'
    
    def _is_showdown(self, seq1: List[str], seq2: List[str]) -> bool:
        """Check if terminal state is a showdown"""
        # Showdown happens if no one folds
        # Fold patterns: p-p, b-p (someone passes after bet)
        seq_str = ''.join(seq1 + seq2)
        
        # Fold cases in Kuhn:
        # "pp" - both pass (showdown)
        # "pbp" - P1 bets, P2 passes (fold)
        # "bp" - P1 passes, P2 bets, P1 passes (fold)
        # "pbb" or "bb" - both bet (showdown)
        
        if 'bp' in seq_str or 'pbp' in seq_str:
            return False  # Fold
        return True  # Showdown
    
    def _get_pot_contributions(self, seq1: List[str], seq2: List[str]) -> Tuple[float, float]:
        """Get pot contributions for each player"""
        # Kuhn: ante=1, bet=1
        # Initial pot: each player antes 1
        
        contrib1 = 1.0  # Ante
        contrib2 = 1.0  # Ante
        
        # Count bets
        for action in seq1:
            if action == 'b':
                contrib1 += 1.0
        
        for action in seq2:
            if action == 'b':
                contrib2 += 1.0
        
        return contrib1, contrib2
    
    def _build_win_lose_matrix(self) -> np.ndarray:
        """
        Build win-lose matrix W[i,j] = outcome when P1 has hand i, P2 has hand j
        
        Values:
         1: P1 wins
        -1: P1 loses
         0: Tie (or incompatible)
        """
        W = np.zeros((self.n_hands, self.n_hands))
        
        hand_strength = {'J': 0, 'Q': 1, 'K': 2}
        
        for i, h1 in enumerate(self.hands):
            for j, h2 in enumerate(self.hands):
                if h1 == h2:
                    # Incompatible (same card can't be dealt twice)
                    W[i, j] = 0
                elif hand_strength[h1] > hand_strength[h2]:
                    # P1 wins
                    W[i, j] = 1
                else:
                    # P1 loses
                    W[i, j] = -1
        
        return W
    
    def _build_hand_beliefs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build hand belief distributions (uniform for Kuhn)"""
        
        # Uniform beliefs over 3 hands
        mu1 = np.ones(self.n_hands) / np.sqrt(self.n_hands)
        mu2 = np.ones(self.n_hands) / np.sqrt(self.n_hands)
        
        # Diagonal matrices
        Lambda1 = np.diag(mu1)
        Lambda2 = np.diag(mu2)
        
        return mu1, mu2, Lambda1, Lambda2
    
    def _build_incompatibility_matrix(self) -> np.ndarray:
        """Build H× matrix indicating incompatible hand pairs"""
        
        H_incomp = np.zeros((self.n_hands, self.n_hands))
        
        for i, h1 in enumerate(self.hands):
            for j, h2 in enumerate(self.hands):
                if h1 == h2:
                    # Incompatible (can't both have same card)
                    H_incomp[i, j] = 1
        
        return H_incomp
    
    def _build_C_matrix(self, mu1, mu2, Lambda1, Lambda2, H_incomp) -> np.ndarray:
        """Build correction matrix C = μ₁μ₂ᵀ - Λ₁H×Λ₂"""
        
        # Outer product
        mu_outer = np.outer(mu1, mu2)
        
        # Incompatibility correction
        incomp_term = Lambda1 @ H_incomp @ Lambda2
        
        C = mu_outer - incomp_term
        
        return C
    
def test_kuhn_structure():
    """Test Kuhn structure extraction"""
    print("="*60)
    print("TESTING KUHN STRUCTURE EXTRACTION")
    print("="*60)
    
    extractor = KuhnPokerStructureExtractor()
    structure = extractor.extract_structure()
    
    print("\n" + "="*60)
    print("EXTRACTED STRUCTURE SUMMARY")
    print("="*60)
    print(f"Hands: {structure.hands1}")
    print(f"Win-lose matrix W:\n{structure.W}")
    print(f"P1 sequences: {structure.sequences1}")
    print(f"P2 sequences: {structure.sequences2}")
    print(f"F matrix shape: {structure.F.shape}")
    print(f"S matrix shape: {structure.S.shape}")
    
    return structure

if __name__ == '__main__':
    structure = test_kuhn_structure()