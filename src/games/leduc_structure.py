"""
Extract Kronecker product structure from Leduc poker
Leduc: 6 cards (J, J, Q, Q, K, K), 2 betting rounds
"""
import numpy as np
import pyspiel
from typing import List, Tuple
from games.kuhn_structure import PokerStructure


class LeducPokerStructureExtractor:
    """
    Extract Kronecker structure from Leduc poker
    
    Simplified: Extract structure for FIRST ROUND only
    (Full Leduc has 2 rounds + public card, would take longer)
    """
    
    def __init__(self):
        self.game = pyspiel.load_game("leduc_poker")
        
        # Leduc has J, Q, K (2 of each suit, but suits equivalent)
        self.hands = ['J', 'Q', 'K']
        self.n_hands = 3
        
        print("🃏 Extracting Leduc Poker Kronecker Structure")
        print("   (Simplified: first round structure)")
    
    def extract_structure(self) -> PokerStructure:
        """Extract Kronecker structure"""
        
        print("\n1️⃣ Building betting skeleton...")
        F, S, sequences1, sequences2 = self._build_betting_skeleton()
        
        print(f"   Fold matrix F: {F.shape}")
        print(f"   Showdown matrix S: {S.shape}")
        print(f"   P1 sequences: {len(sequences1)}")
        print(f"   P2 sequences: {len(sequences2)}")
        
        print("\n2️⃣ Building win-lose matrix...")
        W = self._build_win_lose_matrix()
        print(f"   Win-lose matrix W: {W.shape}")
        
        print("\n3️⃣ Building hand beliefs...")
        mu1, mu2, Lambda1, Lambda2 = self._build_hand_beliefs()
        
        print("\n4️⃣ Building incompatibility matrix...")
        H_incomp = self._build_incompatibility_matrix()
        C = self._build_C_matrix(mu1, mu2, Lambda1, Lambda2, H_incomp)
        
        structure = PokerStructure(
            F=F, S=S, W=W,
            Lambda1=Lambda1, Lambda2=Lambda2,
            mu1=mu1, mu2=mu2,
            C=C, H_incomp=H_incomp,
            hands1=self.hands, hands2=self.hands,
            sequences1=sequences1, sequences2=sequences2
        )
        
        print("\n✅ Leduc structure extracted!")
        return structure
    
    def _build_betting_skeleton(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Build betting skeleton for Leduc
        
        Leduc has more complex betting:
        - Raise amounts can vary
        - 2 rounds of betting
        
        For time: we'll extract actual sequences from game tree
        """
        sequences1 = []
        sequences2 = []
        terminal_states = []
        
        # Extract from actual game
        state = self.game.new_initial_state()
        self._enumerate_sequences(state, [], [], sequences1, sequences2, terminal_states, max_depth=15)
        
        # Remove duplicates
        sequences1 = sorted(list(set(sequences1)))
        sequences2 = sorted(list(set(sequences2)))
        
        print(f"   Found {len(sequences1)} P1 sequences, {len(sequences2)} P2 sequences")
        
        n_seq1 = len(sequences1)
        n_seq2 = len(sequences2)
        
        # Build F and S
        F = np.zeros((n_seq1, n_seq2))
        S = np.zeros((n_seq1, n_seq2))
        
        seq1_to_idx = {seq: i for i, seq in enumerate(sequences1)}
        seq2_to_idx = {seq: i for i, seq in enumerate(sequences2)}
        
        for seq1, seq2, is_showdown, payoff in terminal_states:
            i = seq1_to_idx.get(seq1, -1)
            j = seq2_to_idx.get(seq2, -1)
            
            if i >= 0 and j >= 0:
                if is_showdown:
                    S[i, j] = payoff
                else:
                    F[i, j] = payoff
        
        return F, S, sequences1, sequences2
    
    def _enumerate_sequences(self, state, seq1, seq2, sequences1, sequences2, terminals, depth=0, max_depth=20):
        """Enumerate sequences from game tree"""
        
        if depth > max_depth:
            return
        
        if state.is_terminal():
            seq1_str = ''.join(seq1)
            seq2_str = ''.join(seq2)
            
            sequences1.append(seq1_str)
            sequences2.append(seq2_str)
            
            # Simplified payoff extraction
            is_showdown = 'f' not in seq1_str and 'f' not in seq2_str
            payoff = 1.0  # Placeholder
            
            terminals.append((seq1_str, seq2_str, is_showdown, payoff))
            return
        
        if state.is_chance_node():
            for action, prob in state.chance_outcomes():
                self._enumerate_sequences(
                    state.child(action), seq1.copy(), seq2.copy(),
                    sequences1, sequences2, terminals, depth + 1, max_depth
                )
            return
        
        current_player = state.current_player()
        
        # Limit actions to keep structure manageable
        legal_actions = state.legal_actions()
        if len(legal_actions) > 4:  # Cap action branching
            legal_actions = legal_actions[:4]
        
        for action in legal_actions:
            # Map action to simple string
            action_str = self._action_to_string(action)
            
            new_seq1 = seq1.copy()
            new_seq2 = seq2.copy()
            
            if current_player == 0:
                new_seq1.append(action_str)
            else:
                new_seq2.append(action_str)
            
            self._enumerate_sequences(
                state.child(action), new_seq1, new_seq2,
                sequences1, sequences2, terminals, depth + 1, max_depth
            )
    
    def _action_to_string(self, action: int) -> str:
        """Map action to string"""
        # Leduc actions: fold=0, call=1, raise=2
        if action == 0:
            return 'f'  # fold
        elif action == 1:
            return 'c'  # call
        else:
            return 'r'  # raise
    
    def _build_win_lose_matrix(self) -> np.ndarray:
        """Win-lose matrix (same as Kuhn: J < Q < K)"""
        W = np.zeros((self.n_hands, self.n_hands))
        
        hand_strength = {'J': 0, 'Q': 1, 'K': 2}
        
        for i, h1 in enumerate(self.hands):
            for j, h2 in enumerate(self.hands):
                if h1 == h2:
                    W[i, j] = 0
                elif hand_strength[h1] > hand_strength[h2]:
                    W[i, j] = 1
                else:
                    W[i, j] = -1
        
        return W
    
    def _build_hand_beliefs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Uniform hand beliefs"""
        mu1 = np.ones(self.n_hands) / np.sqrt(self.n_hands)
        mu2 = np.ones(self.n_hands) / np.sqrt(self.n_hands)
        Lambda1 = np.diag(mu1)
        Lambda2 = np.diag(mu2)
        return mu1, mu2, Lambda1, Lambda2
    
    def _build_incompatibility_matrix(self) -> np.ndarray:
        """Incompatibility matrix"""
        H_incomp = np.zeros((self.n_hands, self.n_hands))
        for i in range(self.n_hands):
            H_incomp[i, i] = 1
        return H_incomp
    
    def _build_C_matrix(self, mu1, mu2, Lambda1, Lambda2, H_incomp) -> np.ndarray:
        """Correction matrix"""
        mu_outer = np.outer(mu1, mu2)
        incomp_term = Lambda1 @ H_incomp @ Lambda2
        return mu_outer - incomp_term


def test_leduc_structure():
    """Test Leduc structure extraction"""
    print("="*60)
    print("TESTING LEDUC STRUCTURE EXTRACTION")
    print("="*60)
    
    extractor = LeducPokerStructureExtractor()
    structure = extractor.extract_structure()
    
    print("\n" + "="*60)
    print("EXTRACTED STRUCTURE SUMMARY")
    print("="*60)
    print(f"Hands: {structure.hands1}")
    print(f"Win-lose matrix W:\n{structure.W}")
    print(f"F matrix shape: {structure.F.shape}")
    print(f"S matrix shape: {structure.S.shape}")
    print(f"Total matrix size: {structure.F.shape[0] * 3} × {structure.F.shape[1] * 3}")
    
    return structure


if __name__ == '__main__':
    structure = test_leduc_structure()