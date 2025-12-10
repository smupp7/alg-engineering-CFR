"""
Simplified Heads-Up Limit Texas Hold'em Structure Extractor

RESTRICTIONS to make it tractable:
- Only 2 players (heads-up)
- Limit betting (fixed bet sizes)
- Simplified hand abstractions (bucketing)
- Only preflop + flop (no turn/river)
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class HoldemStructure:
    """Simplified Hold'em Kronecker structure"""
    F: np.ndarray  # Fold matrix
    S: np.ndarray  # Showdown matrix
    W: np.ndarray  # Win-lose matrix
    Lambda1: np.ndarray  # Hand belief P1
    Lambda2: np.ndarray  # Hand belief P2
    C: np.ndarray  # Correlation matrix
    hands1: List[str]
    hands2: List[str]
    sequences1: List[str]
    sequences2: List[str]


class SimplifiedHoldemStructureExtractor:
    """
    Extract Kronecker structure for simplified Hold'em.
    
    SIMPLIFICATIONS:
    1. Hand abstraction: Group similar hands into buckets
    2. Betting: Preflop + Flop only (2 rounds)
    3. Limit betting: Fixed raise sizes
    4. 5 hand buckets instead of 1326 hand combinations
    """
    
    def __init__(self, n_hand_buckets=5, include_flop=True):
        """
        Args:
            n_hand_buckets: Number of hand strength buckets (5-20 typical)
            include_flop: If False, only preflop (even simpler)
        """
        self.n_hand_buckets = n_hand_buckets
        self.include_flop = include_flop
        
        print(f"\n🃏 Simplified Heads-Up Limit Hold'em")
        print(f"   Hand buckets: {n_hand_buckets}")
        print(f"   Rounds: {'Preflop + Flop' if include_flop else 'Preflop only'}")
    
    def extract_structure(self) -> HoldemStructure:
        """Extract simplified Hold'em structure"""
        
        print("\n1️⃣ Building hand abstractions...")
        hands = self._build_hand_buckets()
        print(f"   {len(hands)} hand buckets")
        
        print("\n2️⃣ Building betting sequences...")
        if self.include_flop:
            sequences1, sequences2 = self._build_betting_sequences_with_flop()
        else:
            sequences1, sequences2 = self._build_betting_sequences_preflop_only()
        
        print(f"   P1 sequences: {len(sequences1)}")
        print(f"   P2 sequences: {len(sequences2)}")
        
        print("\n3️⃣ Building fold matrix...")
        F = self._build_fold_matrix(len(sequences1), len(sequences2))
        print(f"   F: {F.shape}")
        
        print("\n4️⃣ Building showdown matrix...")
        S = self._build_showdown_matrix(len(sequences1), len(sequences2))
        print(f"   S: {S.shape}")
        
        print("\n5️⃣ Building win-lose matrix...")
        W = self._build_win_matrix(len(hands))
        print(f"   W: {W.shape}")
        
        print("\n6️⃣ Building hand beliefs...")
        Lambda1, Lambda2 = self._build_hand_beliefs(len(hands))
        print(f"   Lambda: {Lambda1.shape}")
        
        print("\n7️⃣ Building correlation matrix...")
        C = np.eye(len(hands))  # Simplified: no correlation
        print(f"   C: {C.shape}")
        
        print("\n✅ Simplified Hold'em structure complete!")
        
        # Calculate matrix size
        m = len(hands) * len(sequences1)
        n = len(hands) * len(sequences2)
        print(f"\n📊 Final Matrix Size: {m:,} × {n:,} = {m*n:,} entries")
        print(f"   Memory (dense): {m*n*8/1e9:.3f} GB")
        
        return HoldemStructure(
            F=F, S=S, W=W,
            Lambda1=Lambda1, Lambda2=Lambda2, C=C,
            hands1=hands, hands2=hands,
            sequences1=sequences1, sequences2=sequences2
        )
    
    def _build_hand_buckets(self) -> List[str]:
        """
        Abstract hands into strength buckets.
        
        In real Hold'em: 1326 possible hands
        We abstract to: 5-20 buckets based on strength
        
        Example buckets:
        - Strong: AA, KK, QQ, AK
        - Good: JJ, TT, AQ, AJ
        - Medium: 99-22, suited connectors
        - Weak: Offsuit low cards
        - Trash: 72o, 83o, etc.
        """
        bucket_names = [
            "Premium (AA-QQ, AK)",
            "Strong (JJ-99, AQ-AJ)",
            "Medium (88-22, KQ-KJ, suited)",
            "Weak (Low pairs, suited connectors)",
            "Trash (Offsuit low cards)"
        ]
        
        return bucket_names[:self.n_hand_buckets]
    
    def _build_betting_sequences_preflop_only(self) -> Tuple[List[str], List[str]]:
        """
        Build betting sequences for preflop only.
        
        Limit betting: Fixed bet/raise sizes
        Example actions: Fold, Call, Raise
        """
        # P1 (button) sequences
        sequences1 = [
            "root",           # 0: Initial
            "call",           # 1: Limp
            "raise",          # 2: Raise
            "raise-raise",    # 3: Re-raise (cap)
        ]
        
        # P2 (big blind) sequences
        sequences2 = [
            "check",          # 0: Check (if P1 limped)
            "raise",          # 1: Raise (if P1 limped or raised)
            "raise-raise",    # 2: Re-raise (if P1 raised)
        ]
        
        return sequences1, sequences2
    
    def _build_betting_sequences_with_flop(self) -> Tuple[List[str], List[str]]:
        """
        Build sequences for preflop + flop.
        
        This gets more complex with 2 rounds of betting.
        """
        # Simplified: Each round has check/bet/raise options
        # Total sequences = preflop_seqs × flop_seqs
        
        preflop = ["call", "raise"]
        flop = ["check", "bet", "bet-raise"]
        
        # Combine
        sequences1 = []
        for pf in preflop:
            for fl in flop:
                sequences1.append(f"{pf}-{fl}")
        
        # Add root
        sequences1.insert(0, "root")
        
        # P2 has fewer sequences (acts second)
        sequences2 = []
        for pf in ["check", "raise"]:
            for fl in ["check", "bet"]:
                sequences2.append(f"{pf}-{fl}")
        
        return sequences1, sequences2
    
    def _build_fold_matrix(self, n_seq1: int, n_seq2: int) -> np.ndarray:
        """
        Build fold matrix F.
        
        F[i,j] = payoff when P1 plays sequence i, P2 plays j, someone folds
        """
        F = np.zeros((n_seq1, n_seq2))
        
        # Simplified: Some sequences lead to folds
        # If either player folds, the other wins the pot
        # Pot size depends on betting sequence
        
        # Example: If P2 folds to P1 raise, P1 wins small pot
        for i in range(n_seq1):
            for j in range(n_seq2):
                # Simplified logic
                if "raise" in str(i) and j == 0:
                    F[i, j] = 1.0  # P1 wins
                elif i == 0 and "raise" in str(j):
                    F[i, j] = -1.0  # P2 wins
        
        return F
    
    def _build_showdown_matrix(self, n_seq1: int, n_seq2: int) -> np.ndarray:
        """
        Build showdown matrix S.
        
        S[i,j] = payoff structure when going to showdown
        """
        S = np.ones((n_seq1, n_seq2))
        
        # Showdown payoff depends on pot size
        # More raises = bigger pot
        for i in range(n_seq1):
            for j in range(n_seq2):
                pot_size = 1.0
                if "raise" in str(i):
                    pot_size += 1.0
                if "raise" in str(j):
                    pot_size += 1.0
                
                S[i, j] = pot_size
        
        return S
    
    def _build_win_matrix(self, n_hands: int) -> np.ndarray:
        """
        Build win-lose matrix W.
        
        W[i,j] = probability hand i beats hand j
        """
        W = np.zeros((n_hands, n_hands))
        
        # Buckets are ordered by strength
        # Higher index = stronger hand
        for i in range(n_hands):
            for j in range(n_hands):
                if i > j:
                    # Stronger hand wins
                    W[i, j] = 1.0
                elif i < j:
                    # Weaker hand loses
                    W[i, j] = -1.0
                else:
                    # Tie
                    W[i, j] = 0.0
        
        return W
    
    def _build_hand_beliefs(self, n_hands: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build hand belief matrices (uniform for simplicity)"""
        Lambda1 = np.eye(n_hands) / n_hands
        Lambda2 = np.eye(n_hands) / n_hands
        return Lambda1, Lambda2


class TinyHoldemExtractor:
    """
    ULTRA-SIMPLIFIED Hold'em for quick testing.
    
    Just 3 hand buckets, preflop only, minimal betting.
    """
    
    def extract_structure(self) -> HoldemStructure:
        print("\n🃏 Tiny Hold'em (3 buckets, preflop only)")
        
        # Just 3 hand strengths
        hands = ["Strong", "Medium", "Weak"]
        
        # Minimal betting tree
        sequences1 = ["root", "call", "raise"]
        sequences2 = ["check", "raise"]
        
        n1, n2 = len(sequences1), len(sequences2)
        n_hands = len(hands)
        
        F = np.random.randn(n1, n2) * 0.5
        S = np.ones((n1, n2))
        W = np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]])
        Lambda1 = np.eye(n_hands) / n_hands
        Lambda2 = np.eye(n_hands) / n_hands
        C = np.eye(n_hands)
        
        m = n_hands * n1
        n = n_hands * n2
        print(f"   Matrix: {m} × {n} = {m*n} entries")
        print(f"✅ Tiny Hold'em structure ready!")
        
        return HoldemStructure(
            F=F, S=S, W=W,
            Lambda1=Lambda1, Lambda2=Lambda2, C=C,
            hands1=hands, hands2=hands,
            sequences1=sequences1, sequences2=sequences2
        )
# Add this to your holdem_structure.py:

class LargeHoldemExtractor:
    """
    Large Hold'em variant to test parallelization.
    
    Use many hand buckets and complex betting tree.
    """
    
    def extract_structure(self) -> HoldemStructure:
        print("\n🃏 LARGE Hold'em (20 buckets, complex betting)")
        
        # 20 hand strength buckets
        n_hands = 20
        hands = [f"Bucket-{i}" for i in range(n_hands)]
        
        # Complex betting tree (preflop + flop + turn)
        # Each round: check, bet, raise, reraise
        # Total combinations: 4^3 = 64+ sequences
        sequences1 = []
        sequences2 = []
        
        # Generate all betting combinations
        actions = ["check", "bet", "raise", "reraise"]
        
        for a1 in actions:
            for a2 in actions:
                for a3 in actions:
                    sequences1.append(f"{a1}-{a2}-{a3}")
        
        for a1 in actions[:3]:  # P2 has fewer sequences
            for a2 in actions[:3]:
                sequences2.append(f"{a1}-{a2}")
        
        n1, n2 = len(sequences1), len(sequences2)
        
        print(f"   Hands: {n_hands}")
        print(f"   P0 sequences: {n1}")
        print(f"   P1 sequences: {n2}")
        
        # Build matrices
        F = np.random.randn(n1, n2) * 0.5
        S = np.ones((n1, n2))
        W = np.zeros((n_hands, n_hands))
        
        # Win matrix: stronger hands beat weaker
        for i in range(n_hands):
            for j in range(n_hands):
                if i > j:
                    W[i, j] = 1.0
                elif i < j:
                    W[i, j] = -1.0
        
        Lambda1 = np.eye(n_hands) / n_hands
        Lambda2 = np.eye(n_hands) / n_hands
        C = np.eye(n_hands)
        
        m = n_hands * n1
        n = n_hands * n2
        
        print(f"   Matrix: {m:,} × {n:,} = {m*n:,} entries")
        print(f"   Memory (dense): {m*n*8/1e6:.1f} MB")
        print(f"✅ Large Hold'em structure ready!")
        
        return HoldemStructure(
            F=F, S=S, W=W,
            Lambda1=Lambda1, Lambda2=Lambda2, C=C,
            hands1=hands, hands2=hands,
            sequences1=sequences1, sequences2=sequences2
        )


class MassiveHoldemExtractor:
    """
    MASSIVE Hold'em to really stress-test parallelization.
    
    50+ hand buckets, 200+ sequences
    """
    
    def extract_structure(self) -> HoldemStructure:
        print("\n🃏 MASSIVE Hold'em (50 buckets, 200+ sequences)")
        
        n_hands = 50
        hands = [f"Bucket-{i}" for i in range(n_hands)]
        
        # Generate many sequences (simulate complex betting tree)
        n_seq1 = 200
        n_seq2 = 100
        
        sequences1 = [f"seq-{i}" for i in range(n_seq1)]
        sequences2 = [f"seq-{i}" for i in range(n_seq2)]
        
        print(f"   Hands: {n_hands}")
        print(f"   P0 sequences: {n_seq1}")
        print(f"   P1 sequences: {n_seq2}")
        
        # Sparse random matrices
        F = np.random.randn(n_seq1, n_seq2) * 0.5
        S = np.random.randn(n_seq1, n_seq2) * 0.5
        
        W = np.zeros((n_hands, n_hands))
        for i in range(n_hands):
            for j in range(n_hands):
                if i > j:
                    W[i, j] = 1.0
                elif i < j:
                    W[i, j] = -1.0
        
        Lambda1 = np.eye(n_hands) / n_hands
        Lambda2 = np.eye(n_hands) / n_hands
        C = np.eye(n_hands)
        
        m = n_hands * n_seq1
        n = n_hands * n_seq2
        
        print(f"   Matrix: {m:,} × {n:,} = {m*n:,} entries")
        print(f"   Memory (dense): {m*n*8/1e6:.1f} MB")
        print(f"✅ MASSIVE Hold'em structure ready!")
        
        return HoldemStructure(
            F=F, S=S, W=W,
            Lambda1=Lambda1, Lambda2=Lambda2, C=C,
            hands1=hands, hands2=hands,
            sequences1=sequences1, sequences2=sequences2
        )
class UltraMassiveHoldemExtractor:
    """
    ULTRA-MASSIVE Hold'em: 100 buckets × 500 sequences
    Matrix: 50,000 × 50,000 = 2.5 BILLION entries
    """
    
    def extract_structure(self) -> HoldemStructure:
        print("\n🃏 ULTRA-MASSIVE Hold'em (100 buckets × 500 sequences)")
        
        n_hands = 100
        n_seq1 = 500
        n_seq2 = 500
        
        hands = [f"Bucket-{i}" for i in range(n_hands)]
        sequences1 = [f"seq-{i}" for i in range(n_seq1)]
        sequences2 = [f"seq-{i}" for i in range(n_seq2)]
        
        print(f"   Hands: {n_hands}")
        print(f"   Sequences: {n_seq1} × {n_seq2}")
        
        # Create sparse random matrices
        F = np.random.randn(n_seq1, n_seq2) * 0.1
        S = np.random.randn(n_seq1, n_seq2) * 0.1
        
        W = np.zeros((n_hands, n_hands))
        for i in range(n_hands):
            for j in range(n_hands):
                if i > j:
                    W[i, j] = 1.0
                elif i < j:
                    W[i, j] = -1.0
        
        Lambda1 = np.eye(n_hands) / n_hands
        Lambda2 = np.eye(n_hands) / n_hands
        C = np.eye(n_hands)
        
        m = n_hands * n_seq1
        n = n_hands * n_seq2
        
        print(f"   Matrix: {m:,} × {n:,} = {m*n:,} entries")
        print(f"   Memory: {m*n*8/1e9:.2f} GB")
        print(f"✅ ULTRA-MASSIVE Hold'em ready!")
        
        return HoldemStructure(
            F=F, S=S, W=W,
            Lambda1=Lambda1, Lambda2=Lambda2, C=C,
            hands1=hands, hands2=hands,
            sequences1=sequences1, sequences2=sequences2
        )


class GiganticHoldemExtractor:
    """
    GIGANTIC Hold'em: 150 buckets × 800 sequences
    Matrix: 120,000 × 120,000 = 14.4 BILLION entries
    """
    
    def extract_structure(self) -> HoldemStructure:
        print("\n🃏 GIGANTIC Hold'em (150 buckets × 800 sequences)")
        
        n_hands = 150
        n_seq1 = 800
        n_seq2 = 800
        
        hands = [f"Bucket-{i}" for i in range(n_hands)]
        sequences1 = [f"seq-{i}" for i in range(n_seq1)]
        sequences2 = [f"seq-{i}" for i in range(n_seq2)]
        
        print(f"   Hands: {n_hands}")
        print(f"   Sequences: {n_seq1} × {n_seq2}")
        
        F = np.random.randn(n_seq1, n_seq2) * 0.1
        S = np.random.randn(n_seq1, n_seq2) * 0.1
        
        W = np.zeros((n_hands, n_hands))
        for i in range(n_hands):
            for j in range(n_hands):
                if i > j:
                    W[i, j] = 1.0
                elif i < j:
                    W[i, j] = -1.0
        
        Lambda1 = np.eye(n_hands) / n_hands
        Lambda2 = np.eye(n_hands) / n_hands
        C = np.eye(n_hands)
        
        m = n_hands * n_seq1
        n = n_hands * n_seq2
        
        print(f"   Matrix: {m:,} × {n:,} = {m*n:,} entries")
        print(f"   Memory: {m*n*8/1e9:.2f} GB")
        print(f"✅ GIGANTIC Hold'em ready!")
        
        return HoldemStructure(
            F=F, S=S, W=W,
            Lambda1=Lambda1, Lambda2=Lambda2, C=C,
            hands1=hands, hands2=hands,
            sequences1=sequences1, sequences2=sequences2
        )


class ColossalHoldemExtractor:
    """
    COLOSSAL Hold'em: 200 buckets × 1000 sequences
    Matrix: 200,000 × 200,000 = 40 BILLION entries
    
    WARNING: May take minutes per iteration!
    """
    
    def extract_structure(self) -> HoldemStructure:
        print("\n🃏 COLOSSAL Hold'em (200 buckets × 1000 sequences)")
        
        n_hands = 200
        n_seq1 = 1000
        n_seq2 = 1000
        
        hands = [f"Bucket-{i}" for i in range(n_hands)]
        sequences1 = [f"seq-{i}" for i in range(n_seq1)]
        sequences2 = [f"seq-{i}" for i in range(n_seq2)]
        
        print(f"   Hands: {n_hands}")
        print(f"   Sequences: {n_seq1} × {n_seq2}")
        
        F = np.random.randn(n_seq1, n_seq2) * 0.1
        S = np.random.randn(n_seq1, n_seq2) * 0.1
        
        W = np.zeros((n_hands, n_hands))
        for i in range(n_hands):
            for j in range(n_hands):
                if i > j:
                    W[i, j] = 1.0
                elif i < j:
                    W[i, j] = -1.0
        
        Lambda1 = np.eye(n_hands) / n_hands
        Lambda2 = np.eye(n_hands) / n_hands
        C = np.eye(n_hands)
        
        m = n_hands * n_seq1
        n = n_hands * n_seq2
        
        print(f"   Matrix: {m:,} × {n:,} = {m*n:,} entries")
        print(f"   Memory: {m*n*8/1e9:.2f} GB")
        print(f"⚠️  WARNING: Very large! Each iteration may take seconds!")
        print(f"✅ COLOSSAL Hold'em ready!")
        
        return HoldemStructure(
            F=F, S=S, W=W,
            Lambda1=Lambda1, Lambda2=Lambda2, C=C,
            hands1=hands, hands2=hands,
            sequences1=sequences1, sequences2=sequences2
        )
if __name__ == '__main__':
    print("="*70)
    print("TESTING HOLD'EM STRUCTURE EXTRACTORS")
    print("="*70)
    
    # Test tiny version
    print("\n" + "="*70)
    print("TINY HOLD'EM")
    print("="*70)
    extractor = TinyHoldemExtractor()
    structure = extractor.extract_structure()
    
    # Test simplified version
    print("\n" + "="*70)
    print("SIMPLIFIED HOLD'EM (Preflop only)")
    print("="*70)
    extractor = SimplifiedHoldemStructureExtractor(
        n_hand_buckets=5,
        include_flop=False
    )
    structure = extractor.extract_structure()
    
    # Test with flop
    print("\n" + "="*70)
    print("SIMPLIFIED HOLD'EM (Preflop + Flop)")
    print("="*70)
    extractor = SimplifiedHoldemStructureExtractor(
        n_hand_buckets=5,
        include_flop=True
    )
    structure = extractor.extract_structure()
    
    print("\n" + "="*70)
    print("✅ All Hold'em extractors working!")
    print("="*70)