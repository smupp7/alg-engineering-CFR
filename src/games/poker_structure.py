"""
Extract Kronecker product structure from poker games
Following Farina et al. 2020
"""
import numpy as np
import pyspiel
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class PokerStructure:
    """
    Kronecker structure of poker game
    
    A = C ⊗ F + (Λ₁WΛ₂) ⊗ S
    
    Where:
    - F: Fold payoff matrix (betting skeleton)
    - S: Showdown payoff matrix (betting skeleton)
    - W: Win-lose matrix (hand matchups)
    - Λ₁, Λ₂: Diagonal matrices of hand probabilities
    - C: Incompatibility correction matrix
    """
    F: np.ndarray  # Fold payoffs
    S: np.ndarray  # Showdown payoffs
    W: np.ndarray  # Win-lose matrix
    Lambda1: np.ndarray  # P1 hand probabilities (diagonal)
    Lambda2: np.ndarray  # P2 hand probabilities (diagonal)
    mu1: np.ndarray  # P1 hand belief vector
    mu2: np.ndarray  # P2 hand belief vector
    C: np.ndarray  # Incompatibility matrix
    H_incomp: np.ndarray  # Incompatibility indicator
    
    hands1: List[str]  # P1 hands
    hands2: List[str]  # P2 hands
    sequences1: List[str]  # P1 action sequences
    sequences2: List[str]  # P2 action sequences


