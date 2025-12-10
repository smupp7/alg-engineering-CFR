import numpy as np
from typing import List

from numba.cuda import local
from open_spiel.python.algorithms.exploitability import best_response

from game_tree import GameStateType, GameState, Sequence, InfoSet, InfoSetVec, SeqVec, GameEnv
import pyspiel

def normalize(vec: np.array):
    vec_positive = vec * (vec > 0)
    vec_sum = np.sum(vec_positive)
    if vec_sum > 1e-9:
        return vec_positive / vec_sum
    else:
        return np.ones_like(vec) / len(vec)

def compute_utility_vector(game: GameEnv, player: int, policies: List[SeqVec]) -> SeqVec:
    env_trans_prob = game.env_trans_prob
    utilities = SeqVec(game, player)

    def traverse(game_state: GameState, ext_reach: float) -> float:
        _ev = 0.
        infoset = game_state.infoset
        if game_state.type == GameStateType.Chance:
            for i, action in enumerate(infoset.actions):
                _ev += traverse(game_state.transitions[i], ext_reach * env_trans_prob[infoset][i])
        elif game_state.type == GameStateType.Player:
            if game_state.player == player:
                for i, action in enumerate(infoset.actions):
                    utilities[infoset][i] += traverse(game_state.transitions[i], ext_reach)
            else:
                for i, action in enumerate(infoset.actions):
                    _ev += traverse(game_state.transitions[i], ext_reach * policies[game_state.player][infoset][i])
        elif game_state.type == GameStateType.Terminal:
            _ev = ext_reach * game_state.returns[player]
        return _ev

    traverse(game.history_root, ext_reach = 1.)
    return utilities

def compute_counterfactual_value(game: GameEnv, player: int, policies: List[SeqVec]) -> SeqVec:
    utilities = compute_utility_vector(game, player, policies)

    cf_value = SeqVec(game, player)
    def traverse(seq: Sequence) -> (float, float):
        _ev = 0.
        for infoset in seq.nxt_infosets:
            for i, action in enumerate(infoset.actions):
                cf_value[infoset][i] = traverse(infoset.nxt_sequences[i]) + utilities[infoset][i]
                _ev += cf_value[infoset][i] * policies[player][infoset][i]
        return _ev

    traverse(game.sequences_root[player])
    return cf_value

def get_uniform_policy(game: GameEnv, player: int) -> SeqVec:
    return SeqVec(game, player).map(normalize)

def get_reach_probability(game: GameEnv, player: int, policy: SeqVec) -> SeqVec:
    reach = SeqVec(game, player)

    def traverse(seq: Sequence, agent_reach: float):
        for infoset in seq.nxt_infosets:
            for i, action in enumerate(infoset.actions):
                reach[infoset][i] = agent_reach * policy[infoset][i]
                traverse(infoset.nxt_sequences[i], reach[infoset][i])

    traverse(game.sequences_root[player], agent_reach = 1.)
    return reach

def compute_policy_from_reach(game: GameEnv, player: int, reach: SeqVec) -> SeqVec:
    return reach.map(normalize)

def compute_best_response(game: GameEnv, player: int, policies: List[SeqVec]) -> SeqVec:
    utilities = compute_utility_vector(game, player, policies)
    best_response = SeqVec(game, player)

    def traverse(seq: Sequence) -> float:
        max_ev = 0.
        for infoset in seq.nxt_infosets:
            local_max_ev = -np.inf
            best_i = None
            for i, action in enumerate(infoset.actions):
                max_ev_nxt = traverse(infoset.nxt_sequences[i]) + utilities[infoset][i]
                if max_ev_nxt > local_max_ev:
                    local_max_ev = max_ev_nxt
                    best_i = i
            max_ev += local_max_ev
            best_response[infoset][best_i] = 1.
        return max_ev

    traverse(game.sequences_root[player])
    return best_response

def compute_expected_value(game: GameEnv, player: int, policies: List[SeqVec]) -> float:
    utilities = compute_utility_vector(game, player, policies)
    ev = get_reach_probability(game, player, policies[player]).dot(utilities)
    return ev

def compute_exploitability(game: GameEnv, player: int, policies: List[SeqVec]) -> float:
    utilities = compute_utility_vector(game, player, policies)
    best_response = compute_best_response(game, player, policies)
    max_ev = get_reach_probability(game, player, best_response).dot(utilities)
    ev = get_reach_probability(game, player, policies[player]).dot(utilities)
    return max_ev - ev

def nash_conv(game: GameEnv, policies: List[SeqVec]) -> (float, List[float]):
    gaps = [compute_exploitability(game, player, policies) for player in range(game.num_players)]
    return np.sum(gaps), gaps

class TabularPolicy(pyspiel.Policy):
    def __init__(self, game: GameEnv, policies: List[SeqVec]):
        self.game = game
        self.policies = policies
        super().__init__()

    def action_probabilities(self, state, player_id = None):
        player = state.current_player()
        infoset_label = state.information_state_string()
        infoset = self.game.infoset_map[player][infoset_label]

        action_probs = normalize(self.policies[player][infoset])

        profiles = {}
        for i, action in enumerate(infoset.actions):
            profiles[action] = action_probs[i]
        return profiles