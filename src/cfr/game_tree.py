import numpy as np
from enum import Enum
import pyspiel
from typing import List, Callable, Any

class GameStateType(Enum):
    Chance = 0
    Player = 1
    Terminal = 2

class GameState:
    def __init__(self):
        self.type = GameStateType.Chance
        self.player = None
        self.transitions = None
        self.transition_probs = None
        self.infoset = None
        self.game = None
        self.returns = []

class InfoSet:
    def __init__(self):
        self.player = None
        self.actions = None
        self.nxt_sequences = None
        self.infoset_idx = None
        self.sequence_idx = None
        self.game = None
        self.label = None

    def __repr__(self) -> str:
        return str(self.player) + " " + self.label

class Sequence:
    def __init__(self):
        self.player = None
        self.nxt_infosets = []

class InfoSetVec:
    def __init__(self, game: "GameEnv", player: int, *, default_value = 0.,
                 array: np.ndarray = None, dtype: np.dtype = np.float32):
        self.game = game
        self.player = player
        if array is not None:
            self.array = array.copy()
        else:
            size = game.infoset_nums[player]
            self.array = np.ones(size, dtype = dtype) * default_value

    def _array_operator_sanity_check(self, infoset: InfoSet) -> None:
        if self.game != infoset.game:
            raise ValueError("Game environment must match")
        if self.player != infoset.player:
            raise ValueError("Player must match")

    def __getitem__(self, infoset: InfoSet):
        self._array_operator_sanity_check(infoset)
        return self.array[infoset.infoset_idx]

    def __setitem__(self, infoset: InfoSet, value: np.float32) -> None:
        self._array_operator_sanity_check(infoset)
        self.array[infoset.infoset_idx] = value

    def __repr__(self) -> str:
        result = ""
        for infoset in self.game.infoset_list[self.player]:
            result += f"{infoset.label}, {self[infoset]} \n"
        return result

    def _math_opt_type_cast(self, other) -> "InfoSetVec":
        if isinstance(other, InfoSetVec):
            if self.game != other.game:
                raise ValueError("Game environment must match")
            if self.player != other.player:
                raise ValueError("Player must match")
            return other
        else:
            return InfoSetVec(self.game, self.player, default_value = other)

    def __neg__(self) -> "InfoSetVec":
        return InfoSetVec(self.game, self.player, array = -self.array)

    def __add__(self, other) -> "InfoSetVec":
        other = self._math_opt_type_cast(other)
        return InfoSetVec(self.game, self.player, array = self.array + other.array)

    def __radd__(self, other) -> "InfoSetVec":
        return self.__add__(other)

    def __sub__(self, other) -> "InfoSetVec":
        other = self._math_opt_type_cast(other)
        return InfoSetVec(self.game, self.player, array = self.array - other.array)

    def __rsub__(self, other) -> "InfoSetVec":
        other = self._math_opt_type_cast(other)
        return InfoSetVec(self.game, self.player, array = other.array - self.array)

    def __mul__(self, other) -> "InfoSetVec":
        other = self._math_opt_type_cast(other)
        return InfoSetVec(self.game, self.player, array = self.array * other.array)

    def __rmul__(self, other) -> "InfoSetVec":
        return self.__mul__(other)

    def __truediv__(self, other) -> "InfoSetVec":
        other = self._math_opt_type_cast(other)
        return InfoSetVec(self.game, self.player, array = self.array / other.array)

    def copy(self) -> "InfoSetVec":
        return InfoSetVec(self.game, self.player, array = self.array.copy())

    def seqvec(self) -> "SeqVec":
        vec = SeqVec(self.game, self.player, dtype = self.array.dtype)
        for infoset in self.game.infoset_list[self.player]:
            vec[infoset] = self[infoset]
        return vec

class SeqVec:
    def __init__(self, game: "GameEnv", player: int, *, default_value = 0.,
                 array: np.ndarray = None, dtype: np.dtype = np.float32):
        self.game = game
        self.player = player
        if array is not None:
            self.array = array.copy()
        else:
            size = game.sequence_nums[player]
            self.array = np.ones(size, dtype = dtype) * default_value

    def _array_operator_sanity_check(self, infoset: InfoSet) -> None:
        if self.game != infoset.game:
            raise ValueError("Game environment must match")
        if self.player != infoset.player:
            raise ValueError("Player must match")

    def __getitem__(self, infoset: InfoSet) -> np.ndarray:
        self._array_operator_sanity_check(infoset)
        return self.array[infoset.sequence_idx: infoset.sequence_idx + len(infoset.actions)]

    def __setitem__(self, infoset: InfoSet, value: np.ndarray) -> None:
        self._array_operator_sanity_check(infoset)
        self.array[infoset.sequence_idx: infoset.sequence_idx + len(infoset.actions)] = value

    def __repr__(self) -> str:
        result = ""
        for infoset in self.game.infoset_list[self.player]:
            result += f"{infoset.label}, {self[infoset]} \n"
        return result

    def _math_opt_type_cast(self, other) -> "SeqVec":
        if isinstance(other, SeqVec):
            if self.game != other.game:
                raise ValueError("Game environment must match")
            if self.player != other.player:
                raise ValueError("Player must match")
            return other
        else:
            return SeqVec(self.game, self.player, default_value = other)

    def __neg__(self) -> "SeqVec":
        return SeqVec(self.game, self.player, array = -self.array)

    def __add__(self, other) -> "SeqVec":
        other = self._math_opt_type_cast(other)
        return SeqVec(self.game, self.player, array = self.array + other.array)

    def __radd__(self, other) -> "SeqVec":
        return self.__add__(other)

    def __sub__(self, other) -> "SeqVec":
        other = self._math_opt_type_cast(other)
        return SeqVec(self.game, self.player, array = self.array - other.array)

    def __rsub__(self, other) -> "SeqVec":
        other = self._math_opt_type_cast(other)
        return SeqVec(self.game, self.player, array = other.array - self.array)

    def __mul__(self, other) -> "SeqVec":
        other = self._math_opt_type_cast(other)
        return SeqVec(self.game, self.player, array = self.array * other.array)

    def __rmul__(self, other) -> "SeqVec":
        return self.__mul__(other)

    def __truediv__(self, other) -> "SeqVec":
        other = self._math_opt_type_cast(other)
        return SeqVec(self.game, self.player, array = self.array / other.array)

    def copy(self) -> "SeqVec":
        return SeqVec(self.game, self.player, array = self.array.copy())

    def reduce(self, func: Callable[[np.ndarray], Any]) -> InfoSetVec:
        result = InfoSetVec(self.game, self.player, default_value = func(self[self.game.infoset_list[self.player][0]]))
        for infoset in self.game.infoset_list[self.player]:
            result[infoset] = func(self[infoset])
        return result

    def map(self, func: Callable[[np.ndarray], np.ndarray]) -> "SeqVec":
        result = SeqVec(self.game, self.player, default_value = func(self[self.game.infoset_list[self.player][0]])[0])
        for infoset in self.game.infoset_list[self.player]:
            result[infoset] = func(self[infoset])
        return result

    def dot(self, other) -> float:
        other = self._math_opt_type_cast(other)
        return np.dot(self.array, other.array)

class GameEnv:
    def __init__(self):
        self.num_players = None
        self.infoset_map = None
        self.infoset_list = None
        self.infoset_nums = None
        self.sequence_nums = None

        self.sequences_root = None
        self.history_root = None
        self.env_trans_prob = None

    def load_open_spiel_game(self, env: pyspiel.Game):
        self.num_players = env.num_players()
        self.infoset_map = [{} for _ in range(self.num_players + 1)]
        self.infoset_list = [[] for _ in range(self.num_players + 1)]
        self.infoset_nums = [0 for _ in range(self.num_players + 1)]
        self.sequence_nums = [0 for _ in range(self.num_players + 1)]

        self.sequences_root = [Sequence() for _ in range(self.num_players)]

        self._transit_prob_tmp = []
        self.history_root = self.open_spiel_recursive_build(env.new_initial_state(), self.sequences_root)
        self.env_trans_prob = SeqVec(self, -1, array = np.array(self._transit_prob_tmp))

    def open_spiel_recursive_build(self, state: pyspiel.State, sequences: List[Sequence]) -> GameState:
        # print(state, sequences)
        game_state = GameState()
        game_state.game = self

        if state.is_chance_node():
            game_state.type = GameStateType.Chance
            game_state.player = -1
            game_state.transitions = [] * len(state.chance_outcomes())

            infoset_label = state.history_str()
            game_state.infoset = infoset = self.infoset_map[-1][infoset_label] = InfoSet()
            self.infoset_list[-1].append(infoset)
            infoset.player = -1
            infoset.infoset_idx = self.infoset_nums[-1]
            infoset.sequence_idx = self.sequence_nums[-1]
            infoset.label = infoset_label
            infoset.game = self

            infoset.actions = []
            for (action, prob) in state.chance_outcomes():
                infoset.actions.append(action)
                self._transit_prob_tmp.append(prob)

            self.infoset_nums[-1] += 1
            self.sequence_nums[-1] += len(state.chance_outcomes())

            game_state.transitions = []
            for (action, prob) in state.chance_outcomes():
                state_nxt = state.clone()
                state_nxt.apply_action(action)
                game_state.transitions.append(self.open_spiel_recursive_build(state_nxt, sequences))

        elif state.is_player_node():
            game_state.type = GameStateType.Player
            game_state.player = player = state.current_player()
            infoset_label = state.information_state_string()

            if infoset_label not in self.infoset_map[player]:
                game_state.infoset = infoset = self.infoset_map[player][infoset_label] = InfoSet()
                self.infoset_list[player].append(infoset)
                infoset.player = player
                infoset.actions = state.legal_actions()
                infoset.infoset_idx = self.infoset_nums[player]
                infoset.sequence_idx = self.sequence_nums[player]
                infoset.label = infoset_label
                infoset.game = self

                self.infoset_nums[player] += 1
                self.sequence_nums[player] += len(infoset.actions)

                sequences[player].nxt_infosets.append(infoset)
                infoset.nxt_sequences = []
                for action in infoset.actions:
                    sequence = Sequence()
                    sequence.player = player
                    infoset.nxt_sequences.append(sequence)

            else:
                game_state.infoset = infoset = self.infoset_map[player][infoset_label]

            game_state.transitions = []
            for i, action in enumerate(infoset.actions):
                state_nxt = state.clone()
                state_nxt.apply_action(action)
                sequences_nxt = sequences.copy()
                sequences_nxt[player] = infoset.nxt_sequences[i]
                game_state.transitions.append(self.open_spiel_recursive_build(state_nxt, sequences_nxt))

        elif state.is_terminal():
            game_state.type = GameStateType.Terminal
            game_state.returns = state.returns()

        return game_state

    def load_libgg_game(self, path_to_file: str, num_players = 2):
        self.num_players = num_players
        self.infoset_map = [{} for _ in range(self.num_players + 1)]
        self.infoset_list = [[] for _ in range(self.num_players + 1)]
        self.infoset_nums = [0 for _ in range(self.num_players + 1)]
        self.sequence_nums = [0 for _ in range(self.num_players + 1)]

        self.sequences_root = [Sequence() for _ in range(self.num_players)]

        self._game_states = {}
        self._infoset_labels = {}

        with open(path_to_file, 'r') as f:
            for line in f.readlines():
                tokens = line.split()
                if tokens[0] == "node":
                    self._game_states[tokens[1]] = tokens
                elif tokens[0] == "infoset":
                    for label in tokens[3:]:
                        self._infoset_labels[label] = tokens[1]

        self._transit_prob_tmp = []
        self.history_root = self.libgg_recursive_build("/", self.sequences_root)
        self.env_trans_prob = SeqVec(self, -1, array = np.array(self._transit_prob_tmp))

    def libgg_recursive_build(self, state: str, sequences: List[Sequence]) -> GameState:
        # print(state, sequences)
        game_state = GameState()
        game_state.game = self

        tokens = self._game_states[state]
        if tokens[2] == "chance":
            game_state.type = GameStateType.Chance
            game_state.player = -1
            game_state.transitions = [] * len(tokens[4:])

            infoset_label = state
            game_state.infoset = infoset = self.infoset_map[-1][infoset_label] = InfoSet()
            self.infoset_list[-1].append(infoset)
            infoset.player = -1
            infoset.infoset_idx = self.infoset_nums[-1]
            infoset.sequence_idx = self.sequence_nums[-1]
            infoset.label = infoset_label
            infoset.game = self

            infoset.actions = []
            for action_prob in tokens[4:]:
                action, prob = action_prob.split('=')
                prob = float(prob)
                infoset.actions.append(action)
                self._transit_prob_tmp.append(prob)

            self.infoset_nums[-1] += 1
            self.sequence_nums[-1] += len(tokens[4:])

            game_state.transitions = []
            for action_prob in tokens[4:]:
                action, prob = action_prob.split('=')
                state_nxt = f"{state}C:{action}/"
                game_state.transitions.append(self.libgg_recursive_build(state_nxt, sequences))

        elif tokens[2] == "player":
            game_state.type = GameStateType.Player
            game_state.player = player = int(tokens[3]) - 1
            infoset_label = self._infoset_labels[state]

            if infoset_label not in self.infoset_map[player]:
                game_state.infoset = infoset = self.infoset_map[player][infoset_label] = InfoSet()
                self.infoset_list[player].append(infoset)
                infoset.player = player
                infoset.actions = tokens[5:]
                infoset.infoset_idx = self.infoset_nums[player]
                infoset.sequence_idx = self.sequence_nums[player]
                infoset.label = infoset_label
                infoset.game = self

                self.infoset_nums[player] += 1
                self.sequence_nums[player] += len(infoset.actions)

                sequences[player].nxt_infosets.append(infoset)
                infoset.nxt_sequences = []
                for action in infoset.actions:
                    sequence = Sequence()
                    sequence.player = player
                    infoset.nxt_sequences.append(sequence)

            else:
                game_state.infoset = infoset = self.infoset_map[player][infoset_label]

            game_state.transitions = []
            for i, action in enumerate(infoset.actions):
                state_nxt = f"{state}P{player + 1}:{action}/"
                sequences_nxt = sequences.copy()
                sequences_nxt[player] = infoset.nxt_sequences[i]
                game_state.transitions.append(self.libgg_recursive_build(state_nxt, sequences_nxt))

        elif tokens[2] == "terminal":
            game_state.type = GameStateType.Terminal
            game_state.returns = [0. for _ in tokens[4:]]
            for player_payoff in tokens[4:]:
                player = int(player_payoff.split("=")[0]) - 1
                payoff = float(player_payoff.split("=")[1])
                game_state.returns[player] = payoff

        return game_state