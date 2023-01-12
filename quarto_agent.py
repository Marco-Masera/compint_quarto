import quarto
import numpy as np 
from copy import deepcopy 
from math import log
from quarto_agent_lib.quarto_utils import checkState
from quarto_agent_lib.collapse_state.collapse_state import collapse
from quarto_agent_lib.state_reward import StateReward, FixedRules
import random


#State = [np_array[chessboard], assigned_pawn, set[remaining])
class RealAgent():
    #Higher = more precision but slower
    MAX_NODES = 5000
    MINMAX_FROM = 3

    def __init__(self):
        self.get_random_genome()
        self.cache = dict() # (reward, initial_depth)
        self.reward_extimator = StateReward(genome=StateReward.load_genome())

    def get_random_genome(self):
        self.WIDTHS = np.array([10 for _ in range(9 + 5 - RealAgent.MINMAX_FROM)]) #0..8 per length 5..13; dopo per stati con minmax
        self.FIXED_RULE = [FixedRules.max_active_lines, FixedRules.min_active_lines, FixedRules.min_common_features, FixedRules.min_n_pawns_biggest_line][random.randint(0,3)]

    def solve_with_fixed_rules(self, states):
        return [  -self.FIXED_RULE(state) for state in states ]

    def solve_with_minmax(self, states, depth):
        result = []
        #For each state to be avaluated
        for state in states:
            #Check if winning or full
            full, winning = checkState(state[0])
            if (winning):
                result.append(-1000)
                continue
            if (full):
                result.append(0)
                continue
            
            #Collapse and cache
            collapsed = collapse(state[0], state[1])
            state = [ np.array(collapsed[0]), collapsed[1], list(set([x for x in range(16)]) - set(state[0]) - set([state[1]]))]

            best_result = -99999
            #For each move possible in that state
            for box in state[0]:
                if (box != -1): continue 
                box = state[1]
                for pawn in state[2]:
                    best_result = max(best_result, -(self.solve_states([[state[0], pawn, state[2]]]))[0], depth )
                box = -1
            result.append(best_result)
        return result 

    def solve_with_heuristic(self, states, initial_depth, depth, width):
        results = []
        for state in states:
            #Check if winning or full
            full, winning = checkState(state[0])
            if (winning):
                results.append(-1000)
                #self.cache[str(state)] = (-1000, 1000)
                continue
            if (full):
                results.append(0)
                #self.cache[str(state)] = (0, 1000)
                continue
            """if (str(state) in self.cache):
                cached = self.cache[str(state)]
                if (cached[1] >= initial_depth):
                    results.append(cached[0])
                    continue"""
            
            if (depth >= self.MAX_DEPTH):
                results.append(self.reward_extimator.get_reward(state))
                continue

            children = []
            for index, box in enumerate(state[0]):
                if (box != -1): continue 
                for pawn in state[2]:
                    new_board = np.copy(state[0])
                    new_board[index] = state[1]
                    collapsed = collapse(new_board[0], pawn)
                    new_state = [ np.array(collapsed[0]), collapsed[1], list(set([x for x in range(16)]) - set(state[0]) - set([state[1]]))]
                    children.append( new_state )

            children.sort(key = lambda x: self.reward_extimator.get_reward(x))
            children = children[:width]
            r = ( - min( self.solve_with_heuristic( children, initial_depth, depth + 1, width )))
            #.cache[str(state)] = (r, initial_depth)
            results.append(r)
        return results

    def solve_states(self, states, initial_depth = None):
        states_depth = StateReward.count_state_size(states[0][0])
        if (initial_depth == None):
            initial_depth = states_depth
        if (states_depth < RealAgent.MINMAX_FROM):
            return self.solve_with_fixed_rules(states)
        if (states_depth < 5):
            return self.solve_with_minmax(states, initial_depth)
        #Collapse 
        for state in states:
            collapsed = collapse(state[0], state[1])
            state = [ np.array(collapsed[0]), collapsed[1], list(set([x for x in range(16)]) - set(state[0]) - set([state[1]]))]
        #Get width
        INDEX = initial_depth - 5
        if (INDEX < 0):
            INDEX = 9 + (initial_depth - RealAgent.MINMAX_FROM)
        WIDTH = self.MAX_NODES[INDEX]
        self.MAX_DEPTH = RealAgent.getDepthByWidth(WIDTH)
        return self.solve_with_heuristic(states, initial_depth, states_depth, WIDTH)

    def getDepthByWidth(width):
        return int( log(RealAgent.MAX_NODES, width) )

class QuartoAgent(quarto.Player):
    def __init__(self, quarto: quarto.Quarto) -> None:
        super().__init__(quarto)
        self.chosen_piece = None

    def choose_piece(self) -> int:
        if (self.chosen_piece!=None):
            return self.chosen_piece
        #Since it's the relationships between pieces that counts,
        #on the first move it doesn't change anything what piece we choose
        return 0 

    def place_piece(self) -> tuple[int, int]:
        usable_piece = self.quarto.get_selected_piece() #Index
        pass
        #return random.randint(0, 3), random.randint(0, 3)