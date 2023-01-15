import quarto
import numpy as np 
from copy import deepcopy, copy
from math import log
from quarto_agent_lib.quarto_utils import checkState
from quarto_agent_lib.collapse_state.collapse_state import collapse
from quarto_agent_lib.state_reward import StateReward, FixedRules
import random


#State = [np_array[chessboard], assigned_pawn, set[remaining])
class RealAgent():
    #Higher = more precision but slower
    MAX_NODES = 10000
    MINMAX_FROM = 5

    def __init__(self, use_debug_random_reward = False):
        self.get_random_genome()
        self.cache = dict() # (reward, initial_depth)
        if (not use_debug_random_reward):
            self.reward_extimator = StateReward(genome=StateReward.load_genome())
        else:
            self.reward_extimator = StateReward()

    def get_random_genome(self):
        self.WIDTHS = np.array([6 for _ in range(9 + 5 - RealAgent.MINMAX_FROM)]) #0..8 per length 5..13; dopo per stati con minmax
        self.FIXED_RULE = [FixedRules.max_active_lines, FixedRules.min_active_lines, FixedRules.min_common_features, FixedRules.min_n_pawns_biggest_line][random.randint(0,3)]

    def hash_state(state, chosen_pawn):
        s = 0
        for index, item in enumerate(state):
            it = int(item) #prevent overflow
            s += (it * (17**(index)))
        s += (chosen_pawn * (17**(16)))
        return s


    def solve_with_fixed_rules(self, states):
        return [  -self.FIXED_RULE(state) for state in states ]


    def solve_with_minmax(self, states, depth):
        result = []
        #For each state to be avaluated
        for index,state in enumerate(states):
            print(f"{index} of {len(states)}")
            #Check if winning or full
            full, winning = checkState(state[0])
            if (winning):
                result.append(-1000)
                self.cache[str(state)] = (-1000, 1000)
                continue
            if (full):
                result.append(0)
                self.cache[str(state)] = (0, 1000)
                continue
            
            #Collapse and cache
            collapsed = collapse(state[0], state[1])
            state = [ np.array(collapsed[0]), collapsed[1], list(set([x for x in range(16) if x != collapsed[1]]) - set(collapsed[0]) )]
            if (str(state) in self.cache):
                cached = self.cache[str(state)]
                if (cached[1] >= depth):
                    result.append(cached[0])
                    continue
            best_result = -99999
            #For each move possible in that state
            for index, box in enumerate(state[0]):
                if (box != -1): continue 
                for pawn in state[2]:
                    state_c = copy(state[0])
                    state_c[index] = state[1]
                    best_result = max(best_result, -(self.solve_states([ [state_c, pawn, state[2]] ]))[0], depth )
                    if (best_result == 1000): break
            result.append(best_result)
            self.cache[str(state)] = (best_result, 1000)
        return result 


    def solve_with_heuristic(self, states, initial_depth, depth, width):
        results = []
        for state in states:
            #Check if winning or full
            full, winning = checkState(state[0])
            if (winning):
                results.append(-1000)
                self.cache[str(state)] = (-1000, 1000)
                continue
            if (full):
                results.append(0)
                self.cache[str(state)] = (0, 1000)
                continue
            if (str(state) in self.cache):
                cached = self.cache[str(state)]
                if (cached[1] >= initial_depth):
                    results.append(cached[0])
                    continue
            
            count = len([x for x in state[0] if x!=-1])

            if (depth >= self.MAX_DEPTH or count == 15):
                results.append(self.reward_extimator.get_reward(state))
                continue


            children = []
            for index, box in enumerate(state[0]):
                if (box != -1): continue 
                for pawn in state[2]:
                    new_board = np.copy(state[0])
                    new_board[index] = state[1]
                    collapsed = collapse(new_board, pawn)
                    new_state = [ np.array(collapsed[0]), collapsed[1], list( (set( [x for x in range(16)] ) - set(collapsed[0])) - set( [collapsed[1],]))]
                    children.append( new_state )

            children.sort(key = lambda x: self.reward_extimator.get_reward(x))
            children = children[:width]
            r = ( - min( self.solve_with_heuristic( children, initial_depth, depth + 1, width )))
            self.cache[str(state)] = (r, initial_depth)
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
        collapsed_states = []
        for state in states:
            collapsed = collapse(state[0], state[1])
            collapsed_states.append([ np.array(collapsed[0]), collapsed[1], list( (set( [x for x in range(16)] ) - set(collapsed[0])) - set( [collapsed[1],]))])
        #Get width
        if (initial_depth < 14):
            INDEX = initial_depth - 5
            if (INDEX < 0):
                INDEX = 9 + (initial_depth - RealAgent.MINMAX_FROM)
            WIDTH = self.WIDTHS[INDEX]
            self.MAX_DEPTH = RealAgent.getDepthByWidth(WIDTH)
        else:
            WIDTH = 50
            self.MAX_DEPTH = 17
        return self.solve_with_heuristic(collapsed_states, initial_depth, states_depth, WIDTH)

    def getDepthByWidth(width):
        return int( log(RealAgent.MAX_NODES, width) )

class QuartoAgent(quarto.Player):
    def __init__(self, quarto: quarto.Quarto, debug_use_random_reward = False) -> None:
        super().__init__(quarto)
        self.chosen_piece = None
        self.realAgent = RealAgent(debug_use_random_reward)

    def choose_piece(self) -> int:
        if (self.chosen_piece!=None):
            return self.chosen_piece
        #Since it's the relationships between pieces that counts,
        #on the first move it doesn't change anything what piece we choose
        return 0 

    def place_piece(self) -> tuple[int, int]:
        usable_piece = self.get_game().get_selected_piece() #Index
        moves = []
        states = []

        board = self.get_game().get_board_status()
        usable_pieces = set([i for i in range(0,16)]) - set([usable_piece])
        for i in range(0,4):
            for j in range(0,4):
                if (board[i][j] in usable_pieces):
                    usable_pieces.remove(board[i][j])
        if (len(usable_pieces)==0):
            for i in range(0,4):
                for j in range(0,4):
                    if (board[i][j] != -1): return (j,i)
        for i in range(0,4):
            for j in range(0,4):
                if (board[i][j] != -1): continue
                for piece in usable_pieces:
                    moves.append((i,j,piece))
                    board[i][j] = usable_piece
                    states.append(self.convert_state_format(board, piece)) 
                    board[i][j] = -1
    
        solved = self.realAgent.solve_states(states)
        index = solved.index(min(solved))
        move = moves[index]
        self.chosen_piece = move[2]
        return (move[1],move[0])


    def print_board(self,board):
        '''
        Print the board
        '''
        for row in board:
            print("\n -------------------")
            print("|", end="")
            for element in row:
                print(f" {element: >2}", end=" |")
        print("\n -------------------\n")

    def convert_state_format(self, board, assigned_pawn):
        new_board = []
        for i in range(0,4):
            for j in range(i, -1, -1):
                new_board.append(board[j][i-j])
        new_board.append((board[3][1]))
        new_board.append((board[2][2]))
        new_board.append((board[1][3]))
        new_board.append((board[3][2]))
        new_board.append((board[2][3]))
        new_board.append((board[3][3]))
        return [new_board, assigned_pawn]