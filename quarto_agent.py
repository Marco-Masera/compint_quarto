import quarto
import numpy as np 
from copy import deepcopy, copy
from math import log
from quarto_agent_lib.quarto_utils import checkState
from quarto_agent_lib.collapse_state.collapse_state import collapse
from quarto_agent_lib.state_reward import StateReward, FixedRules
import random
import json

#Rl layer for the agent
class StatesCache():
    def __init__(self):
        self.cache = StatesCache.load_cache()
        self.queue = []
    
    def load_cache():
        try:
            with open("states_cache/cache.json", 'r') as source:
                return json.load(source) 
        except IOError as e:
            return dict()
    
    def new_match(self):
        self.queue = []

    def add_state(self, state):
        if (len(state)>=3):
            states = states[:2]
        self.queue.append(state) 

    def set_last_state(self, victory = True):
        #We add states before the move (state our player is into) and after (state the enemy is into)
        #If player wins, last state is negative, if player lose, last state is positive
        self.queue.reverse()
        if (victory):
            reward = -1
        else:
            reward = 1
        for state in self.queue:
            #Save state with reward
            self.cache_specific_state(state, reward)
            reward = -(reward * 0.7)
            if (reward < 0.1 and reward > -0.1):
                break

    def hash_state(state):
        s = 0
        for index, item in enumerate(state[0]):
            it = int(item) #prevent overflow
            s += (it * (17**(index)))
        s += (state[1] * (17**(16)))
        return str(s)

    def cache_specific_state(self, state, reward):
        if (len(state)>=3):
            states = states[:2]
        hashed = StatesCache.hash_state(state)
        if (hashed in self.cache):
            self.cache[hashed] = (self.cache[hashed]+reward)/2
        else:
            self.cache[hashed] = reward

    def save_cache(self):
        c2 = StatesCache.load_cache()
        #Merge the two
        for key in self.cache.keys():
            if (key in c2):
                c2[key] = (c2[key] + self.cache[key]) / 2
            else:
                c2[key] = self.cache[key]
        with open("states_cache/cache.json", 'w') as exp:
            exp.write(json.dumps(c2))
        self.cache = c2

    #Check if state in permanent and return reward
    def check_state(self,state):
        if (len(state)>=3):
            state = state[:2]
        hashed = StatesCache.hash_state(state)
        if (hashed in self.cache):
            return self.cache[hashed]
        return 0


#State = [np_array[chessboard], assigned_pawn, set[remaining])
class RealAgent():
    #Higher = more precision but slower
    MAX_NODES = 2000
    MIN_WIDTH = 2
    MAX_WIDTH = 40

    def __init__(self, states_cache = None, random_genome = False, use_debug_random_reward = False):
        if (random_genome):
            self.get_random_genome()
        else:
            self.load_genome()
        self.cache = dict() # (reward, initial_depth)
        if (not use_debug_random_reward):
            self.reward_extimator = StateReward(genome=StateReward.load_genome())
        else:
            self.reward_extimator = StateReward()
        self.states_cache = states_cache

    def random_mutation(self):
        if (random.randint(0,3)==0):
            return 
        if (random.randint(0,3)==0):
            self.FIXED_RULE_N = random.randint(0,3)
            self.FIXED_RULE = FixedRules.get_function(self.FIXED_RULE_N)
        if (random.randint(0,1)==0):
            index = random.randint(0, len(self.WIDTHS)-1)
            if (random.randint(0,1)==0):
                self.WIDTHS[index] = min(self.WIDTHS[index]+1, RealAgent.MAX_WIDTH)
            else:
                self.WIDTHS[index] = max(self.WIDTHS[index]-1, RealAgent.MIN_WIDTH)

    def export_genome(self):
        print(f"Export: {self.FIXED_RULE_N}")
        with open("agent_params/params.json", 'w') as exp:
            l = [int(x) for x in self.WIDTHS]
            params = { "WIDTHS": l, "RULE": self.FIXED_RULE_N }
            exp.write(json.dumps(params))
    
    def load_genome(self):
        with open("agent_params/params.json", 'r') as exp:
            g = json.load(exp)
            self.WIDTHS = np.array(g["WIDTHS"])
            self.FIXED_RULE = FixedRules.get_function(g["RULE"])
            self.FIXED_RULE_N = g["RULE"]
            
    def crossover(self, other):
        new_ind = RealAgent(self.states_cache, True)
        if (random.randint(0,1)==0):
            new_ind.FIXED_RULE = self.FIXED_RULE
            new_ind.FIXED_RULE_N = self.FIXED_RULE_N
        else:
            new_ind.FIXED_RULE_N = other.FIXED_RULE_N
            new_ind.FIXED_RULE = other.FIXED_RULE
        new_ind.WIDTHS = np.array([
            random.choice([self.WIDTHS[i], other.WIDTHS[i]])
            for i in range(9)
            ]) 
        return new_ind

    def get_random_genome(self):
        self.WIDTHS = np.array([random.randint(RealAgent.MIN_WIDTH,RealAgent.MAX_WIDTH) for _ in range(9)]) #0..8 per length 5..13; dopo per stati con minmax
        self.FIXED_RULE_N = random.randint(0,3)
        self.FIXED_RULE = FixedRules.get_function(self.FIXED_RULE_N)

    def fixed_rule_points(state, rule):
        lines = FixedRules.get_lines(state)
        #Check if winning state
        if len( [l for l in lines if l[0]==True and l[1]==4] )>0:
            return -100
        #Check if losing state
        if len( [l for l in lines 
            if l[0]==True and l[1]==3 and (l[3]&(~(state[1] ^ l[4]))!=0)
        ] )>0:
            return 100
        return - (rule(state, lines))

    def solve_with_fixed_rules(self, states):
        return [  RealAgent.fixed_rule_points(state,self.FIXED_RULE) for state in states ]

    def solve_with_heuristic(self, states, initial_depth, depth, width):
        results = []
        for state in states:
            self.N += 1
            if (self.N % 500 == 0):
                pass#print(self.N)
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

            if (depth >= self.MAX_DEPTH or count == 15 or self.N > self.MAX_NODES):
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

            if (self.states_cache != None):
                children.sort(key = lambda x: 
                    (self.reward_extimator.get_reward(x) + (self.states_cache.check_state(x)*100))/2
                )
            else:
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
        if (states_depth < 5):
            return self.solve_with_fixed_rules(states)

        #Collapse 
        collapsed_states = []
        for state in states:
            collapsed = collapse(state[0], state[1])
            collapsed_states.append([ np.array(collapsed[0]), collapsed[1], list( (set( [x for x in range(16)] ) - set(collapsed[0])) - set( [collapsed[1],]))])
        #Get width
        if (initial_depth < 14):
            INDEX = initial_depth - 5
            if (INDEX < 0):
                INDEX = 0
            WIDTH = self.WIDTHS[INDEX]
            self.MAX_DEPTH = RealAgent.getDepthByWidth(WIDTH) + initial_depth
        else:
            WIDTH = 50
            self.MAX_DEPTH = 17
        self.N = 0
        return self.solve_with_heuristic(collapsed_states, initial_depth, states_depth, WIDTH)

    def getDepthByWidth(width):
        return int( log(RealAgent.MAX_NODES, width) )
        
    def __lt__(self, other):
        return False

class QuartoAgent(quarto.Player):
    def __init__(self, quarto: quarto.Quarto, realAgent, use_cache, save_states, debug_use_random_reward) -> None:
        super().__init__(quarto)
        assert (use_cache or not save_states)
        self.chosen_piece = None
        self.realAgent = realAgent
        if (use_cache):
            self.st_cache = StatesCache()
            self.realAgent.states_cache = self.st_cache
        self.save_states = save_states

    def get_agent(quarto: quarto.Quarto, use_cache = True, save_states = False, debug_use_random_reward = False):
        return QuartoAgent(quarto, realAgent=RealAgent(), use_cache = use_cache, save_states=save_states, debug_use_random_reward=debug_use_random_reward)

    def get_agent_random_genome(quarto: quarto.Quarto, use_cache = True, save_states = False, debug_use_random_reward = False):
        return QuartoAgent(quarto, realAgent=RealAgent(random_genome=True, use_debug_random_reward=debug_use_random_reward), use_cache = use_cache, save_states=save_states, debug_use_random_reward=debug_use_random_reward)

    def get_agent_custom_realagent(quarto: quarto.Quarto, real_agent: RealAgent,use_cache=True,  save_states = False, debug_use_random_reward = False):
        #debug_use_random_reward = False
        return QuartoAgent(quarto, real_agent, use_cache, save_states, debug_use_random_reward)

    def new_match(self):
        self.st_cache.new_match()
    def end_match(self, you_won):
        self.st_cache.set_last_state(you_won)
        self.st_cache.save_cache()

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

        #Add state to StatesCache
        if (self.save_states):
            self.st_cache.add_state(self.convert_state_format(board, usable_piece))

        for i in range(0,4):
            for j in range(0,4):
                if (board[i][j] in usable_pieces):
                    usable_pieces.remove(board[i][j])
        if (len(usable_pieces)==0):
            for i in range(0,4):
                for j in range(0,4):
                    if (board[j][i] == -1): 
                        return (i,j)
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
        
        if (self.save_states):
            self.st_cache.add_state(states[index])

        self.chosen_piece = move[2]
        return (move[1],move[0])

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