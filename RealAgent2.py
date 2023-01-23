
import numpy as np 
from copy import deepcopy, copy
from math import log
from quarto_agent_lib.quarto_utils import checkState
from quarto_agent_lib.collapse_state.collapse_state import collapse
from quarto_agent_lib.state_reward import StateReward, FixedRules

class RealAgent2():
    DEPTHS = [0, 0, 0, 0, 0, 3, 3, 3, 3, 10, 10, 5, 7, 7, 7, 7]
    MAX_NODES = [0, 0, 0, 0, 0, 13, 13, 13, 13, -1, -1, -1, -1, -1, -1, -1] #2197 states
    #DEPTHS = [0, 0, 0, 0, 0, 4, 4, 4, 4, 10, 10, 5, 7, 7, 7, 7]
    #MAX_NODES = [0, 0, 0, 0, 0, 7, 7, 7, 7, -1, -1, -1, -1, -1, -1, -1] ##2401 states
    #DEPTHS = [0, 0, 0, 0, 0, 5, 5, 5, 5, 10, 10, 5, 7, 7, 7, 7]
    #MAX_NODES = [0, 0, 0, 0, 0, 4, 4, 4, 4, -1, -1, -1, -1, -1, -1, -1] #1024 states
    #DEPTHS = [0, 0, 0, 0, 0, 7, 7, 7, 7, 10, 10, 5, 7, 7, 7, 7]
    #MAX_NODES = [0, 0, 0, 0, 0, 3, 3, 3, 3, -1, -1, -1, -1, -1, -1, -1] #2187 states
    #SKIP_COLLAPSE_PAWNS = True
    def __init__(self, states_cache = None, random_genome = False, use_debug_random_reward = False):
        self.reward_extimator = StateReward(genome=StateReward.load_genome())
        self.cache = dict()

    def solve_with_reward(self, state):
        
        #Check if winning or full
        #collapsed = collapse(state[0], state[1], RealAgent2.SKIP_COLLAPSE_PAWNS)
        #new_state = [ np.array(collapsed[0]), collapsed[1], list( (set( [x for x in range(16)] ) - set(collapsed[0])) - set( [collapsed[1],]))]        
        #return self.reward_extimator.get_reward(new_state)
        new_state = [ np.array(state[0]), state[1], list( (set( [x for x in range(16)] ) - set(state[0])) - set( [state[1],]))]        
        return self.reward_extimator.get_reward(new_state)

    def check_cache(self,state):
        hashed = str(state[:2])
        if (hashed in self.cache):
            return self.cache[hashed]
        return None
    def put_cache(self,state, reward):
        hashed = str(state[:2])
        self.cache[hashed] = reward
        
    N = 0
    def solve_with_minmax(self,states, in_depth, max_nodes, depth = 0):
        valuations = []
        for state in states:
            RealAgent2.N += 1
            #if (RealAgent2.N % 1000 == 0):
            #    print(f"{RealAgent2.N} - current depth: {depth} - n = {len(states)}")

            cache = self.check_cache(state)
            if (cache):
                #print("Cache hit")
                valuations.append(cache)
                continue

            full, winning = checkState(state[0])
            if (winning):
                valuations.append(-1000)
                self.put_cache(state, -1000)
                return valuations
            if (full):
                self.put_cache(state, 0)
                valuations.append(0)
                continue 
            if (depth >= in_depth):
                valuations.append(self.solve_with_reward(state))
                continue

            best_extimate = 20
            children = []
            available = set( [x for x in range(16)]) - set(state[0]) - set([state[1],])
            available_new = available - set([state[1],])
            for p in available:
                if (len(children)==max_nodes):
                    break
                for i in range(len(state[0])):
                    if (state[0][i]!=-1):
                        continue 
                    copied = [copy(state[0]), p]
                    copied[0][i] = state[1] 
                    #Extreme pruning
                    if (max_nodes==-1):
                        children.append(copied)
                    elif ((len(children) < max_nodes)):
                        ext = self.reward_extimator.get_reward([copied[0],copied[1], list( available_new - set( [copied[1],]))])
                        if (ext < best_extimate):
                            best_extimate = ext 
                        if (ext < 0):
                            children.append(copied)
                            if (len(children)==max_nodes):
                                break
            
            if (len(children)==0):
                valuations.append(20)
                self.put_cache(state, 20)
                continue
            if (depth >= in_depth-1):
                result = best_extimate
            else:
                result = min(self.solve_with_minmax(children, in_depth, max_nodes, depth+1))
            if (result == -1000):
                #print(f"Current depth: {depth} - Found winning move at this depth")
                pass
            self.put_cache(state, -result)
            valuations.append(-result)
            if (result == 1000):
                #print(f"Current depth: {depth} - All moves here lead to lose")
                return valuations
        return valuations
    
    def prune(self, states,size):
        return sorted(
            states, 
            key=lambda s: self.reward_extimator.get_reward([s[0],s[1], list( (set( [x for x in range(16)] ) - set(s[0])) - set( [s[1],]))])
            )[:size]

    def solve_states(self, states, initial_depth = None):
        states_depth = StateReward.count_state_size(states[0][0])
        if (states_depth < 5):
            return [ 
                -(FixedRules.min_n_pawns_biggest_line(state, FixedRules.get_lines(state))) 
                for state in states
            ]
        self.cache = dict()
        if (RealAgent2.MAX_NODES[states_depth] != -1 and len(states)>RealAgent2.MAX_NODES[states_depth]):
            states = self.prune(states, RealAgent2.MAX_NODES[states_depth])
        print(f" Depth: {states_depth}; Max nodes: {len(states)} - max depth: {RealAgent2.DEPTHS[states_depth]} ")
        return self.solve_with_minmax(states, RealAgent2.DEPTHS[states_depth], RealAgent2.MAX_NODES[states_depth])
        exit()
        

    def __lt__(self, other):
        return False