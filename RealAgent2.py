
import numpy as np 
from copy import deepcopy, copy
from math import log
from quarto_agent_lib.quarto_utils import checkState
from quarto_agent_lib.collapse_state.collapse_state import collapse
from quarto_agent_lib.state_reward import StateReward, FixedRules

class RealAgent2():
    DEPTHS =         [0, 0, 0, 0, 0, 4,  4,  5,  5,  10, 10, 5, 7, 7, 7, 7]
    MAX_NODES =      [0, 0, 0, 0, 0, 8,  8,  6,  6, -1, -1, -1, -1, -1, -1, -1] #10.000 states
    MAX_EVALS =      [0, 0, 0, 0, 0, 42, 42, 38, 38, -1, -1, -1, -1, -1, -1, -1]


    def __init__(self, states_cache = None, random_genome = False, use_debug_random_reward = False):
        self.reward_extimator = StateReward(genome=StateReward.load_genome())
        self.cache = dict()

    def solve_with_reward(self, state):
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
        
    #N = 0; N2 = 0 #
    def solve_with_minmax(self,states, in_depth, max_nodes, max_eval, size, depth = 0):
        valuations = []
        for state_ in states:
            if (depth==0):
                state = state_[0]
                index = state_[1]
            else:
                state = state_
                index = -1

            cache = self.check_cache(state)
            if (cache):
                valuations.append((cache, index))
                continue

            if (size==15):
                valuations.append((self.solve_with_reward(state), index))
                continue

            full, winning = checkState(state[0])
            if (winning):
                valuations.append((-1000, index))
                self.put_cache(state, -1000)
                return valuations
            if (full):
                self.put_cache(state, 0)
                valuations.append((0,index))
                continue 
            if (depth >= in_depth):
                valuations.append((self.solve_with_reward(state), index))
                continue

            best_extimate = 20
            children = []
            skip = False; hard_skip = False
            available = set( [x for x in range(16)]) - set(state[0]) - set([state[1],])
            available_new = available - set([state[1],])
            avg_reward = 0; n_values = 0
            for p in available:
                if (hard_skip==True):
                        break
                for i in range(len(state[0])):
                    if (state[0][i]!=-1):
                        continue 
                    
                    if (max_nodes==-1):
                        copied = [copy(state[0]), p]
                        copied[0][i] = state[1] 
                        children.append((0,copied))
                        continue
                    #Extreme pruning - only states with reward < 0 are visited
                    if (skip or len(children)==max_eval):
                        old = copied[0][i]
                        copied[0][i] = state[1] 
                        full, winning = checkState(copied[0])
                        copied[0][i] = old
                        if (winning):
                            copied = [copy(state[0]), p]
                            copied[0][i] = state[1] 
                            children = [(-180,copied)]
                            hard_skip = True 
                            break
                    else:
                        copied = [np.array(state[0]), p]
                        copied[0][i] = state[1] 
                        if ((len(children) < max_nodes)):
                            ext = self.reward_extimator.get_reward([copied[0],copied[1], list( available_new - set( [copied[1],]))], size=size+1)
                            avg_reward += ext 
                            n_values += 1
                            if (ext < best_extimate):
                                best_extimate = ext 
                            if (ext < 0):
                                if (ext==-180):
                                    skip = True
                                    hard_skip = True
                                    children = [(-180,copied)]
                                    break
                                children.append((ext,copied))
                                if (len(children)==max_eval):
                                    skip = True
            
            if (len(children)==0):
                result =  - (avg_reward / n_values)
                valuations.append((result, index))
                self.put_cache(state, result)
                continue
            if (depth >= in_depth-1):
                result = best_extimate
            else:
                children = sorted(children, key = lambda x: x[0])
                if (len(children)>max_nodes):
                    children = children[:max_nodes]
                children = [s[1] for s in children]
                result = min(self.solve_with_minmax(children, in_depth, max_nodes, max_eval, size+1, depth+1))[0]
            self.put_cache(state, -result)
            valuations.append((-result, index))
        return valuations
    
    def prune(self, states,size, depth):
        return sorted(
            states, 
            key=lambda s: self.reward_extimator.get_reward([s[0][0],s[0][1], list( (set( [x for x in range(16)] ) - set(s[0][0])) - set( [s[0][1],]))], size=depth)
            )[:size]

    def solve_states(self, states, initial_depth = None):
        states_depth = StateReward.count_state_size(states[0][0])
        states = [ (s, index) for index, s in enumerate(states) ]
        if (states_depth < 5):
            return [ 
                (-(FixedRules.min_n_pawns_biggest_line(state[0], FixedRules.get_lines(state[0]))), state[1] )
                for state in states
            ]
        self.cache = dict()
        if (RealAgent2.MAX_NODES[states_depth] != -1 and len(states)>RealAgent2.MAX_NODES[states_depth]):
            states = self.prune(states, RealAgent2.MAX_NODES[states_depth], states_depth)
        #print(f" Depth: {states_depth}; Max nodes: {len(states)} - max depth: {RealAgent2.DEPTHS[states_depth]} ")
        x = self.solve_with_minmax(states, RealAgent2.DEPTHS[states_depth], RealAgent2.MAX_NODES[states_depth],RealAgent2.MAX_EVALS[states_depth], states_depth, 0)
        exit()
        

    def __lt__(self, other):
        return False