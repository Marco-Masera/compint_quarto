
import numpy as np 
from copy import copy
import json
import random
from quarto_agent_lib.quarto_utils import checkState
from quarto_agent_lib.state_reward import StateReward, LINES

#Set of fixed rules
class FixedRules():
    def get_lines(state):
        mylines = np.array([(False, 0,0,0,0) for _ in range(16)]) #(Active, N_Pawns, N_Features, Acc, last)
        for index,line in enumerate(LINES):
            count = 0; acc = 15; last = None
            for box in line[0]:
                if (state[0][box]!=-1):
                    count += 1
                    if (last != None):
                        acc = acc & (~(state[0][box] ^ last))
                    last = state[0][box]
            
            if (acc == 0 or count==0):
                mylines[index] = (False, 0, 0, 0, 0)
            else:
                n_ones = 0
                for bit in str(bin(acc)):
                    if (bit=='1'): n_ones+=1
                mylines[index] = (True, count, n_ones, acc, last)
        return mylines

    def min_active_lines(state, lines):
        #Higher score if fewer active lines
        return 15 - len([ l for l in lines if l[0]==False ])
    def min_n_pawns_biggest_line(state, lines):
        #Higher score if line with max number of pawns have fewer
        line = max(lines, key=lambda x: x[1])
        return 4 - line[1]
    def min_common_features(state, lines):
        #Higher score if line with max number of common features have fewer
        line = max(lines, key=lambda x: x[2])
        return 4 - line[2]

class QuartoRealAgent():
    #Agent static params
    def get_default_params():
        return {"DEPTHS" :         [0]*4 + [4 for _ in range(5)] + [13]*7, #!!!
                "MAX_NODES" :      [0]*4 + [5 for _ in range(5)] + [-1]*7, 
                "MAX_EVALS" :      [0]*4 + [28 for _ in range(5)] + [-1]*7}

    def __init__(self, params = None, skip_rl_layer = False, random_reward_function = False):
        self.cache = dict()
        if (random_reward_function==True):
            self.reward_extimator = StateReward()
        else:
            self.reward_extimator = StateReward(genome=StateReward.load_genome())
        self.skip_rl_layer = skip_rl_layer
        if (skip_rl_layer==True):
            self.perm_cache = dict()
        else:
            self.perm_cache = QuartoRealAgent.load_cache()
        if (params==None):
            params = QuartoRealAgent.get_default_params()
        self.DEPTHS = params["DEPTHS"]
        self.MAX_NODES = params["MAX_NODES"]
        self.MAX_EVALS = params["MAX_EVALS"]
        self.random_reward_function = random_reward_function

    def save_cache(self):
        if (self.skip_rl_layer==True):
            return
        with open("states_cache/cache.json", 'w') as exp:
            exp.write(json.dumps(self.perm_cache))
    
    def load_cache():
        try:
            with open("states_cache/cache.json", 'r') as source:
                return json.load(source) 
        except IOError as e:
            return dict()

    def add_to_perm_cache(self, state, reward):
        self.perm_cache[str(state[:2])] = reward

    def solve_with_reward(self, state):
        cached = self.check_cache(state, temp_too=False)
        if (not cached is None):
            return cached
        new_state = [ np.array(state[0]), state[1], list( (set( [x for x in range(16)] ) - set(state[0])) - set( [state[1],]))]        
        return self.reward_extimator.get_reward(new_state)

    def check_cache(self,state, temp_too = True):
        hashed = str(state[:2])
        if (hashed in self.perm_cache):
            return self.perm_cache[hashed]
        if (temp_too==True and hashed in self.cache):
            return self.cache[hashed]
        return None

    def put_cache(self,state, reward):
        hashed = str(state[:2])
        self.cache[hashed] = reward
        
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
                    #After limit of evaluations reached, it only checks if the state is winning or tie
                    if (skip or len(children)==max_eval):
                        copied = [copy(state[0]), p]
                        copied[0][i] = state[1] 
                        full, winning = checkState(copied[0])
                        if (winning):
                            self.add_to_perm_cache(copied, -1000)
                            self.add_to_perm_cache(state, 1000)
                            children = [(-180,copied)]
                            hard_skip = True 
                            break
                        else:
                            cached = self.check_cache(copied, False)
                            if (not cached is None):
                                children.append((cached, copied))
                    else:
                        #radical pruning: only states with extimated reward < 0 are visited
                        if ((len(children) < max_eval)):
                            copied = [np.array(state[0]), p]
                            copied[0][i] = state[1] 
                            ext = self.check_cache([copied[0],copied[1]])
                            if (ext is None):
                                ext = self.reward_extimator.get_reward([copied[0],copied[1], list( available_new - set( [copied[1],]))], size=size+1)
                                if (self.random_reward_function==True and ext!=-180):
                                    ext = random.randint(-15,5)
                            avg_reward += ext 
                            n_values += 1
                            if (ext < best_extimate):
                                best_extimate = ext 
                            if (ext < 0):
                                if (ext==-180):
                                    skip = True
                                    hard_skip = True
                                    children = [(-180,copied)]
                                    #self.add_to_perm_cache(copied, -1000)
                                    self.add_to_perm_cache(state, 1000)
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
            if (result == 1000):
                self.add_to_perm_cache(state, -1000)
            elif (result == -1000):
                self.add_to_perm_cache(state, 1000)
            valuations.append((-result, index))
        return valuations
    
    def prune(self, states,size, depth):
        if (depth==4):
            return sorted(
            states, 
            key=lambda s: -FixedRules.min_n_pawns_biggest_line(s[0], FixedRules.get_lines(s[0]))
            )[:size]
        return sorted(
            states, 
            key=lambda s: self.reward_extimator.get_reward([s[0][0],s[0][1], list( (set( [x for x in range(16)] ) - set(s[0][0])) - set( [s[0][1],]))], size=depth)
            )[:size]

    def solve_states(self, states, initial_depth = None):
        self.cache = dict()
        states_depth = StateReward.count_state_size(states[0][0])
        states = [ (s, index) for index, s in enumerate(states) ]
        if (states_depth < 4):
            x = [ 
                (-(FixedRules.min_n_pawns_biggest_line(state[0], FixedRules.get_lines(state[0]))), state[1] )
                for state in states
            ]
            random.shuffle(x)
            return x

        if (self.MAX_NODES[states_depth] != -1 and len(states)>self.MAX_NODES[states_depth]):
            states = self.prune(states, self.MAX_NODES[states_depth], states_depth)
        
        return self.solve_with_minmax(states, self.DEPTHS[states_depth], self.MAX_NODES[states_depth],self.MAX_EVALS[states_depth], states_depth, 0)
        exit()
    def __lt__(self, other):
        return False