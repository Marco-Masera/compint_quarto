import numpy as np
import json
import random
from collapse_state.collapse_state import collapse
from quarto_utils import checkState
from state_reward import StateReward

export_file = "dataset/raw/dataset_v9_"
DEPTH = 11
STOP_AT = 10 #Anche 10
MAX_BEFORE_DISCARD = 8000000

cache = None 
toExport = None
num = 0

class exported_info:
    def __init__(self, minmax_result, wins, loses, state):
        self.minmax_result = minmax_result       #MinMax result: 1 iff state leads to victory, 0 if leads to tie, -1 if leads to lose
        self.wins = wins                         #Without regards to MinMax, considers all branches sprawing from this state and counts the victories
        self.loses = loses                       #Like wins but for defeats
        #Can be computed after, but i might be useful to keep track 
        #of the ration wins/loses and the total number of leaves. High ration and low number of leaves = very good as it will be easy for the agent to find the best states
        self.state = state
    def get_value(self):
        if (self.wins + self.loses > 0):
            return (self.minmax_result*100) + ((self.wins / (self.wins + self.loses))*40)
        return (self.minmax_result)
    def get_tuple(self):
        return [[[int(x) for x in self.state[0]], self.state[1]], self.get_value()]



#State = (Chessboard state, chosen_pawn)
def solve_state(state, depth=0):   #Returns 1 if it is a good state for him, -1 if bad- 0 if stall
    global cache; global num

    MIN_NEG = 12
    MIN_NEUT = 12
    MAX_BEFORE_GIVEUP = 200
    state = collapse(state[0], state[1])
    if (str(state) in cache):
        return (cache[str(state)].minmax_result, cache[str(state)].wins + cache[str(state)].loses)
    num += 1

    if (num > MAX_BEFORE_DISCARD):
        return (-1001,0)
    if (num % 100000 == 0):
        print(f"States visited: {num}")

    #Check if there is a victory or if the chessboard is full
    chess_full, victory = checkState(state[0])
    if (victory):
        cache[str(state)] = exported_info(-1, 0, 1, state)
        return (-1,1) #Ha perso
    if (chess_full):
        cache[str(state)] = exported_info(0, 0, 0, state)
        return (0, 1)
    
    my_best_move = -1 
    wins = 0
    loses = 0
    neutral = 0
    tries = 0

    allowed = set([x for x in range(0,16)]) - set(state[0])
    my_pawn = state[1]
    if (my_pawn != None and my_pawn in allowed):
        allowed.remove(my_pawn)
    if (len(allowed)==0):
        allowed = [None]

    #Generate and shuffle the possible moves
    if (my_pawn!=None):
        possible_moves = [(index, i) for index in range(16) if state[0][index]==-1 for i in allowed]
    else: 
        possible_moves = [(-1, i) for i in allowed]
    random.shuffle(possible_moves)

    for index, i in possible_moves:
        #Debug
        if (depth==0):
            print(f"Computing one over {len(state[0])}")
        #Simulate move
        new_s = (np.copy(state[0]), i)
        if (my_pawn!=None):
            new_s[0][index] = my_pawn
        next = solve_state(new_s, depth=depth+1)
        result = -next[0]
        tries += next[1]
        if (result == 1001):
            return (-1001,0)

        if (result > my_best_move):
            my_best_move = result 
        if (result == 1):
            wins += 1
        elif (result == -1):
            loses += 1
        else:
            neutral += 1
        #If a winning branch has been found and at least MIN_TRIEST different leaves are reached, it can return
        if (wins >= 1 and ((neutral > MIN_NEUT and loses > MIN_NEG) or tries >= MAX_BEFORE_GIVEUP)):
            break
            
    cache[str(state)] = exported_info(my_best_move, wins, loses, state)
    toExport[str(state)] = exported_info(my_best_move, wins, loses, state)
    return (my_best_move, tries)

#State = (Chessboard state, chosen_pawn)
def solve_state_undeterministic(state, reward_ext, use_reward_at,  depth=0):   
    global cache; global num

    state = collapse(state[0], state[1])
    if (str(state) in cache):
        return cache[str(state)]
    num += 1

    if (num > MAX_BEFORE_DISCARD):
        return (-1001,False)
    if (num % 100000 == 0):
        print(f"States visited: {num}")

    #Check if there is a victory or if the chessboard is full
    chess_full, victory = checkState(state[0])
    if (victory):
        cache[str(state)] = (-140, False) 
        return (-140,False) 
    if (chess_full):
        cache[str(state)] = (0, False)
        return (0, False)

    #If at least use_reward_at pawns on the table, use the reward_extimator
    if (StateReward.count_state_size(state[0])>=use_reward_at):
        value = reward_ext.get_reward(StateReward.process_state(state))
        cache[str(state)] = (value, True)
        return (value, True)

    allowed = set([x for x in range(0,16)]) - set(state[0])
    my_pawn = state[1]

    if (my_pawn != None and my_pawn in allowed):
        allowed.remove(my_pawn)
    if (len(allowed)==0):
        allowed = [None]

    

    #Generate and shuffle the possible moves
    if (my_pawn!=None):
        possible_moves = [(index, i) for index in range(16) if state[0][index]==-1 for i in allowed]
    else: 
        possible_moves = [(-1, i) for i in allowed]
    random.shuffle(possible_moves)

    MAX_TRIES = 20
    best = -140
    best_n = []
    tries = 0
    is_extimate = True

    for index, i in possible_moves:
        #Debug
        if (depth==0):
            print(f"Computing one over {len(state[0])}")
        #Simulate move
        new_s = (np.copy(state[0]), i)
        if (my_pawn!=None):
            new_s[0][index] = my_pawn
        next = solve_state_undeterministic(new_s, reward_ext, use_reward_at, depth=depth+1)
        result = -next[0]
        tries += next[1]
        if (result > best):
            is_extimate = is_extimate and next[1]
            best = result
            if (best > 30):
                break

        if (result == 1001):
            return (-1001,0)
    


    cache[str(state)] = (best, is_extimate)
    toExport[str(state)] = exported_info(best, 0, 0, state)
    return (best, is_extimate)


def generate_one(depth, iteration, deterministic = True, stop_at = None, reward_ext = None):
    global cache; global num; global toExport
    assert(deterministic or (stop_at!=None and reward_ext!=None))

    toExport = dict()
    num = 0
    initial_state = []
    all_pawns = set([x for x in range(0,15)])
    for _ in range(16-depth):
        pawn = random.sample(all_pawns, 1)[0]
        all_pawns.remove(pawn)
        initial_state.append(pawn)
    for i in range(0, depth):
        initial_state.append(-1)
    random.shuffle(initial_state)

    if (deterministic):
        s = solve_state( (np.array(initial_state), None) )
    else:
        s = solve_state_undeterministic((np.array(initial_state), None), reward_ext, stop_at)
    
    if (s[0] == -1001):
        print(f"Iteration {iteration} failed")
        return

    #Transform cache for export
    to_export = []
    for value in toExport.values():
        tuple = value.get_tuple()
        #Discard non informative states
        if (tuple[1] == 0 and random.randint(0,100)<80):
            continue
        #print(tuple)
        to_export.append(tuple)

    with open(export_file+f"_{iteration}.json", 'w') as dataset:
        dataset.write(json.dumps({'exp': to_export}))
    print(f"Iteration {iteration} succeeded")


def generate_dataset(depth):
    global cache; global num
    cache = dict()
    random.seed()
    for i in range(5):
        generate_one(depth, i)

def generate_dataset_undeterministic(depth):
    global cache; global num
    cache = dict()
    random.seed()
    for i in range(5):
        generate_one(depth, i, False, STOP_AT, StateReward(StateReward.load_genome()))


    
generate_dataset_undeterministic(DEPTH)