import random
import quarto
from copy import deepcopy
from quarto_agent_lib.collapse_state.collapse_state import collapse
import numpy as np
from quarto_agent import QuartoAgent, RealAgent, StatesCache

def save_parent(state, cache):
    collapsed = collapse(state[0], state[1])
    state = [ np.array(collapsed[0]), collapsed[1]]
    cache.cache_specific_state(state, 1.5)
    

def train_on_state(agent, state, save_positive):
    reward_actual = agent.solve_with_minmax([state])[0]
    if (reward_actual >= 0):
        return 
    collapsed = collapse(state[0], state[1])
    state = [ np.array(collapsed[0]), collapsed[1], list( (set( [x for x in range(16)] ) - set(collapsed[0])) - set( [collapsed[1],]))]
    reward_ext = agent.reward_extimator.get_reward(state)
    if (reward_ext > -50 and reward_actual < 0):
        print(f"{reward_ext} - {reward_actual}")
        s = StatesCache()
        s.cache_specific_state(state, -1.5)
        for i in range(16):
            if (state[0][i]==-1):
                continue 
            new_s = deepcopy(state)
            new_s[1] = state[0][i]
            new_s[0][i] = -1
            save_parent(new_s, s)
        s.save_cache()

def train():
    random.seed()
    game = quarto.Quarto(no_print=True)
    agent = QuartoAgent.get_agent(game, use_cache = True, save_states = False).realAgent
    for k in range(5000):
        pawns = set([i for i in range(16)])
        state = [
            np.array([-1 for _ in range(16)]),
            -1
        ]
        for _ in range(8):
            pawn = random.sample(list(pawns), k=1)[0]
            while(True):
                i = random.randint(0,15)
                if state[0][i]==-1:
                    break 
            state[0][i] = pawn
            pawns.remove(pawn)
        state[1] = random.sample(list(pawns), k=1)[0]
        train_on_state(agent,state, k<500)
        if (k%100==0):
            print(k)
train()
        
