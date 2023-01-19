import random
import quarto
from quarto_agent import QuartoAgent, RealAgent
from quarto_agent_lib.state_reward import StateReward, FixedRules
N_INDIVIDUALS = 18
N_SURVIVORS = 8
N_GEN = 30
EXPORT_AT = 5
Nf = 0

def fight(ind1, ind2):
    global Nf
    Nf += 1
    game = quarto.Quarto(no_print=True)
    agent_1 = QuartoAgent.get_agent_custom_realagent(game, ind1, save_states = True)
    agent_2 = QuartoAgent.get_agent_custom_realagent(game, ind2)
    game.set_players((agent_1, agent_2))
    agent_1.new_match()
    winner = game.run()
    if (winner == -1):
        return 0
    agent_1.end_match(winner==0)
    if (winner == 0):
        return 1
    return -1

def value_individuals(individuals):
    points = [0 for _ in range(len(individuals))]
    for i in range(0, len(individuals)):
        for j in range(i+1, len(individuals)):
            print(f"{j + (i*len(individuals))} over {(len(individuals) * (len(individuals)-1)) / 2}")
            result = 0
            if (j%2==0):
                result = fight(individuals[i], individuals[j])
            else:
                result = -fight(individuals[j], individuals[i])
            points[i] += result 
            points[j] -= result
    #return [ [points[i], individuals[i]] for i in range(len(individuals))]
    return points #todo 

def train_de(genomes):
    random.seed()
    
    inds = []
    for genome in genomes:
        ind = RealAgent(random_genome=True)
        ind.WIDTHS = genome["WIDTHS"]
        ind.FIXED_RULE_N = genome["RULE"]
        ind.FIXED_RULE = FixedRules.get_function(ind.FIXED_RULE_N)
        inds.append(ind)
    points = [0 for _ in range(len(genomes))]
    for i in range(2):
        print(f"Iteration {i}")
        new_p = value_individuals(inds)
        for i in range(len(genomes)):
            points[i] += new_p[i]
    for i in range(len(genomes)):
        print(f"Points: {points[i]}")
        print(genomes[i])
        print("...")

def train():
    genomes = [
        {'WIDTHS': [3, 3, 3, 3, 3, 4, 5, 5, 5], 'RULE': 2}, 
        {'WIDTHS': [2, 2, 2, 2, 2, 4, 5, 5, 5], 'RULE': 2},
        {'WIDTHS': [4, 4, 4, 4, 4, 4, 5, 5, 5], 'RULE': 2},
        {'WIDTHS': [6, 6, 6, 6, 6, 7, 7, 7, 7], 'RULE': 2},
        {'WIDTHS': [7, 7, 7, 7, 7, 7, 7, 7, 7], 'RULE': 2},
        {'WIDTHS': [8, 8, 8, 8, 8, 7, 7, 7, 7], 'RULE': 2},
    ]
    train_de(genomes)

def train_():
    random.seed()
    individuals = [ RealAgent(random_genome=True) for _ in range(N_INDIVIDUALS) ]
    for i in range(N_GEN):
        print(f"Generation {i} of {N_GEN}")
        l = int(len(individuals)/2)
        ind_ = value_individuals(individuals[:l])
        ind_.extend(value_individuals(individuals[l:len(individuals)]))
        ind_.sort(reverse = True)
        new_gen = ind_[:N_SURVIVORS]
        print(f"Generation {i} - best score {ind_[0][0]}")
        print(f"Generation {i} - second best score {ind_[1][0]}")
        print(f"Generation {i} - third best score {ind_[2][0]}")
        if (i == N_GEN-1 or i%EXPORT_AT):
            print("Exp")
            ind_[0][1].export_genome()
        individuals = []
        for _ in range(N_INDIVIDUALS):
            c = random.choices(new_gen, k=2)
            i = (c[0][1].crossover(c[1][1]))
            i.random_mutation()
            individuals.append(i)
        