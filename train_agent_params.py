import random
import quarto
from quarto_agent import QuartoAgent, RealAgent

N_INDIVIDUALS = 18
N_SURVIVORS = 8
N_GEN = 2

Nf = 0

def fight(ind1, ind2):
    global Nf
    print(f"Fight n {Nf}")
    Nf += 1
    game = quarto.Quarto(no_print=True)
    agent_1 = QuartoAgent.get_agent_custom_realagent(game, ind1)
    agent_2 = QuartoAgent.get_agent_custom_realagent(game, ind2)
    game.set_players((agent_1, agent_2))
    winner = game.run()
    if (winner == -1):
        return 0
    if (winner == 0):
        return 1
    return -1

def value_individuals(individuals):
    #Each individual fights 6 others
    points = [0 for _ in range(len(individuals))]

    for i in range(0, len(individuals)):
        for j in range(i+1, len(individuals)):
            result = 0
            if (j%2==0):
                result = fight(individuals[i], individuals[j])
            else:
                result = -fight(individuals[j], individuals[i])
            points[i] += result 
            points[j] -= result
            print("End fight")
    return [ [points[i], individuals[i]] for i in range(len(individuals))]

def train():
    random.seed()
    individuals = [ RealAgent(random_genome=True) for _ in range(N_INDIVIDUALS) ]
    for i in range(N_GEN):
        print(f"Generation {i} of {N_GEN}")
        l = int(len(individuals)/3)
        individuals = value_individuals(individuals[:l])
        individuals.extend(value_individuals(individuals[l:l*2]))
        individuals.extend(value_individuals(individuals[l*2:len(individuals)]))
        individuals.sort(reverse = True)
        new_gen = individuals[:N_SURVIVORS]
        print(f"Generation {i} - best score {individuals[0][0]}")
        print(f"Generation {i} - second best score {individuals[1][0]}")
        print(f"Generation {i} - third best score {individuals[2][0]}")
        if (i == N_GEN-1):
            print("Exp")
            individuals[0][1].export_genome()
            break
        individuals = []
        for _ in range(N_INDIVIDUALS):
            c = random.choices(new_gen, k=2)
            i = (c[0][1].crossover(c[1][1]))
            i.random_mutation()
            individuals.append(i)
        