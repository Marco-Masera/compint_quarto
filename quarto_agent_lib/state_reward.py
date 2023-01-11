import numpy as np
import json
import random
import copy
from quarto_utils import checkState, bits_in_common_multiple

#Init some static stuff to calculate the feature faster during runtime
LINES = [
    # ( (caselle),[(quale_riga_tocca, casella )])
    ((0, 1, 3, 6), [] ),   #0
    ((2, 4, 7, 10), []),      #1
    ((5,8, 11, 13), []),      #2
    ((9, 12, 14, 15), []),    #3
    ((6, 10, 13, 15), []),    #4
    ((3, 7, 11, 14), []),     #5
    ((1, 4, 8, 12), []),     #6
    ((0, 2, 5, 9), []),       #7
    ((6, 7, 8, 9), []),       #8
    ((0, 4, 11, 15), [])]     #9
LINES_PRECEDING = np.array([0 for _ in range(16)])
count_ = 0
for index,line in enumerate(LINES):
    LINES_PRECEDING[index] = count_
    for index2 in range(index+1, len(LINES)):
        line2 = LINES[index2]
        check = -1
        for c in line2[0]:
            if (c in line[0]):
                check = c
                line[1].append([index2, c])
                count_  = count_ + 1
                break
LINES_PRECEDING[len(LINES_PRECEDING)-1] = count_
MULTILINE_BLOCK = ((LINES_PRECEDING[len(LINES_PRECEDING)-1] * 6) + 80)


class StateReward:
    MIN_SIZE = 5
    state_length = (MULTILINE_BLOCK)*2
    #Process state is used to convert the raw state from the agent to the format used to compute the function
    #return [[chessboard], assigned_pawn, set[remaining], {reward} ]
    def process_state_for_dataset(state):
        return [state[0][0], state[0][1], list(set([x for x in range(16)]) - set(state[0][0]) - set([state[0][1]]) ), state[1]]

    def process_state(state):
        return [np.array(state[0]), state[1], list(set([x for x in range(16)]) - set(state[0]) - set([state[1]]))]

    def get_random_genome(self):
        genome = dict()
        for i in range(1,15):
            genome[i] = np.array([random.uniform(-1,1) for _ in range(StateReward.state_length)])
        return genome
    
    def load_genome():
        with open("training/reward_genome.json", 'r') as source:
            genome = json.load(source)
        g = dict()
        for key in genome.keys():
            g[int(key)] = np.array(genome[key])
        return g

    def count_state_size(state):
        c = 0
        for i in state: 
            if i>-1: c+=1;
        return c
    
    def solve_last_move(self,state):
        st = copy.deepcopy(state)
        for box in st[0]:
            if (box==-1): box = st[1]
        full, winning = checkState(st[0])
        if (winning):
            return 120
        return 0

    def get_reward_from_truth_value(self,truth_value, state_length):
        reward = np.matmul(self.genome[state_length],truth_value)
        if (reward > 0):
            return min(reward,140)
        else:
            return max(-140, reward)

    #State needs to be processed via process_state before this function is called
    def get_reward(self, state): #State = ([chessboard], assigned_pawn, set[remaining])
        global LINES; global LINES_PRECEDING; global MULTILINE_BLOCK;

        size = StateReward.count_state_size(state[0])
        full, winning = checkState(state[0])
        if (winning): #player already won
            return -160
        if (full):
            return 0
        if (size==15):
            return self.solve_last_move(state)
        if (size < StateReward.MIN_SIZE):
            return 0
            
        genome = self.genome[size]
        chessboard = state[0]
        mylines = np.array([(False, 0,0) for _ in range(16)])
        self.truth_value = np.array([0 for _ in range(StateReward.state_length)])
        #mylines = [] -> (Active, acc)
        for index,line in enumerate(LINES):
            count = 0; acc = 15; last = None
            for box in line[0]:
                if (chessboard[box]!=-1):
                    count += 1
                    if (last != None):
                        acc = acc & (~(chessboard[box] ^ last))
                    last = chessboard[box]
            
            if (acc == 0):
                mylines[index] = (False, acc, 0)
                continue 
            #can extend, can block -> calcolati da pawn
            if (state[1]!=None):
                if (count != 0 and acc & (~(state[1] ^ last))):#(can_extend):
                    self.truth_value[(index*8) + count] = 1
                else: # (can_block):
                    self.truth_value[(index*8) + count + 4] = 1

            #posso bloccare avversario -> calcolato
            one_extending = False; one_blocking = False
            for pawn in state[2]:
                if (count!=0 and acc & (~(pawn ^ last))):
                    one_extending = True 
                else:
                    one_blocking = True 
                if (one_blocking and one_extending): break

            if (one_extending):
                self.truth_value[MULTILINE_BLOCK + (index*8) + count] = 1
            elif (one_blocking):
                self.truth_value[MULTILINE_BLOCK + (index*8) + count + 4] = 1
            #aggiorna linea singola -> index
            if (last==None):
                last = -1
            mylines[index] = (True, acc, last)
        #Secondo giro
        for index,line in enumerate(LINES):
            if (mylines[index][0] == False): continue
            for common in line[1]:
                if (mylines[common[0]][0] == False): continue
                if (chessboard[common[1]] != -1): continue 

                #Verifica se posso bloccare entrambe, aumentarle entrambe o una e una
                acc = mylines[index][1]; acc2 = mylines[common[0]][1]
                if (state[1]!=None):
                    can_extend_one = (acc & (~(state[1] ^ mylines[index][2])))!=0
                    can_extend_two = (acc2 & (~(state[1] ^ mylines[common[0]][2])))!=0
                    if (can_extend_one and can_extend_two):
                        self.truth_value[80 + (LINES_PRECEDING[index] * 6)] = 1
                    elif (can_extend_one != can_extend_two):
                        self.truth_value[80 + (LINES_PRECEDING[index] * 6) + 1] = 2
                    else:
                        self.truth_value[80 + (LINES_PRECEDING[index] * 6) + 2] = 3
                #verifica se posso bloccare avversario, verifica se può bloccare me
                canEO = 0; canBO = 0; canM = 0
                for pawn in state[2]:
                    can_extend_one = (acc & (~(pawn ^ mylines[index][2])))!=0
                    can_extend_two = (acc2 & (~(pawn ^ mylines[common[0]][2])))!=0
                    if (can_extend_one and can_extend_two):
                        canEO += 1
                        if (canEO==1):
                            self.truth_value[80 + MULTILINE_BLOCK + (LINES_PRECEDING[index] * 6)] = 1
                        else:
                            self.truth_value[80 + MULTILINE_BLOCK + (LINES_PRECEDING[index] * 6) + 3] = 1
                    elif (can_extend_one != can_extend_two):
                        canBO += 1
                        if (canBO==1):
                            self.truth_value[80 + MULTILINE_BLOCK + (LINES_PRECEDING[index] * 6) + 1] = 1
                        else:
                            self.truth_value[80 + MULTILINE_BLOCK + (LINES_PRECEDING[index] * 6) + 4] = 1
                    else:
                        canM += 1
                        if (canM==1):
                            self.truth_value[80 + MULTILINE_BLOCK + (LINES_PRECEDING[index] * 6) + 2] = 1
                        else:
                            self.truth_value[80 + MULTILINE_BLOCK + (LINES_PRECEDING[index] * 6) + 5] = 1
                    if (canEO >= 2 and canBO >= 2 and canM >= 2): break

        reward = np.matmul(genome,self.truth_value)
        if (reward > 0):
            return min(reward,140)
        else:
            return max(-140, reward)

    def __init__(self, genome = None):
        if (genome==None):
            self.genome = self.get_random_genome()
        else:
            self.genome = genome


    def random_mutations(self, n_mutation):
        pass
    
    def crossover(ind1, ind2):
        return StateReward()

    def __lt__(self, other):
        return False



#The following code handles the learning part
STATES = [] # [ [chessboard], pawn, [remaining], real_reward ]
VALIDATION_STATES = []
SAMPLE_TARGET = 8
#Min 5
LENGTHS_TO_TRAIN = ["6","8","9","10",  "11","12","13","15"]

class Climber:
    def __init__(self):
        self.individual = StateReward(genome=StateReward.load_genome())
        self.build_truth_values()

    def build_truth_values(self):
        self.states_truth_values = dict()
        for size_ in STATES.keys():
            l = []
            for state in STATES[size_]:
                #self.states_truth_values[size_]
                self.individual.get_reward(state)
                value = copy.deepcopy(self.individual.truth_value)
                l.append(value)
            self.states_truth_values[size_] = np.array(l)

    def random_error(self, sample):
        error = 0
        for state in sample:
            reward = 0
            error += (reward - state[3])**2
        return (error / len(sample))


    def value_individual(self,individual, key):
        error = 0
        for index, state in enumerate(STATES[key]):
            truth_value = self.states_truth_values[key][index]
            reward = individual.get_reward_from_truth_value(truth_value, int(key))
            error += (reward - state[3])**2
        return (error / len(STATES[key]))

    def value_individual_globally(self, individual, states = STATES):
        error = 0
        num = 0
        for size_ in LENGTHS_TO_TRAIN:
            if (not size_ in states): continue
            num += len(states[size_])
            for state in states[size_]:
                reward = individual.get_reward(state)
                error += (reward - state[3])**2
        return (error / num)
    
    def print_evaluations(self, individual, states = STATES):
        for size_ in LENGTHS_TO_TRAIN:
            if (not size_ in states): continue
            num = len(states[size_])
            error = 0
            for state in states[size_]:
                reward = individual.get_reward(state)
                error += (reward - state[3])**2
            print(f"Length: {size_}, ds_size: {num}, error: {error/num}")

    def validate(self):
        with_traning_dataset = self.value_individual_globally(self.individual, STATES)
        with_validation_dataset = self.value_individual_globally(self.individual, VALIDATION_STATES)
        print(f"Training dataset error: {with_traning_dataset}. Validation dataset error: {with_validation_dataset}")
        return {"training": with_traning_dataset, "validation": with_validation_dataset}

    def new_gen(self, learning_rate = 0.07):
        for size_ in LENGTHS_TO_TRAIN:
            if (not size_ in STATES): continue
            size_int = int(size_)
            self.error = self.value_individual(self.individual, size_)
            for i in range(StateReward.state_length):
                old_v = self.individual.genome[size_int][i]
                self.individual.genome[size_int][i] = old_v + learning_rate
                error = self.value_individual(self.individual, size_)
                if (error < self.error):
                    self.error = error 
                    continue
                self.individual.genome[size_int][i] = old_v - learning_rate
                error = self.value_individual(self.individual, size_)
                if (error < self.error):
                    self.error = error 
                    continue
                self.individual.genome[size_int][i] = old_v
    
    def export_self(self, num, gen):
        to_export = copy.deepcopy(self.individual.genome)
        for i in range(1,16):
            to_export[i] = list(to_export[i])
        with open(f"training/export_{gen}_{num}", 'w') as exp:
            exp.write(json.dumps(to_export))

def export(climber, iteration, generation):
    with open(f"training/result_{generation}_{iteration}", 'w') as exp:
        exp.write(json.dumps(climber.validate()))
    climber.export_self(iteration, generation)

def load_data():
    global STATES; global VALIDATION_STATES
    with open("dataset/pre_processed/training_dataset.json", 'r') as source:
        STATES = json.load(source)
        for size_ in STATES.keys():
            for state in STATES[size_]:
                state[0] = np.array(state[0])
    with open("dataset/pre_processed/training_dataset_v2.json", 'r') as source:
        states_2 = json.load(source)
        for size_ in states_2.keys():
            for state in states_2[size_]:
                state[0] = np.array(state[0])
            if (size_ in STATES):
                STATES[size_].extend(states_2[size_])
            else:
                STATES[size_] = states_2[size_]

    with open("dataset/pre_processed/validation_dataset.json", 'r') as source:
        VALIDATION_STATES = json.load(source)
        for size_ in VALIDATION_STATES.keys():
            for state in VALIDATION_STATES[size_]:
                state[0] = np.array(state[0])
    with open("dataset/pre_processed/validation_dataset_v2.json", 'r') as source:
        states_2 = json.load(source)
        for size_ in states_2.keys():
            for state in states_2[size_]:
                state[0] = np.array(state[0])
            if (size_ in VALIDATION_STATES):
                VALIDATION_STATES[size_].extend(states_2[size_])
            else:
                VALIDATION_STATES[size_] = states_2[size_]

            

def climb():
    load_data()
    generation = 2

    climber = Climber()
    climber.print_evaluations(climber.individual, VALIDATION_STATES)
    for i in range(40):
        print(f"Gen {i}")
        climber.new_gen(0.15)
    export(climber, 1,generation)
    for i in range(100):
        print(f"Gen {i}")
        climber.new_gen(0.1)
    export(climber, 2,generation)
    exp_n = 2
    for i in range(100):
        print(f"Gen {i}")
        climber.new_gen(0.07)
        
    export(climber, exp_n,generation)
    climber.print_evaluations(climber.individual, VALIDATION_STATES)
    return
    for i in range(150):
        print(f"Gen {i}")
        climber.new_gen(0.05)
        if (i%20==0):
            exp_n += 1
            export(climber, exp_n,generation)
            
    export(climber, exp_n+1,generation)

    climber.validate()

if (__name__=='__main__'):
    climb()


#This class was meant to train the model using GA, but it was never used.
class Island:
    def __init__(self, population_size, offspring_size, mutations):
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.mutations = mutations
        self.population = [StateReward() for _ in range(population_size)]
        self.best_error = -100000
        

    def get_state_sample(self):
        sample = []
        for i in range(9,16):
            for k in STATES[str(i)].keys():
                set_ = STATES[str(i)][k]
                if (len(set_)>SAMPLE_TARGET):
                    sample.extend(random.sample(set_, SAMPLE_TARGET))
                else:
                    sample.extend(set_)
        self.sample = sample

    def value_individual(self,individual):
        error = 0
        for state in self.sample:
            reward = individual.get_reward(state)
            error += (reward - state[3])**2
        return error

    def new_gen(self):
        self.get_state_sample()
        #Mutate and crossover to get offspring_size individuals!
        new_p = []
        for _ in range(self.offspring_size):
            p = StateReward.crossover(random.choice(self.population), random.choice(self.population))
            p.random_mutations(random.randint(1,self.mutations))
            new_p.append((self.value_individual(p), p))
        new_p.sort()
        self.population = [x[1] for x in new_p[:self.population_size]]

    def tsunami(self, survivors):
        self.population =self.population[:survivors] + [StateReward() for _ in range(self.population_size-survivors)]

    def get_best_performer_error(self):
        return self.value_individual(self.population[0])

