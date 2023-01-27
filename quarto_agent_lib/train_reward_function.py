import numpy as np
import json
import random
import copy
from quarto_agent_lib.quarto_utils import checkState
from quarto_agent_lib.state_reward import StateReward

###   The following code handles the learning part
BASE_PATH = "quarto_agent_lib/"
STATES = [] 
VALIDATION_STATES = []
LENGTHS_TO_TRAIN = ["6","8","9","10", "11","12","13","15"]
class Climber:
    def __init__(self):
        self.individual = StateReward(genome=StateReward.load_genome())
        self.build_truth_values()
    #Builds truth value arrays for the records in the dataset.
    #It helps speeding things up - this way it don't need to be computed at each iteration
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
        with open(f"{BASE_PATH}training/export_{gen}_{num}", 'w') as exp:
            exp.write(json.dumps(to_export))


def export(climber, iteration, generation):
    with open(f"{BASE_PATH}training/result_{generation}_{iteration}", 'w') as exp:
        exp.write(json.dumps(climber.validate()))
    climber.export_self(iteration, generation)

def load_data():
    global STATES; global VALIDATION_STATES
    with open(f"{BASE_PATH}dataset/pre_processed/training_dataset.json", 'r') as source:
        STATES = json.load(source)
        for size_ in STATES.keys():
            for state in STATES[size_]:
                state[0] = np.array(state[0])
    with open(f"{BASE_PATH}dataset/pre_processed/training_dataset_v2.json", 'r') as source:
        states_2 = json.load(source)
        for size_ in states_2.keys():
            for state in states_2[size_]:
                state[0] = np.array(state[0])
            if (size_ in STATES):
                STATES[size_].extend(states_2[size_])
            else:
                STATES[size_] = states_2[size_]

    with open(f"{BASE_PATH}dataset/pre_processed/validation_dataset.json", 'r') as source:
        VALIDATION_STATES = json.load(source)
        for size_ in VALIDATION_STATES.keys():
            for state in VALIDATION_STATES[size_]:
                state[0] = np.array(state[0])
    with open(f"{BASE_PATH}dataset/pre_processed/validation_dataset_v2.json", 'r') as source:
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
    for i in range(200):
        print(f"Gen {i}")
        climber.new_gen(0.15)
    export(climber, 1,generation)
    for i in range(800):
        print(f"Gen {i}")
        climber.new_gen(0.1)
    export(climber, 2,generation)
    exp_n = 2
    for i in range(800):
        print(f"Gen {i}")
        climber.new_gen(0.07)
        
    export(climber, exp_n,generation)
    climber.print_evaluations(climber.individual, VALIDATION_STATES)
    return



#This class was meant to train the model using GA, but it was never used
SAMPLE_TARGET = 8
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

