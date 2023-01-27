import numpy as np
import json
import random
import copy
from quarto_agent_lib.quarto_utils import checkState

#Init some static stuff to calculate the features faster during runtime
LINES = [
    # ( (caselle),[(quale_riga_tocca, in quale casella )])
    # ( (caselle),[(quale_riga_tocca, in quale casella )])
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

LINES_REVERSED = [] 
for i in range(16):
    l = []
    for index, line in enumerate(LINES):
        for box in line[0]:
            if (i == box):
                l.append(index)
                break
    LINES_REVERSED.append(l)

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
        with open("quarto_agent_lib/training/reward_genome.json", 'r') as source:
            genome = json.load(source)
        g = dict()
        for key in genome.keys():
            g[int(key)] = np.array(genome[key])
        return g

    def count_state_size(state):
        c = 0
        for i in state: 
            if i>-1: c+=1
        return c
    
    def solve_last_move(self,state):
        st = copy.deepcopy(state)
        for index,box in enumerate(st[0]):
            if (box==-1): st[0][index] = st[1]
        full, winning = checkState(st[0])
        if (winning):
            return 180
        return 0

    def get_reward_from_truth_value(self,truth_value, state_length):
        reward = np.matmul(self.genome[state_length],truth_value)
        if (reward > 0):
            return min(reward,140)
        else:
            return max(-140, reward)

    def update_lines(self, state, mylines, index_):
        global LINES; global LINES_PRECEDING; global MULTILINE_BLOCK;
        chessboard = state[0]
        #Updates only the changed line
        for changed in LINES_REVERSED[index_]:
            count = 0; acc = 15; last = None
            for box in LINES[changed][0]:
                if (chessboard[box]!=-1):
                    count += 1
                    if (last != None):
                        acc = acc & (~(chessboard[box] ^ last))
                    last = chessboard[box]
            #updates single line
            active = acc!=0
            if (not active):
                last = 0
            elif (last==None):
                last = -1
            mylines[changed] = (active, acc, last, count)
        return mylines


    def get_reward(self, state, size = None): #State = ([chessboard], assigned_pawn, set[remaining])
        global LINES; global LINES_PRECEDING; global MULTILINE_BLOCK;
        
        if (size == None):
            size = StateReward.count_state_size(state[0])  
        if (size==15):
            return self.solve_last_move(state)
        if (size < StateReward.MIN_SIZE):
            return 0
            
        genome = self.genome[size]
        chessboard = state[0]
        
        mylines = []
        self.truth_value = np.array([0 for _ in range(StateReward.state_length)])
        not_full = False
        #mylines = [] -> (Active, acc)
        for index,line in enumerate(LINES):
            count = 0; acc = 15; last = None
            for box in line[0]:
                if (chessboard[box]!=-1):
                    count += 1
                    if (last != None):
                        acc = acc & (~(chessboard[box] ^ last))
                    last = chessboard[box]
            if (count < 4):
                not_full = True
            if (acc == 0):
                mylines.append((False, acc, 0, count))
                continue
            if (count==4):
                #I already lost
                return -180
            if (last==None):
                last = -1
            mylines.append((True, acc, last, count))
        mylines = np.array(mylines)
        if (not_full==False):
            return 0 #Tie
        
        for index,line in enumerate(LINES):
            active, acc, last, count = mylines[index]
            #can extend, can block -> computed from pawn
            if (state[1]!=None):
                if (count != 0 and acc & (~(state[1] ^ last))):#(can_extend):
                    self.truth_value[(index*8) + count] = 1
                    if (count==3):
                        #If a line has 3 pawns and I can extend it, I won
                        return 180
                else: # (can_block):
                    self.truth_value[(index*8) + count + 4] = 1

            #Is there a pawn to extend or block the line
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

        #Second round - line x connected lines
        for index,line in enumerate(LINES):
            if (mylines[index][0] == False): continue
            inx = 80 + (LINES_PRECEDING[index] * 6)
            for common in line[1]:
                if (mylines[common[0]][0] == False or chessboard[common[1]] != -1): continue
                #Checks if can block both, increase both, or one and one
                acc = mylines[index][1]; acc2 = mylines[common[0]][1]
                if (state[1]!=None):
                    can_extend_one = (acc & (~(state[1] ^ mylines[index][2])))!=0
                    can_extend_two = (acc2 & (~(state[1] ^ mylines[common[0]][2])))!=0
                    if (can_extend_one and can_extend_two):
                        self.truth_value[inx] = 1
                    elif (can_extend_one != can_extend_two):
                        self.truth_value[inx + 1] = 2
                    else:
                        self.truth_value[inx + 2] = 3
                #Check if i can block the adversary and if he can block me
                canEO = 0; canBO = 0; canM = 0
                inx = 80 + MULTILINE_BLOCK + (LINES_PRECEDING[index] * 6)
                for pawn in state[2]:
                    can_extend_one = (acc & (~(pawn ^ mylines[index][2])))!=0
                    can_extend_two = (acc2 & (~(pawn ^ mylines[common[0]][2])))!=0
                    if (can_extend_one and can_extend_two):
                        canEO += 1
                        if (canEO==1):
                            self.truth_value[inx] = 1
                        else:
                            self.truth_value[inx + 3] = 1
                    elif (can_extend_one != can_extend_two):
                        canBO += 1
                        if (canBO==1):
                            self.truth_value[inx + 1] = 1
                        else:
                            self.truth_value[inx + 4] = 1
                    else:
                        canM += 1
                        if (canM==1):
                            self.truth_value[inx + 2] = 1
                        else:
                            self.truth_value[inx + 5] = 1
                    if (canEO >= 2 and canBO >= 2 and canM >= 2): break

        self.last_lines = mylines
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


