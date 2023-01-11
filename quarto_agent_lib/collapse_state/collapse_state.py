import numpy as np

"""Utility that collapses Quarto's states.
    >Collapse: collapse states by rotation and flips of the matrix
    >Flat_Pawns: collapse states by flatting the pawns used while preserving their relationships"""

"""State: np array of 16 elements:

   0        0
  0 0       1, 2
 0 0 0      3, 4, 5
0 0 0 0     6, 7, 8, 9
 0 0 0      10, 11, 12
  0 0       13, 14
   0        15
"""


def rotate(state):
    s = np.copy(state)
    temp = [s[0], s[2], s[5],s[9]]
    s[9] = s[0]; s[5] = s[1]; s[2]=s[3]; s[0] = s[6]
    s[0] = s[6]; s[1] = s[10]; s[3] = s[13]; s[6] = s[15]
    s[6] = s[15]; s[10] = s[14]; s[13] = s[12]; s[15] = s[9]
    s[9]=temp[0]; s[12]=temp[1]; s[14]=temp[2]; s[15]=temp[3]
    t = s[4]; s[4] = s[7]; s[7] = s[11]; s[11] = s[8]; s[8] = t
    return s


def switch(state,i ,j):
    temp = state[i]
    state[i]= state[j]
    state[j] = temp 
def flip_v(state):
    state = np.copy(state)
    switch(state, 3, 10)
    switch(state, 13, 1)
    switch(state, 15, 0)
    switch(state, 11, 4)
    switch(state, 14, 2)
    switch(state, 12, 5)
    return state 
def flip_h(state):
    state = np.copy(state)
    switch(state, 1, 2)
    switch(state, 3, 5)
    switch(state, 6, 9)
    switch(state, 7, 8)
    switch(state, 10, 12)
    switch(state, 13, 14)
    return state
def sum(state):
    s = 0
    for index, item in enumerate(state):
        it = int(item) #prevent overflow
        s += (it * (17**(index)))
    return s

#Pre processed stuff - for each pawn, the sets of pawns with 0,1,2 or 3 features in common is stored.
pawns = dict()
for i in range(0, 16):
    p = dict()
    p[0] = set([~(i)& 15]) #Negation
    p[1] = set([(~(i) ^ (1<<x)) & 15 for x in range(0,4)]) #1 element in common -> nego e faccio lo switch di 1 bit tra 0..3
    p[3] = set([((i) ^ (1<<x)) & 15 for x in range(0,4)])  #3 elements in common -> Switch di un bit tra 0..3
    p[2] = set([(i) ^ (x) & 15 for x in [3, 5, 6, 9, 10, 12]]) #Due elementi comuni e due negati -> Switch di bit 0011,0101,0110,1001,1010,1100
    p[4] = set() #No other pawn is equal
    pawns[i] = p

#Get number of features in common between two pawns
def n_bit_in_common(n1, n2):
    n3 = (~(n1 ^ n2)) & 15
    count = 0
    for bit in str(bin(n3)):
        if (bit=='1'): count+=1
    return count

all_elems = set([x for x in range(16)])

def flat_pawns_recursive(elems, dipendenze, posizione):
    if (posizione>=len(dipendenze)):
        return [elems] 
    allowed_set = all_elems - set(elems)
    for i in range(0, len(elems)):
        n = dipendenze[i][posizione]
        p = [j for j in range(16) if n_bit_in_common(elems[i], j)==n]
        allowed_set = allowed_set.intersection(set(pawns[elems[i]][n]))
    allowed_set = list(allowed_set)
    allowed_set.sort()
    for item in allowed_set:
        p = flat_pawns_recursive(elems + [item,], dipendenze, posizione+1)
        if (p!=None): return p
    return None

def flat_pawns(state, pawn):
    elems = np.array([x for x in state if x != -1])
    state_length = len(elems)
    remaining = all_elems - set(elems) 
    if (pawn != None):
        remaining.remove(pawn)
        elems = np.append(elems, [pawn])
    remaining = list(remaining)
    if (len(remaining)>0):
        elems = np.append(elems, remaining)

    #Builds dependency matrix
    dip = []
    for i in range(16):
        dip.append([ n_bit_in_common(elems[i], elems[x]) for x in range(16) ])
    #Call the recursive function that makes the flattening
    elems = flat_pawns_recursive([], dip, 0)

    #Ugly messing around with the result again
    j = 0
    for i in range(len(state)):
        if (state[i]!=-1):
            state[i] = elems[0][j]
            j+=1
    if (pawn != None):
        return elems[0][state_length]
    return None

def collapse(state, pawn):
    pawn = flat_pawns(state, pawn)
    states = []
    s1 = state; s2 = flip_v(state)
    states.append(s1)
    states.append(s2)
    for _ in range(0, 4):
        s1 = rotate(s1); s2 = rotate(s2)
        states.append(s1)
        states.append(s2)
    states.sort(key = sum)
    return (max(states, key = sum), pawn)
