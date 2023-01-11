from collapse_state import *
import random 
import numpy as np

def test_pawns():
    for i in range(0,16):
        p = dict()
        p[0] = set([ x for x in range(0,16) if n_bit_in_common(i, x)==0 and x != i ])
        p[1] = set([ x for x in range(0,16) if n_bit_in_common(i, x)==1 and x != i])
        p[2] = set([ x for x in range(0,16) if n_bit_in_common(i, x)==2 and x != i])
        p[3] = set([ x for x in range(0,16) if n_bit_in_common(i, x)==3 and x != i])
        p[4] = set([ x for x in range(0,16) if n_bit_in_common(i, x)==4 and x != i])
        assert pawns[i][0] == p[0]
        assert pawns[i][1] == p[1]
        assert pawns[i][2] == p[2]
        assert pawns[i][3] == p[3]
        assert pawns[i][4] == p[4]


def test_n_bit_in_common():
    assert n_bit_in_common(1, 0) == 3
    assert n_bit_in_common(1, 1) == 4
    assert n_bit_in_common(15, 0) == 0
    assert n_bit_in_common(7, 8) == 0
    assert n_bit_in_common(7, 15) == 3
    assert n_bit_in_common(7, 0) == 1
    assert n_bit_in_common(5, 0) == 2

def test_flat_pawns():
    def actual_test(n):
        #generate random set of pawns
        p = set([x for x in range(0,16)])
        pawns = []
        for _ in range(n):
            t = (random.sample(p, 1)[0])
            pawns.append(t)
            p.remove(t)
        pawns = np.array(pawns)
        #compute relationship between them (matrix)
        r = [[]]*n
        for i in range(0,n):
            r[i] = [-1]*n
            for j in range(0,n): 
                r[i][j] = n_bit_in_common(pawns[i], pawns[j])
        #Flat
        pawns[-1] = flat_pawns(pawns[:-1], pawns[len(pawns)-1])
        #Check that relationship is still the same
        for i in range(0,n):
            for j in range(0,n): 
                assert n_bit_in_common(pawns[i], pawns[j]) == r[i][j]
    
    actual_test(16)

    
test_pawns()
test_n_bit_in_common()
for _ in range(0,5):
    test_flat_pawns()

