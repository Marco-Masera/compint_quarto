import time
#Collection of utils 
diagonals=[(0, 1, 3, 6), (2, 4, 7, 10), (5,8, 11, 13), (9, 12, 14, 15),
        (6, 10, 13, 15), (3, 7, 11, 14), (1, 4, 8, 12), (0, 2, 5, 9),
        (6, 7, 8, 9), (0, 4, 11, 15)]


        

def checkState(state): # -> (isChessboardFull, isWinning)
    global diagonals
    full = True
    for t in diagonals:
        and_ = (~0)&15
        or_ = 0
        for c in t:
            if (state[c] != -1):
                and_ = and_ & state[c]
                or_ = or_ | state[c]
            else:
                and_ = 0
                or_ = 15
                full = False
        if (and_ != 0 or or_ != 15):
            return (True, True)
    return (full, False)
    

def bits_in_common_multiple(list_, list2 = None, last = None, acc = 15):
    for elem in list_:
        if (last != None):
            acc = acc & (~(last ^ elem))
        last = elem 
    if (list2!=None):
        last = None
        for elem in list2:
            if (last != None):
                acc = acc & (~(last ^ elem))
            last = elem 
    return acc