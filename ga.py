from __future__ import division
"""
A genetic algorithm implementation

Created on 27/09/2010

@author: peter
"""

import sys, os, random
from math import *

# Set random seed so that each run gives same results
random.seed(555)

WEIGHT_RATIO = 0.8

def applyWeights(roulette_in):
    """ Add and 'idx' and 'weight' keys to roulette dicts 
        Weights are based on 'score' keys.
        For use in rouletteWheel 
    """
    # First store the original order in 'idx' key
    for i,x in enumerate(roulette_in):
        x['idx'] = i
    # Then sort by score and set weight based on order
    roulette = sorted(roulette_in, key = lambda x: -x['score'])
    for i,x in enumerate(roulette):
        x['weight'] = WEIGHT_RATIO**(i+1)
    total = float(sum([x['weight'] for x in roulette]))
    for x in roulette:
        x['weight'] = x['weight']/total
    if False:
        print 'len(roulette) =', len(roulette)
        print 'sum weights =', sum([x['weight'] for x in roulette])
    if abs(sum([x['weight'] for x in roulette]) - 1.0) >= 1e-6:
        print len(roulette), roulette
        print sum([x['weight'] for x in roulette])
        print sum([x['weight'] for x in roulette]) -1.0
    assert(abs(sum([x['weight'] for x in roulette]) - 1.0) < 1e-6)
    return sorted(roulette, key = lambda x: -x['weight'])

def spinRouletteWheel(roulette_in):
    """ Find the roulette wheel winner
        <roulette> is a list of dicts with keys 'idx' and 'weight'
        Returns an index with probability proportional to dict's 'weight'
    """
    roulette = applyWeights(roulette_in)
    v = random.random()
    base = 0.0
    for x in roulette:
        top = base + float(x['weight'])
        if v <= top:
            return x['idx']
        base = top
    raise RuntimeError('Cannot be here')

def spinRouletteWheelTwice(roulette):
    """" Spin the roulette wheel twice and return 2 different values """
    while True:
        i1 = spinRouletteWheel(roulette)
        i2 = spinRouletteWheel(roulette)
        if i2 != i1:
            return (i1,i2)

def makeShuffleList(size, max_val):
    """ Return a list of <size> unique random values in range [0,<max_val>) """
    assert(size <= max_val)
    shuffle_list = []
    while len(shuffle_list) < size:
        i = random.randrange(max_val)
        if not i in shuffle_list:
            shuffle_list.append(i)
    return shuffle_list

def mutate(split):
    if False:
        shuffle_list = []
        while len(shuffle_list) < len(split)//20:
            i = random.randrange(len(split))
            if not i in shuffle_list:
                shuffle_list.append(i)

    shuffle_list = makeShuffleList(len(split)//20, len(split))
    swaps = [(shuffle_list[i*2],shuffle_list[i*2+1]) for i in range(len(shuffle_list)//2)]

    out = split[:]
    for s in swaps:
        x = out[s[0]]
        out[s[0]] = out[s[1]]
        out[s[1]] = x
    return out

def crossOver_(c1, c2):
    """ Swap half the elements in c1 and c2 """
    assert(len(c1) == len(c2))
    assert(len(c1) > 0)
    assert(len(c2) > 0)
    n = len(c1)

    # Find elements that are not in both lists
    d1 = sorted(c1, key = lambda x: x in c2)
    d2 = sorted(c2, key = lambda x: x in c1)
    for i1,x in enumerate(d1):
        if x in d2:
            break
    for i2,x in enumerate(d2):
        if x in d1:
            break
    m = min(i1, i2)  # number of non-shared elements

    shuffle_list = makeShuffleList(2*(m//2), 2*(m//2))
    swaps = [(shuffle_list[i*2],shuffle_list[i*2+1]) for i in range(len(shuffle_list)//2)]

    for i,s in enumerate(swaps):
        assert(s[0] < 2* len(swaps))
        assert(s[1] < 2* len(swaps))
        d1[s[0]], d2[s[1]] = d2[s[1]], d1[s[0]] 

    return (sorted(d1), sorted(d2))

def crossOverDist(c1, c2, class_distribution):
    assert(c1 != c2)
    if class_distribution == None:
        return crossOver_(c1,c2)
    d1 = []
    d2 = []
    for v in class_distribution.values():
        c1v = [i for i in c1 if v['start'] <= i and i < v['end']]
        c2v = [i for i in c2 if v['start'] <= i and i < v['end']]
        d1v, d2v = crossOver_(c1v, c2v)
        d1 = d1 + d1v
        d2 = d2 + d2v
    return (sorted(d1), sorted(d2))

def crossOver(c1, c2, class_distribution = None):
    if sorted(c1) == sorted(c2):
        print 'crossOver(): c1,c2 =', sorted(c1), sorted(c2)
        assert(sorted(c1) != sorted(c2), 'CrossOver: %s, %s' % (str(sorted(c1)), str(sorted(c2))) )
    for i in range(1000):
        d1,d2 = crossOverDist(c1, c2, class_distribution)
        if d1 != d2:
            return d1,d2
    print 'crossOver() failed: c1,c2 = ', sorted(c1), sorted(c2)
    raise RuntimeError('Cannot be here')

