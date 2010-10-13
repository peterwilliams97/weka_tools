from __future__ import division
"""
A module to split a data set in training and test sets.

1. Finds the splits that give best or worst results in WEKA using "Supplied test 
   set" for a set of classification algorithms that can be found in the code below.
2. Preserves class distribution on original data set.
3. Originally conceived as a way of achieving a split that was fair to all the 
   classification algorithms.

Requires Jython and the WEKA java library weka.jar (see http://bit.ly/weka_tools)

Based on http://www.btbytes.com/2005/11/30/weka-j48-classifier-example-using-jython/

Note: needs WEKA 3.6.x to run (due to changes in the weka.classifiers.Evaluation class)

Created on 22/09/2010

@author: peter
"""

import sys, os, random
from math import *

import java.io.FileReader as FileReader
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean

import weka.core.Instances as Instances
import weka.classifiers.Evaluation as Evaluation
import weka.core.Range as Range

import weka.classifiers.bayes.NaiveBayes as NaiveBayes
import weka.classifiers.bayes.BayesNet as BayesNet
import weka.classifiers.functions.MultilayerPerceptron as MLP
import weka.classifiers.functions.SMO as SMO
import weka.classifiers.trees.J48 as J48
import weka.classifiers.trees.RandomForest as RandomForest
import weka.classifiers.rules.JRip as JRip
import weka.classifiers.lazy.KStar as KStar
import weka.classifiers.meta.MultiBoostAB as MultiBoost

import arff

# The column containing the class
class_index = 0

# Extra output?
verbose = False

# If true then try to find worst performer, else try to find best performer
do_worst = False

def getMultiBoost():
    multi_boost = MultiBoost()
    multi_boost.setOptions(['-W weka.classifiers.functions.SMO'])
    return multi_boost

algo_val_name_all = [(NaiveBayes(), 'NaiveBayes'), (BayesNet(),'BayesNet'), (J48(),'J48'), (JRip(), 'JRip'),
                        (KStar(), 'KStar'), (RandomForest(), 'RandomForest'), (SMO(),'SMO'), (MLP(),'MLP')]
algo_val_name = algo_val_name_all
algo_list = [a[0] for a in algo_val_name]

def runClassifierAlgo(algo, training_filename, test_filename, do_model, do_eval, do_predict):
    """ Run classifier algorithm <algo> on training data in <training_filename> to build a model
        then run in on data in <test_filename> (equivalent of WEKA "Supplied test set") """
    training_file = FileReader(training_filename)
    training_data = Instances(training_file)
    test_file = FileReader(test_filename)
    test_data = Instances(test_file)

   # set the class Index - the index of the dependent variable
    training_data.setClassIndex(class_index)
    test_data.setClassIndex(class_index)

    # create the model
    algo.buildClassifier(training_data)

    evaluation = None
    # only a trained classifier can be evaluated
    if do_eval or do_predict:
        evaluation = Evaluation(test_data)
        buffer = StringBuffer()             # buffer for the predictions
        attRange = Range()                  # no additional attributes output
        outputDistribution = Boolean(False) # we don't want distribution
        evaluation.evaluateModel(algo, test_data, [buffer, attRange, outputDistribution])

    if verbose:
        if do_model:
            print '--> Generated model:\n'
            print algo.toString()
        if do_eval:
            print '--> Evaluation:\n'
            print evaluation.toSummaryString()
        if do_predict:
            print '--> Predictions:\n'
            print buffer

    return {'model':str(algo), 'eval':str(evaluation.toSummaryString()), 'predict':str(buffer) }

def getEvalAlgo(algo, training_filename, test_filename):
    """ Returns evaluation string for algorithm <algo> built from training data in 
        <training_filename> and tested on data in <test_filename> """ 
    result = runClassifierAlgo(algo, training_filename, test_filename, False, True, False)
    return result['eval'].strip()

classify_tag = 'Correctly Classified Instances'

def getAccuracyAlgo(algo, training_filename, test_filename):
    """ Returns accuracy of algorithm <algo> built from training data in <training_filename> 
        and tested on data in <test_filename> """ 
    lines = getEvalAlgo(algo, training_filename, test_filename).split('\n')
    for ln in lines:
        if classify_tag in ln:
            contents = ln[len(classify_tag):]
            parts = [x.strip() for x in contents.strip().split(' ') if len(x) > 0]
            assert(len(parts) == 3)
            accuracy = float(parts[1])
            return accuracy
    raise ValueException('Cannot be here')

def getAccuracy(training_filename, test_filename):
    if do_worst:
        return len(algo_list)*100.0 - sum([getAccuracyAlgo(algo, training_filename, test_filename) for algo in algo_list])
    else:
        return  sum([getAccuracyAlgo(algo, training_filename, test_filename) for algo in algo_list])

training_file_base = '.train.arff'
test_file_base = '.test.arff'

def showSplit(split_vector):
    """ Returns a string showing a list of booleans """
    return ''.join(['|' if a else '.' for a in split_vector[:120]])

def makeTrainingTestSplit(base_data, split_vector, prefix):
    """ Split <base_data> into training and test data sets. Rows with indexes in 
        <split_vector> go into training file and remaining go into test file.
        Writes training and test .arff files and returns their names. 
        File names are prefixed with <prefix> """
    assert(len(base_data) == len(split_vector))
    num_instances = len(base_data)

    training_file_name = prefix + training_file_base
    test_file_name = prefix + test_file_base

    training_data = []
    test_data = []
    for i,x in enumerate(base_data):
        if split_vector[i]:
            test_data.append(x)
        else:
            training_data.append(x)

    arff.writeArff(training_file_name, base_comments, base_relation, base_attrs, training_data)
    arff.writeArff(test_file_name, base_comments, base_relation, base_attrs, test_data)
    return (training_file_name, test_file_name)

def rm(filename):
    try:
        os.remove(filename)
    except:
        pass

def getAccuracyForSplit(base_data, split_vector):
    """ Split <base_data> into training and test data sets. Rows with indexes in 
        <split_vector> go into training file and remaining go into test file.
        Run prediction and return accuracy 
    """
    training_filename, test_filename = makeTrainingTestSplit(base_data, split_vector, 'temp')
    accuracy = getAccuracy(training_filename, test_filename)
    rm(training_filename)
    rm(test_filename)
    return accuracy

def getRandomSplit(class_distribution):
    """ Return a random split with the same distribution as <class_distribution> """
    split_vector = []
    for k,v in sorted(class_distribution.items()):
        part_vector = [(x < v['num_test']) for x in range(v['num'])]
        random.shuffle(part_vector)
        split_vector = split_vector + part_vector
    return split_vector

def getClassDistribution(data):
    """ Returns the number of each class in the <data> set
        Class is assumed to be in column <class_index> """
    classes = set([instance[class_index] for instance in data])
    class_counts = {}.fromkeys(classes, 0)
    class_distribution = {}
    for k in classes:
        class_distribution[k] = {'num':0}

    for instance in sorted(data, key = lambda x: x[class_index]):
        k = instance[class_index]
        class_distribution[k]['num'] += 1

    # Need at least 2 members. 1 for training and 1 for test  
    for k in classes:
        if class_distribution[k]['num'] < 2:
            class_distribution.pop(k)
    classes = class_distribution.keys()

    start = 0
    for k in sorted(classes):
        assert(class_distribution[k]['num'] > 0)
        class_distribution[k]['start'] = start
        class_distribution[k]['end'] = start + class_distribution[k]['num']
        start = class_distribution[k]['end']

    return class_distribution

def getClassDistributionForSplits(data, test_fraction):
    """ Return class distribution for <data> and <test_fraction> adjusted to
        make splits valid integer values """
    class_distribution = getClassDistribution(data)

    num_instances = sum([v['num'] for v in class_distribution.values()])
    num_test = int(floor(num_instances*test_fraction))
    keys = sorted(class_distribution.keys(), key = lambda k: class_distribution[k]['num'])

    # Start with fraction rounded down
    for k in keys:
        class_distribution[k]['num_test'] = max(2,int(floor(class_distribution[k]['num']*test_fraction)))
        assert(class_distribution[k]['num_test'] > 0)

    # Make whole distribution have correct fraction 
    # Start with classes with more members
    for k in keys:
        if sum([v['num_test'] for v in class_distribution.values()]) >= num_test:
            break
        class_distribution[k]['num_test'] = class_distribution[k]['num_test'] + 1
    for k in keys:
        if sum([v['num_test'] for v in class_distribution.values()]) <= num_test:
            break
        class_distribution[k]['num_test'] = class_distribution[k]['num_test'] - 1

    for k in sorted(keys):
        print '%28s' % k, class_distribution[k], '%.2f' % (class_distribution[k]['num_test']/class_distribution[k]['num'])
        assert(class_distribution[k]['num_test'] > 0)

    return class_distribution

WEIGHT_RATIO = 0.8

def applyWeights(roulette_in):
    """ Add 'idx' and 'weight' keys to roulette dicts 
        Weights are based on 'score' keys.
        For use in rouletteWheel() 
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
    return sorted(roulette, key = lambda x: -x['weight'])

def spinRouletteWheel(roulette_in):
    """ Find the roulette wheel winner
        <roulette_in> is a list of dicts with keys 'idx' and 'weight'
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
    raise ValueException('Cannot be here')

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
    """ Swap half the elements in <c1> and <c2> while preserving the distribution in <class_distribution> """
    assert(c1 != c2)
    d1 = []
    d2 = []
    for v in class_distribution.values():
        c1v = [i for i in c1 if v['start'] <= i and i < v['end']]
        c2v = [i for i in c2 if v['start'] <= i and i < v['end']]
        d1v, d2v = crossOver_(c1v, c2v)
        d1 = d1 + d1v
        d2 = d2 + d2v
    assert(len(d1) > 0)
    assert(len(d2) > 0)
    return (sorted(d1), sorted(d2))

def getIndexes(split_vector):
    return sorted([i for i,v in enumerate(split_vector) if v])

def getVector(indexes, size):
    return [i in indexes for i in range(size)]

def uniqueCrossOver_(c1, c2, class_distribution):
    """ Swap half the elements in <c1> and <c2> while preserving the distribution in 
         <class_distribution>.Return two different lists """
    for i in range(1000):
        d1,d2 = crossOverDist(c1, c2, class_distribution)
        if d1 != d2:
            return d1,d2
    raise ValueException('Cannot be here')

def crossOver(v1, v2, class_distribution):
    c1,c2 = getIndexes(v1), getIndexes(v2)
    assert(c1 != c2)
    d1,d2 = uniqueCrossOver_(c1,c2, class_distribution)
    assert(len(d1) > 0)
    assert(len(d2) > 0)
    return [getVector(d, len(v1)) for d in (d1,d2)]

def runGA(base_data, num_instances, test_fraction):
    """ Run a genetic algorithm on <base_data> with <num_instances> instances
        and split it into test set of <test_fraction> and training set of 
        1 - <test_fraction> instances 
    """
    # Create just enough to seed the set with a good coverage
    num_random_samples = 20
    results = []
    existing_splits = []
    history_of_best = []
    best_score = 0.0

    # Data needs to be sorted by class for the class_distribution splits to line up
    base_data.sort()
    class_distribution = getClassDistributionForSplits(base_data, test_fraction)
    
    # Adjust base_classes and base_data to match class distribution
    global base_classes
    base_classes = class_distribution.keys()
    base_data = [x for x in base_data if x[class_index] in class_distribution.keys()]

    # Make a dict with <split_vector> and the accuracies for the classifier for that 
    # split summed over all the classifiers above
    def getScoreDict(split_vector):
        accuracy = getAccuracyForSplit(base_data, split_vector)
        return {'split':split_vector, 'score':accuracy}

    # If <split_vector> not in the current population then score it and add it
    def addSplit(split_vector):
        if not split_vector in existing_splits:
            results.append(getScoreDict(split_vector))
            existing_splits.append(split_vector)

    # First create some random vectors
    while len(existing_splits) < num_random_samples:
        # split_vector = getRandomSplit(num_instances, test_fraction)
        print len(existing_splits), ': ',
        split_vector = getRandomSplit(class_distribution)
        accuracy = getAccuracyForSplit(base_data, split_vector)
        addSplit(split_vector)
    print

    # Write out the best result in case we crash
    results.sort(key = lambda x: -x['score'])
    test_filename, training_filename = makeTrainingTestSplit(base_data, results[class_index]['split'], 'worst' if do_worst else 'best')
    best_score = results[class_index]['score']

    # The main GA cross-over and selection loop
    for cnt in range(1000):
        found = False
        for j in range(10):
            i1,i2 = spinRouletteWheelTwice(results)
            #print (i1,i2),
            c1,c2 = crossOver(results[i1]['split'], results[i2]['split'], class_distribution)
            if c1 in existing_splits:
                c1 = None
            if c2 in existing_splits:
                c2 = None
            if c1==c2:
                c2 = None
            if c1 or c2:
                found = True
                break
        if not found:
            results.sort(key = lambda x: -x['score'])
            print ['%.1f%%' % x['score'] for x in results[:10]]
            print '1. Converged after', cnt, 'GA rounds'
            break
        if c1:
            addSplit(c1)
        if c2: 
            addSplit(c2)

        # Test for convergence
        convergence_number = 10
        results.sort(key = lambda x: -x['score'])
        print ['%.1f%%' % x['score'] for x in results[:10]]
        if results[class_index]['score'] > best_score:
            test_filename, training_filename = makeTrainingTestSplit(base_data, results[class_index]['split'], 'worst' if do_worst else 'best')
            best_score = results[class_index]['score']

        history_of_best.append(results[class_index]['score'])
        if len(history_of_best) >= convergence_number:
            converged = True
            for i in range(1, convergence_number):
                if not history_of_best[i] == history_of_best[0]:
                    converged = False
                    break
            if converged:
                print '2. Converged after', cnt, 'GA rounds'
                break

    # Make the split
    test_filename, training_filename = makeTrainingTestSplit(base_data, results[class_index]['split'], 'worst' if do_worst else 'best')

    # Run classifiers on the split and print out some summary results
    accuracy = getAccuracyForSplit(base_data, results[class_index]['split'])
    print 'Results -----------------------------------------------------'
    print 'do_worst', do_worst
    print 'WEIGHT_RATIO', WEIGHT_RATIO
    print 'num_random_samples',num_random_samples
    print 'accuracy =', accuracy 
    print showSplit(results[class_index]['split'])
    for (algo,name) in algo_val_name:
        eval = getEvalAlgo(algo, test_filename, training_filename)
        print name, '---------------------------------'
        print eval
 
if __name__ == '__main__':

    global base_attrs 
    # Set random seed so that each run gives same results
    random.seed(555)

    if not len(sys.argv) < 3:
        print 'usage: split_data.py <base-file> <test-fraction>'
        sys.exit()

    base_file = sys.argv[1]
    test_fraction = float(sys.argv[2])  # 0.2
    
    print 'input file:', base_file
    print 'test fraction:', test_fraction
    print 'algorithms:', [name for (_,name) in algo_val_name]
    print 'class index:', class_index
    print 'do_worst:', do_worst

    global base_relation
    global base_comments

    base_relation, base_comments, base_attrs, base_data = arff.readArff(base_file)

    base_data.sort()

    print 'base_data', len(base_data), len(base_data[0])
    runGA(base_data, len(base_data), test_fraction)

