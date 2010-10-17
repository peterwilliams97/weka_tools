from __future__ import division
"""
A module to find the subset of attributes that give the best prediction for 
specified classification algorithms and parameters.

Created on 27/09/2010

@author: peter
"""

import sys, os, random, math, time, csv, preprocess_soybeans, weka_classifiers as WC, arff, ga, misc

verbose = False
show_results = False
show_scores = False

# Multi-threaded operation to take advantange of multiple CPU cores
MULTI_THREADED = False

# The column containing the class
class_index = 0

valid_extensions = ['arff', 'csv', 'results']
def makeFileName(base_path, algo_key, subset_size, ext):
    assert(ext in valid_extensions)
    base_filename = os.path.basename(base_path)
    name = '_%s' % os.path.splitext(base_filename)[0]
    algo = '_%s' % algo_key if algo_key else ''
    subset = '_%02d' % subset_size if subset_size >= 0 else ''
    output_name = 'search%s%s%s.%s' % (name, algo, subset, ext)
    return os.path.join(output_dir, output_name)

def writeArffForInclusiveSubset(filename, data, attributes, subset):
    num_attrs = len(attributes)
    assert(len(subset) <= num_attrs)
    for d in data:
        assert(len(d) == num_attrs)
    attrs_subset = [attributes[i] for i in range(num_attrs) if i in subset]
    data_subset = [[d[i] for i in range(num_attrs) if i in subset] for d in data]
    arff.writeArff(filename, None, 'find_best_attr', attrs_subset, data_subset)

def getAccuracyForInclusiveSubset(algo_key, data, attributes, subset):
    """ Return accuracy for algorithm with name <algo_key> on <data> and
        attributes <attributes> for the attribute subset <subset>
    """
    training_filename = makeFileName('find-best-attr', algo_key, None, 'arff')
    writeArffForInclusiveSubset(training_filename, data, attributes, subset)
    result = WC.getAccuracyAlgoKey(algo_key, class_index, training_filename)
    misc.rm(training_filename)
    return result

def getRandomExcluding(num_values, exclusion_set):
    """ Return a random integer from 0 to <num_values>-1 excluding numbers in <exclusion_set> """
    while True:
        n = random.randint(0, num_values -1)
        if n not in exclusion_set:
            return n

def getInclusiveSubset(attributes, exclusive_subset):
    """ Return inclusive subset corresponding to <exlcusive_subset> """
    #print 'getInclusiveSubset', len(attributes), exclusive_subset
    assert(class_index not in exclusive_subset)
    return [i for i in range(len(attributes)) if i not in exclusive_subset]

def getSubsetResultDict(algo_key, data, attributes, exclusive_subset):
    """ Returns a dict that shows results of running <algo_key> classfier on
        <data> for (complement of <exclusive_subset>) <attributes) """
    inclusive_subset = getInclusiveSubset(attributes, exclusive_subset)
    accuracy, eval = getAccuracyForInclusiveSubset(algo_key, data, attributes, inclusive_subset)
    result = {'subset':exclusive_subset, 'score':accuracy, 'eval': eval}
    if verbose or show_scores:
        print 'getSubsetResultDict =>', result['score'], result['subset']
    return result

def getSubsetResultDictList(algo_key, data, attributes, exclusive_subset_list):
    """ Returns a list of dicts that shows results of running <algo_key> classfier on
        <data> for (complement of x) <attributes) for x in  <exclusive_subset_list>
    """
    assert(None not in exclusive_subset_list)
    number = len(exclusive_subset_list)
    inclusive_subset_list = [getInclusiveSubset(attributes, subset) for subset in exclusive_subset_list]

    training_filename_list = [makeFileName('find-best-attr%02d'% i, algo_key, None, 'arff') for i in range(number)]
    for i in range(number):
        writeArffForInclusiveSubset(training_filename_list[i], data, attributes, inclusive_subset_list[i])

    score_eval_list = WC.getAccuracyAlgoKeyList(algo_key, class_index, training_filename_list)
    #print 'score_eval_list', score_eval_list

    for training_filename in training_filename_list:
        misc.rm(training_filename)

    result_list = [{'subset':exclusive_subset_list[i], 'score':score_eval_list[i][0], 'eval': score_eval_list[i][1]} for i in range(number)]
    if verbose or show_scores:
        for result in result_list:
            print 'getSubsetResultDict =>', result['score'], result['subset']
    return result_list

def getCsvResultHeader():
    """ Return header for results file """
    return ['num_attrs', 'accuracy', 'excluded_attributes']

def getCsvResultRow(result, attributes):
    """ Return a row of the results file """
    assert(isinstance(result['score'], float))
    #-1 is for the class attribute
    num_attrs = str(len(attributes) -1 - len(result['subset']))
    included_attributes = ';'.join([attributes[i]['name'] for i in result['subset']])
    accuracy = '%.03f' % (result['score']/100.0)
    return [num_attrs, accuracy, included_attributes]

candidates_per_round = 100

def findBestAttributesForSubsetSize(base_filename, algo_key, data, attributes, previous_results, subset_size):
    """ One round. Start with best <n-1> results and use these to seed the <n> round 
        <previous_results> are <n-1> results
    """
    num_attrs = len(attributes)
    #num_random_samples = 20
    results = []
    existing_subsets = []
    history_of_best = []

    print base_filename, algo_key, len(attributes), len(previous_results), subset_size

    def addSubsetList(subset_list):
        subset_list = misc.removeDuplicates([subset for subset in subset_list if subset and subset not in existing_subsets])
        for subset in subset_list:
            assert(subset not in existing_subsets)
        if len(subset_list):
            r_list = getSubsetResultDictList(algo_key, data, attributes, subset_list)
            for r in r_list:
                results.append(r)
                existing_subsets.append(r['subset'])
                #print '***r subset=', r['subset'], 'score=', r['score']
            results.sort(key = lambda x: -x['score'])

    def addSubset(subset):
        if MULTI_THREADED:
            addSubsetList([subset])
        else:
            if subset:
                if not subset in existing_subsets:
                    r = getSubsetResultDict(algo_key, data, attributes, subset)
                    results.append(r)
                    existing_subsets.append(subset)
                    results.sort(key = lambda x: -x['score'])

    for s in previous_results:
        assert(len(s['subset']) == subset_size - 1)

    previous_best = sorted(previous_results[:candidates_per_round], key = lambda x: x['score'])

    """ Populate subsets with previous best plus other elements """
    subsets = []
    done = False
    for r in previous_best:
        for i in range(len(attributes)):
            if not i in r['subset'] and i != class_index:
                s = sorted(set(r['subset'][:] + [i]))
            #s = sorted(set(r['subset'][:] + [getRandomExcluding(num_attrs, r['subset'] + [class_index])]))
                assert(len(s) == subset_size)
                if not s in subsets:
                    subsets.append(s)
                    if len(subsets) >= candidates_per_round:
                        done = True
                        break
        if done:
            break

    # Compute scores for initial population
    superset = []
    for subset in subsets:
        addSubset(subset)
        superset = list(set(superset + subset))
    # print 'superset', len(superset), sorted(superset)
    assert(len(superset) == len(attributes)-1)
    
    if verbose:
        print 'subsets', subsets

    if subset_size > 1:
        counters = [0]*20
        for cnt in range(1000):
            found = False
            for j in range(1000):
                i1,i2 = ga.spinRouletteWheelTwice(results)
                c1,c2 = ga.crossOver(results[i1]['subset'], results[i2]['subset'])
                if i1 < len(counters):
                    counters[i1] += 1
                if i2 < len(counters):
                    counters[i2] += 1

                if c1 in existing_subsets:
                    c1 = None
                if c2 in existing_subsets:
                    c2 = None
                if c1 or c2:
                    found = True
                    break

            if not found:
                c1 = None 
                c2 = None
                for j in range(1000):
                    i1,i2 = ga.spinRouletteWheelTwice(results)
                    c1 = list(set(results[i1]['subset'] + results[i2]['subset']))
                    random.shuffle(c1)
                    c1 = sorted(c1[:subset_size])
                    if not c1 in existing_subsets:
                        found = True
                        if verbose:
                            print '***', c1
                        break

            if not found:
                print '1. Converged after', cnt, 'GA rounds'
                if verbose:
                    print existing_subsets
                    print 'counters', counters
                    for i,n in enumerate(counters):
                        print '%2d: %4d' % (i,n), results[i]['subset'], results[i]['score']
                #exit()
                break

            if MULTI_THREADED:
                addSubsetList([c1,c2])
            else:
                addSubset(c1)
                addSubset(c2)

            # Test for convergence. Top <convergence_number> scores are the sme
            convergence_number = 10
            if verbose or show_results:
                print ['%.1f%%' % x['score'] for x in results[:10]]

            history_of_best.append(results[0]['score'])
            if verbose:
                print 'history_of_best:', results[0]['score'], '=>', history_of_best[:10]
            history_of_best.sort(key = lambda x: -x)
            if len(history_of_best) >= convergence_number:
                if verbose:
                    print 'history_of_best =', history_of_best[:10]
                converged = True
                for i in range(1, convergence_number):
                    if history_of_best[i] != history_of_best[0]:
                        converged = False
                        break
                if converged:
                    print '2**. Converged after', cnt, 'GA rounds'
                    break

    best_arff = makeFileName(base_filename, algo_key, subset_size, 'arff')
    best_results = makeFileName(base_filename, algo_key, subset_size, 'results')
    best_subset = results[0]['subset']

    best_inclusive_subset = getInclusiveSubset(attributes, best_subset)
    writeArffForInclusiveSubset(best_arff, data, attributes, best_inclusive_subset)

    file(best_results, 'w').write(results[0]['eval'])
    print 'Results: WEIGHT_RATIO', ga.WEIGHT_RATIO, 'candidates_per_round', candidates_per_round
    print 'accuracy =', '%.1f%%' % results[0]['score']
    print 'best 10  =', ['%.1f%%' % x['score'] for x in results[:10]]
    print 'exclusive subset =', best_subset
    print '       =', [attributes[i]['name'] for i in best_subset]
    print 'inclusive subset =', best_inclusive_subset
    print '       =', [attributes[i]['name'] for i in best_inclusive_subset]
    print '-------------------------------------------------------------'
    return results[:candidates_per_round]
 
def findBestAttributes(base_filename, algo_key, data, attributes):
    num_attrs = len(attributes) - 1

    # Track results for each round
    series_results = []

    # Loop through all sizes of subsets of attributes, largest first
    for subset_size in range(num_attrs):
        if subset_size == 0:
            results = [getSubsetResultDict(algo_key, data, attributes, [])]
        else:
            results = findBestAttributesForSubsetSize(base_filename, algo_key, data, attributes, results, subset_size)
        results[0]['num_attrs'] = num_attrs-subset_size
        series_results.append(results[0])

        # Write out the results
        out_filename = makeFileName(base_filename, algo_key, -1, 'csv')
        header = getCsvResultHeader()
        results_matrix = [getCsvResultRow(r, attributes) for r in series_results]
        csv.writeCsv(out_filename, results_matrix, header)
    return series_results


if __name__ == '__main__':
    # Set random seed so that each run gives same results
    random.seed(555)

    global output_dir

    WC.initWekaAlgoThreads(2)

    if len(sys.argv) < 3:
        print 'Usage: jython find_best_attributes.py  <arff-file> <output-dir>'
        sys.exit()

    filename = sys.argv[1]
    output_dir = sys.argv[2]

    misc.mkDir(output_dir)

    relation, comments, attributes, data = arff.readArff(filename)

    print 'Algorithms to test:', WC.all_algo_keys
 
    for algo_key in WC.all_algo_keys:
        print '======================= findBestAttributes:', filename, algo_key 
        findBestAttributes(filename, algo_key, data, attributes)

    WC.killWekaAlgoThreads()
 