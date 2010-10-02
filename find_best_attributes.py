from __future__ import division
"""
A module to find the subset of attributes that give the best prediction for 
specified classification algorithms and parameters.

Created on 27/09/2010

@author: peter
"""

import sys, os, random, math, csv, preprocess_soybeans, weka_classifiers as WC, arff, ga

# Set random seed so that each run gives same results
random.seed(555)

# The column containing the class
class_index = 0

def getAccuracyForSubset(algo_key, data, attributes, subset):
    num_attrs = len(attributes)
    assert(len(subset) <= num_attrs)
    for d in data:
        assert(len(d) == num_attrs)
    attrs_subset = [attributes[i] for i in range(num_attrs) if i in subset]
    data_subset = [[d[i] for i in range(num_attrs) if i in subset] for d in data]

    training_filename = 'find_best_attr.arff'
    arff.writeArff(training_filename, None, 'find_best_attr', attrs_subset, data_subset)
    return WC.getAccuracyAlgoKey(algo_key, class_index, training_filename)

def getRandomExcluding(num_values, exclusion_set):
    """ Return a random integer from 0 to <num_values>-1 excluding numbers in <exclusion_set> """
    while True:
        n = random.randint(0, num_values -1)
        if n not in exclusion_set:
            return n

valid_extensions = ['arff', 'csv']
def makeFileName(base_filename, algo_key, subset_size, ext):
    assert(ext in valid_extensions, 'Invalid file type')
    base = '_%s' % os.path.splitext(base_filename)[0]
    algo = '_%s' % algo_key if algo_key else ''
    subset = '_%02d' % subset_size if subset_size >= 0 else ''
    return 'search%s%s%s.%s' % (base, algo, subset, ext)

def getSubsetResultDict(algo_key, data, attributes, exclusive_subset):
    """ Returns a dict that shows results of running <algo_key> classfier on
        <data> for (complement of <exclusive_subset>) <attributes) """
    inclusive_subset = [i for i in range(len(attributes)) if i not in exclusive_subset]
    accuracy = getAccuracyForSubset(algo_key, data, attributes, inclusive_subset)
    result = {'subset':exclusive_subset, 'score':accuracy}
    print 'getSubsetResultDict =>', result
    return result

def getCsvResultHeader():
    return ['num_attrs', 'accuracy', 'excluded_attributes']

def getCsvResultRow(result, attributes):
    if False:
        print 'getCsvResultRow', result
        print type(result['score']), result['score']
        print type(result['subset']), result['subset']
        print type(attributes), attributes
    assert(isinstance(result['score'], float))
    num_attrs = str(len(attributes) - len(result['subset']))
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
    best_score = 0.0

    print base_filename, algo_key, len(attributes), len(previous_results), subset_size

    def addSubset(subset):
        if not subset in existing_subsets:
            r = getSubsetResultDict(algo_key, data, attributes, subset)
            results.append(r)
            existing_subsets.append(subset)
            results.sort(key = lambda x: -x['score'])
            best_score = results[0]['score']

    for s in previous_results:
        assert(len(s['subset']) == subset_size - 1)

    previous_best = sorted(previous_results[:candidates_per_round], key = lambda x: x['score'])

    """ Populate subsets with previous best plus other elements """
    subsets = []
    done = False
    for r in previous_best:
        for i in range(len(attributes)):
            s = sorted(set(r['subset'][:] + [getRandomExcluding(num_attrs, r['subset'] + [class_index])]))
            assert(len(s) == subset_size)
            if not s in subsets:
                subsets.append(s)
                if len(subsets) >= candidates_per_round:
                    done = True
                    break
        if done:
            break

    # Compute scores for initial population
    for subset in subsets:
        addSubset(subset)

    for cnt in range(1000):
        found = False
        for j in range(10):
            #print 'len(results)', len(results)
            i1,i2 = ga.spinRouletteWheelTwice(results)
            c1,c2 = ga.crossOver(results[i1]['subset'], results[i2]['subset'])
            if not c1 in existing_subsets and not c2 in existing_subsets:
                found = True
                break
        if not found:
            print '1. Converged after', cnt, 'GA rounds'
            break
        addSubset(c1)
        addSubset(c2)

        # Test for convergence. Top <convergence_number> scores are the sme
        convergence_number = 10
        print ['%.1f%%' % x['score'] for x in results[:10]]

        history_of_best.append(best_score)
        history_of_best.sort()
        if len(history_of_best) >= convergence_number:
            print 'history_of_best =', history_of_best[:10]
            converged = True
            for i in range(1, convergence_number):
                if history_of_best[i] != history_of_best[0]:
                    converged = False
                    break
            if converged:
                print '2. Converged after', cnt, 'GA rounds'
                break

    best_filename = makeFileName(base_filename, algo_key, subset_size,'arff')
    best_subset = results[0]['subset']
    best_attributes = [attributes[i] for i in best_subset]
    best_data = [[d[i] for i in best_subset] for d in data]
    arff.writeArff(best_filename, None, 'best_attr_%s_%02d' % (algo_key,subset_size), best_attributes, best_data)
    print 'Results -----------------------------------------------------'
    print 'WEIGHT_RATIO', ga.WEIGHT_RATIO, 'candidates_per_round', candidates_per_round
    print 'accuracy =', results[0]['score'] 
    print 'subset =', results[0]['subset']
    if False:
        for name in WC.algo_dict.keys():
            eval = WC.getEvalAlgoKey(name, class_index, best_filename)
            print name, '---------------------------------'
            print eval
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

    if len(sys.argv) < 2:
        print 'Usage: jython find_best_attributes.py  <arff-file>'
        sys.exit()

    filename = sys.argv[1]
    algo_key = 'BayesNet' 
    relation, comments, attributes, data = arff.readArff(filename)


    findBestAttributes(filename, algo_key, data, attributes)

