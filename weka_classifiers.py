from __future__ import division
"""
A wrapper for some WEKA classifiers.

Requires Jython and the WEKA java library weka.jar (see http://bit.ly/weka_tools)

Based on this code example:

    http://www.btbytes.com/2005/11/30/weka-j48-classifier-example-using-jython/

Note: needs WEKA 3.6.x to run (due to changes in the weka.classifiers.Evaluation class)

Created on 27/09/2010

@author: peter
"""

import java.io.FileReader as FileReader
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

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

import misc

verbose = False

def getMultiBoost():
    multi_boost = MultiBoost()
    multi_boost.setOptions(['-W weka.classifiers.functions.SMO'])
    return multi_boost

# The classifiers in this module
algo_list = [(NaiveBayes(), 'NaiveBayes'), (BayesNet(),'BayesNet'), (J48(),'J48'), (JRip(), 'JRip'),
                 (KStar(), 'KStar'), (RandomForest(), 'RandomForest'), (SMO(),'SMO'), (MLP(),'MLP'), 
                 (getMultiBoost(), 'MultiBoost')]
algo_dict = dict([(x[1], x[0]) for x in algo_list])
# Indication of time it takes algo to run. Higher is longer
algo_duration = {}.fromkeys(algo_dict.keys(), 0)
algo_duration['SMO'] = 5
algo_duration['MLP'] = 10
algo_duration['MultiBoost'] = 15

# Algo keys sorted in order of computation time 
all_algo_keys = ['NaiveBayes', 'J48', 'BayesNet', 'JRip', 'RandomForest', 'KStar', 'SMO', 'MLP', 'MultiBoost']

def runClassifierAlgo(algo, class_index, training_filename, test_filename, do_model, do_eval, do_predict):
    """ If <test_filename>
            Run classifier algorithm <algo> on training data in <training_filename> to build a model
            then test on data in <test_filename> (equivalent of Weka "Supplied test set") 
        else
            do 10 fold CV lassifier algorithm <algo> on data in <training_filename>
        
        <class_index> is the column containing the dependent variable 
        
        http://weka.wikispaces.com/Generating+classifier+evaluation+output+manually
        http://weka.sourceforge.net/doc.dev/weka/classifiers/Evaluation.html
    """
    print ' runClassifierAlgo: training_filename= ', training_filename, ', test_filename=', test_filename
    misc.checkExists(training_filename)

    training_file = FileReader(training_filename)
    training_data = Instances(training_file)
    if test_filename:
        test_file = FileReader(test_filename)
        test_data = Instances(test_file)
    else:
        test_data = training_data

   # set the class Index - the index of the dependent variable
    training_data.setClassIndex(class_index)
    test_data.setClassIndex(class_index)

    # create the model
    if test_filename:
        algo.buildClassifier(training_data)

    evaluation = None
    # only a trained classifier can be evaluated
    if do_eval or do_predict:
        evaluation = Evaluation(test_data)
        buffer = StringBuffer()             # buffer for the predictions
        attRange = Range()                  # no additional attributes output
        outputDistribution = Boolean(False) # we don't want distribution
        if test_filename:
            evaluation.evaluateModel(algo, test_data, [buffer, attRange, outputDistribution])
        else:
           # evaluation.evaluateModel(algo, [String('-t ' + training_filename), String('-c 1')])
           # print evaluation.toSummaryString()
            rand = Random(1)
            evaluation.crossValidateModel(algo, training_data, 4, rand)
            if False:
                print 'percentage correct =', evaluation.pctCorrect()
                print 'area under ROC =', evaluation.areaUnderROC(class_index)
                confusion_matrix = evaluation.confusionMatrix()
                for l in confusion_matrix:
                    print '** ', ','.join('%2d'%int(x) for x in l)

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

def getEvalAlgo(algo, class_index, training_filename, test_filename = None):
    """ Returns evaluation string for algorithm <algo> built from training data in <training_filename> 
        and tested on data in <test_filename> """ 
    result = runClassifierAlgo(algo, class_index, training_filename, test_filename, False, True, False)
    return result['eval'].strip()

def getEvalAlgoKey(algo_key, class_index, training_filename):
    algo = algo_dict[algo_key]
    return getEvalAlgo(algo, class_index, training_filename)

classify_tag = 'Correctly Classified Instances'

def getAccuracyAlgo(algo, class_index, training_filename, test_filename = None):
    """ Returns accuracy of algorithm <algo> built from training data in <training_filename> 
        and tested on data in <test_filename> """ 
    eval = getEvalAlgo(algo, class_index, training_filename, test_filename)
    lines = eval.split('\n')
    for ln in lines:
        if classify_tag in ln:
            contents = ln[len(classify_tag):]
            parts = [x.strip() for x in contents.strip().split(' ') if len(x) > 0]
            assert(len(parts) == 3)
            accuracy = float(parts[1])
            assert(isinstance(accuracy, float))
            return (accuracy, eval)
    raise ValueException('Cannot be here')

def getAccuracyAlgoKey(algo_key, class_index, training_filename):
    """ Return accuracy of algo named <algo_key> on <class_index> for <training_filename>
        for 10 fold CV """
    algo = algo_dict[algo_key]
    return getAccuracyAlgo(algo, class_index, training_filename)

training_file_base = '.train.arff'
test_file_base = '.test.arff'

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

    preprocess_soybeans.writeArff(training_file_name, 'soybean', base_classes, base_attrs, training_data)
    preprocess_soybeans.writeArff(test_file_name, 'soybean', base_classes, base_attrs, test_data)
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

"""
    Threaded access
    Computation time in this package is usually limited by the time it takes classifiers to run.
    Modern computers usually have several CPU cores so we will multi-thread classifiers
    A request queue for classifications will be processed by an array of threads
"""
import Queue
from threading import Thread


class ClassifierThread(Thread):
    _num_instances = 0
    
    def __init__ (self, id):
        Thread.__init__(self)
        self._id = id
        self._num_instances += 1

    def run(self):
        print 'Starting ClassifierThread', id
        while True:
            params = classifier_queue.get()
            if params == 'die':
                self._num_instances -= 1
                return
            result = getAccuracyAlgoKey(params['algo_key'], params['class_index'], params['training_filename'])

            classifier_queue.task_done()
            results_queue.put(result)

classifier_queue = Queue.Queue()
results_queue = Queue.Queue()
worker_thread_list = []

def initWekaAlgoThreads(num_worker_threads = 0):
    print 'initMultiThreadWekaAlgos', num_worker_threads
    if num_worker_threads <= 0:
        num_cpus = misc.detectNumberCpus()
        print 'num cpus', num_cpus
        num_worker_threads = max(1, misc.detectNumberCpus() - 1)
    print 'num worker_threads', num_worker_threads

    for i in range(num_worker_threads):
         t = ClassifierThread(i)
         t.daemon = True
         t.start()
         worker_thread_list.append(t)

def killWekaAlgoThreads():
    while ClassifierThread._num_instances > 0:
        classifier_queue.put('die')

def processEvalAlgoKeyRequests(params_list):
    """ Process a list of classifiers in parallel
        Return when they are all done
    """
    # print 'processEvalAlgoKeyRequests', len(params_list), params_list
    assert(classifier_queue)
    assert(results_queue)
    assert(len(worker_thread_list) > 0)

    num_requests = len(params_list)
    # Run classifiers, slowest first
    for params in sorted(params_list, key = lambda x: -algo_duration[x['algo_key']]):
        classifier_queue.put(params)

    if False:
        return [results_queue.get() for i in range(num_requests)]
    else:
        results_list = []
        for i in range(num_requests):
            results = results_queue.get()
            results_list.append(results)
            results_queue.task_done()
        return results_list

def getAccuracyAlgoKeyList(algo_key, class_index, training_filename_list):
    """ Return accuracies of algorithm corresponding to <algo_key> with class index <class_index>
        for list of files in <training_filename_list>
    """
    params_list = [{'algo_key':algo_key, 'class_index':class_index, 'training_filename':fn} for fn in training_filename_list]
    return processEvalAlgoKeyRequests(params_list)

if __name__ == '__main__':
    pass

