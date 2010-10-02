from __future__ import division
"""
A wrapper for some Weka classifiers.

Requires Jython and the Weka java library weka.jar (see 
http://github.com/peterwilliams97/weka_tools/blob/master/readme.markdown)

Based on this code example:

    http://www.btbytes.com/2005/11/30/weka-j48-classifier-example-using-jython/

Note: needs Weka 3.6.x to run (due to changes in the weka.classifiers.Evaluation class)

Created on 27/09/2010

@author: peter
"""

#import sys, os, random
#from math import *

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

verbose = False

def getMultiBoost():
    multi_boost = MultiBoost()
    multi_boost.setOptions(['-W weka.classifiers.functions.SMO'])
    return multi_boost

algo_list = [(NaiveBayes(), 'NaiveBayes'), (BayesNet(),'BayesNet'), (J48(),'J48'), (JRip(), 'JRip'),
                 (KStar(), 'KStar'), (RandomForest(), 'RandomForest'), (SMO(),'SMO'), (MLP(),'MLP'), 
                 (getMultiBoost(), 'MultiBoost')]
algo_dict = dict([(x[1], x[0]) for x in algo_list])


def runClassifierAlgo(algo, class_index, training_filename, test_filename, do_model, do_eval, do_predict):
    """ Run classifier algorithm <algo> on training data in <training_filename> to build a model
        then run in on data in <test_filename> (equivalent of Weka "Supplied test set") 
        <class_index> is the column containing the dependent variable 
        
        http://weka.wikispaces.com/Generating+classifier+evaluation+output+manually
        http://weka.sourceforge.net/doc.dev/weka/classifiers/Evaluation.html
        """
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
            rand = Random(1)
            evaluation.crossValidateModel(algo, training_data, 10, rand)

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
    lines = getEvalAlgo(algo, class_index, training_filename, test_filename).split('\n')
    for ln in lines:
        if classify_tag in ln:
            contents = ln[len(classify_tag):]
            parts = [x.strip() for x in contents.strip().split(' ') if len(x) > 0]
            assert(len(parts) == 3)
            accuracy = float(parts[1])
            assert(isinstance(accuracy, float))
            return accuracy
    raise ValueException('Cannot be here')

def getAccuracyAlgoKey(algo_key, class_index, training_filename):
    """ Return accuracy of algo named <algo_key> on <class_index> for <training_filename>
        for 10 fold CV """
    algo = algo_dict[algo_key]
    return getAccuracyAlgo(algo, class_index, training_filename)

def getAccuracy(training_filename, test_filename):
    algo_list = [NaiveBayes(), BayesNet(), J48(), RandomForest(), JRip(), KStar(), SMO(), MLP(), MultiBoost()]

    #algo_list = [JRip(), KStar()]# , RandomForest(), getMultiBoost()]
    #algo_list = [J48()]
    #algo_list = [NaiveBayes()]
    if False:
        good_list = [MLP()]
        bad_list = [SMO()]
        return len(bad_list)*100.0 \
                + sum([getAccuracyAlgo(algo, training_filename, test_filename) for algo in good_list]) \
                - sum([getAccuracyAlgo(algo, training_filename, test_filename) for algo in bad_list])
    if do_worst:
        return len(algo_list)*100.0 - sum([getAccuracyAlgo(algo, training_filename, test_filename) for algo in algo_list])
    else:
        return  sum([getAccuracyAlgo(algo, training_filename, test_filename) for algo in algo_list])

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

    if False:
        print 'base_data', len(base_data), len(base_data[0])
        print 'training_data', len(training_data), len(training_data[0])
        print 'test_data', len(test_data), len(test_data[0])

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

if __name__ == '__main__':
    pass

