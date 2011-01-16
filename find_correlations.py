from __future__ import division
"""
A module to find the attributes that give the best prediction for 
a specified classification algorithms and parameters.

Created on 3/01/2011

@author: peter
"""

import sys, os, random, math, time, optparse, csv, preprocess_soybeans, weka_classifiers as WC, arff, ga, misc
import find_best_attributes

if __name__ == '__main__':
    # Set random seed so that each run gives same results
    random.seed(555)

    parser = optparse.OptionParser('usage: python ' + sys.argv[0] + ' [options] <input file>')
    parser.add_option('-o', '--output', dest='output_dir', default='output', help='output directory')

    (options, args) = parser.parse_args()
    if len(args) < 1:
        print parser.usage
        print 'options:', options
        print 'args', args
        exit()

    filename = args[0]
    output_dir = options.output_dir

    misc.mkDir(output_dir)

    relation, comments, attributes, data = arff.readArff(filename)

    #find_best_attributes.findBestAttributes(options.output_dir, filename, 'NaiveBayes', data, attributes, True)
    #print WC.all_algo_keys
    #exit()
    
    #for algo_key in WC.all_algo_keys[2:]:
    for algo_key in ['JRip', 'NaiveBayes', 'RandomForest', 'KStar','SMO', 'MLP', 'MultiBoost',  'J48', 'BayesNet',  ]:
        print '======================= findBestAttributes:', filename, algo_key 
        find_best_attributes.findBestAttributes(options.output_dir, filename, algo_key, data, attributes, True)

 