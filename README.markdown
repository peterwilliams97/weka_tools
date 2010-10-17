# Set of [Jython](http://www.jython.org/) tools to perform data mining tasks using [Weka](http://www.cs.waikato.ac.nz/ml/weka/)

[http://bit.ly/weka_tools](http://bit.ly/weka_tools)

Needs Jython and Weka.

Uses UCI Michalski and Chilausky [soybean data set](http://archive.ics.uci.edu/ml/machine-learning-databases/soybean/ "soybean data")

Originally developed for a [class assignment](http://bit.ly/weka_data_mining).

## Summary
1. ** setup.bat** Shows how to set up classpath to use WEKA from Jython
2. **preprocess_soybeans.py** Pre-processes the soybean data set
3. **[find_best_attributes.py](http://github.com/peterwilliams97/weka_tools/blob/master/find_best_attributes.py)** Finds subset of attributes that give best classification accuracy for a given algorithm and data set
4. **arff.py** Weka .arff file reader and writer
5. **[split_data.py](http://bit.ly/split_data)** Splits a WEKA .arff file to preserve class distribution and maximize or minimize aggregate accuracy of a set of classifiers. Output is 2 WEKA .arff files
6. **[find_soybean_split.bat](http://github.com/peterwilliams97/weka_tools/blob/master/find_soybean_split.bat) / [find_soybean_split.sh](http://github.com/peterwilliams97/weka_tools/blob/master/find_soybean_split.sh) ** Shows how to run split_data.py on a pre-processed soybean .arff file

Results are in the [data](http://github.com/peterwilliams97/weka_tools/tree/master/data/) directory.

## Example use of [split_data.py](http://bit.ly/split_data)
The batch/shell file [find_soybean_split.bat](http://github.com/peterwilliams97/weka_tools/blob/master/find_soybean_split.bat) / [find_soybean_split.sh](http://github.com/peterwilliams97/weka_tools/blob/master/find_soybean_split.sh) runs [split_data.py](http://github.com/peterwilliams97/weka_tools/blob/master/data/soybean-large.data.orig.best.train.arff) on [soybean-large.data.orig.arff](http://github.com/peterwilliams97/weka_tools/blob/master/data/soybean-large.data.orig.best.test.arff) to create the training and test files [soybean-large.data.orig.best.train.arff](http://bit.ly/split_data) and [soybean-large.data.orig.best.test.arf](http://bit.ly/split_data) which give the classification results [soybean.split.results.txt](http://github.com/peterwilliams97/weka_tools/blob/master/data/soybean.split.results.txt) whose summary is

Classifier | Correct (out of 60) | Percentage Correct
-----------|---------------------|------------------
NaiveBayes |     57              | 95 %
J48        |     58              | 96.67 %
BayesNet   |     59              | 98.33 %
RandomForest |   59              | 98.33 %
JRip      |      60              | 100 %
KStar     |      60              | 100 %
SMO       |      60              | 100 %
MLP       |      60              | 100 %