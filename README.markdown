Set of [Jython](http://www.jython.org/) tools to perform data mining tasks using [Weka](http://www.cs.waikato.ac.nz/ml/weka/)

[http://bit.ly/weka_tools](http://bit.ly/weka_tools)

Needs Jython and Weka.

Uses UCI Michalski and Chilausky [soybean data set](http://archive.ics.uci.edu/ml/machine-learning-databases/soybean/ "soybean data")

Originally developed for a [class assignment](http://bit.ly/weka_data_mining).

1. ** setup.bat** Shows how to set up classpath to use WEKA from Jython
2. **preprocess_soybeans.py** Pre-processes the soybean data set
3. **find_best_attributes.py** Finds subset of attributes that give best classification accuracy for a given algorithm and data set
4. **arff.py** Weka .arff file reader and writer
5. **[split_data.py](http://bit.ly/split_data)** Splits a WEKA .arff file to preserve class distribution and maximize or minimize aggregate accuracy of a set of classifiers. Output is 2 WEKA .arff files
6. **find_soybean_split.bat** Shows how to run split_data.py on a pre-processed soybean .arff file
