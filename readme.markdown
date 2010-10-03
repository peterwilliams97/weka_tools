Set of [Jython](http://www.jython.org/) tools to perform data mining tasks using [Weka](http://www.cs.waikato.ac.nz/ml/weka/)
[http://bit.ly/weka_tools](http://bit.ly/weka_tools)

Needs Jython and Weka.

Uses UCI Michalski and Chilausky [soybean data set](http://archive.ics.uci.edu/ml/machine-learning-databases/soybean/ "soybean data")

1. ** setup.bat** Shows how to set up classpath to use Weka
2. **run-ga.bat** Shows how to run split_data.py
3. **split_data.py**  Finds split of data file into training and test sets that give best results. Output is 2 Weka files
4. **preprocess_soybeans.py** Pre-processes the soybean data set
5. **find_best_attributes.py** Finds subset of attributes that give best classification accuracy for a given algorithm and data set
6. **arff.py** Weka .arff file reader and writer
