from __future__ import division
"""
Pre-process the soybean data set
http://archive.ics.uci.edu/ml/machine-learning-databases/soybean/

1. Read data. 
2. Create random data with same percentage of each attribute for comparison
	Confirm this random by going to Weka, classifying with J48 with 100 fold CV. Kappa should be close to 0
3. Check for duplicates. Compare number of duplicates to expected rate
4. Remove duplicates if greater than expected rate. 
	Save files showing duplicates with 'sorted' in their names
5. Add attribute names
6. Convert to .arff format
	Originals ('orig'), combined ('combined') and with duplicates removed

The downloaded data is stored in the following directory in the following files
Update these file names to match their locations on your computer 
""" 
dir = r'C:\dev\5045assignment1'
training_file = 'soybean-large.data.csv'
test_file = 'soybean-large.test.csv'
combined_file = 'soybean-combined.csv'
classes_file = 'soybean.classes'
attrs_file = 'soybean.attributes'
random_file = 'soybean-random.csv'

import math, os, sys, random, datetime, shutil, csv

def extractAttrs(data):
	""" Extract attributes from a raw data set which as class in the 
	    in the first column """
	return [instance[1:] for instance in data] 

def getDuplicates(in_data):
	""" Return duplicate instances in a data matrix. This is, all instances 
		except the first in a sorted version of the matrix """
	data = sorted(in_data)
	duplicates = []
	for i in range(1, len(data)):
		if data[i] == data[i-1]:
			duplicates.append(data[i])
	return duplicates

def markDuplicates(in_data):
	""" Mark the duplicates in a data file for verification and debugging """
	data = sorted(in_data)
	marked = [data[0]]
	for i in range(1, len(data)):
		instance = data[i]
		if instance == data[i-1]:
			marked.append(instance + ['duplicate'])
		else:
			marked.append(instance)
	return marked

def removeDuplicates(data, duplicates, remove_all):
	""" Return instances in duplicates from data. If remove_all then remove all instances
		else remove all but first instance """
	matches = [{'instance':instance, 'number':0} for instance in duplicates]
	out = []
	for instance in data:
		is_duplicate = False
		for m in matches:
			if m['instance'] == instance:
				if m['number'] > 0 or remove_all:
					is_duplicate = True
				m['number'] = m['number'] + 1
		if not is_duplicate:
			out.append(instance)
	return out

def appendDescription(dir, file_name, description):
	""" Append a description to file path made up of dir and file_name """
	path = os.path.join(dir, file_name)
	base, ext = os.path.splitext(path)
	return base + '.' + description + ext

def buildPath(dir, file_name, ext, description = None):
	""" Build a path from a dir file_name and ext and optionally description """
	path = os.path.join(dir, file_name)
	base, _ = os.path.splitext(path)
	if not description == None:
		base = base + '.' + description
	return base + ext

def clean(key):
	""" Remove characters not allowed in .arff files and trim """
	return key.strip().replace('%','').replace(' ', '-')

def parseAttrLine(line):
	""" Parse a line from an attributes file
		Each line looks like:
	 		1. date:		april,may,june,july,august,september,october,?. """
	pre, post = line.strip().split(':')
	number, attr = pre.strip().split('.')
	attr = attr.strip().replace('%','').replace(' ', '-')
	vals = [clean(x) for x in post.strip().strip('.').split(',')]
	return {'num':int(number), 'attr':clean(attr), 'vals':vals}

def parseAttrs(file_name):
	""" Parse an attributes file """
	lines = file(file_name).read().strip().split('\n')
	lines = [x.strip() for x in lines if len(x.strip()) > 0]
	return [parseAttrLine(x) for x in lines]

def parseClasses(file_name):
	""" Parse a classes file """
	lines = file(file_name).read().strip().split('\n')
	lines = [x.strip() for x in lines if len(x.strip()) > 0]
	classes = []
	for l in lines:
		classes = classes + [clean(x) for x in l.split(',')]
	return classes

def makeHeaderRow(attrs):
	"""	Make a header from attrs . class label goes at start """
	return ['class'] + [attrs[i]['attr'] for i in range(len(attrs))]

def applyAttrs(data, attrs):
	""" Add attribute names to a data file of enumerated values 
		Returns named attributes and a header line
	"""
	assert(len(data[0]) == len(attrs) + 1)
	num_attrs = len(attrs)
	num_instances = len(data)

	out = [None] * len(data)
	for row in range(num_instances):
		instance = data[row]
		out[row] = [instance[0]] + ['?' if instance[i+1] == '?' else attrs[i]['vals'][int(instance[i+1])] for i in range(num_attrs)]

	return out

def getClassDistribution(data):
	classes = set([instance[0] for instance in data])
	class_distribution = {}.fromkeys(classes, 0)
	for instance in data:
		class_distribution[instance[0]] = class_distribution[instance[0]] + 1
	return class_distribution

def dictToMatrix(d):
	m = [[k,d[k]] for k in sorted(d.keys())]
	return sorted(m, key = lambda x: -x[1])

def removeClassesWithFewerInstances(data, class_distribution, min_instances):
	return [instance for instance in data if class_distribution[instance[0]] >= min_instances]

def writeArff(file_name, relation, classes, attrs, data, make_copies = False):
	""" Write a Weka .arff file """
	#print 'writeArff:', file_name, len(data), len(data[0])
	f = file(file_name, 'w')
	f.write('%\n')
	f.write('%% %s \n' % os.path.basename(file_name))
	f.write('%\n')
	f.write('% Created by ' + os.path.basename(sys.argv[0]) + ' on ' + datetime.date.today().strftime("%A, %d %B %Y") + '\n')
	f.write('% Code at http://bit.ly/b7Kkqt\n')
	f.write('%\n')
	f.write('% Constructed from raw data in http://archive.ics.uci.edu/ml/machine-learning-databases/soybean/\n')
	f.write('%% %d instances\n' % len(data))
	f.write('%% %d attributes + 1 class = %d columns\n' % (len(data[0]) - 1, len(data[0])))
	f.write('\n')
	f.write('@RELATION ' + relation + '\n\n')
	f.write('@ATTRIBUTE %-15s {%s}\n' % ('class', ','.join([x for x in classes if not x == '?'])))
	for a in attrs:
		f.write('@ATTRIBUTE %-15s {%s}\n' % (a['attr'], ','.join([x for x in a['vals'] if not x == '?'])))
	f.write('\n@DATA\n\n')
	for instance in data:
		f.write(', '.join(instance) + '\n')
	f.close()

	if make_copies:
		""" Copy .arff files to .arff.txt so they can be viewed from Google docs """
		print 'writeArff:', file_name + '.txt', '-- duplicate'
		shutil.copyfile(file_name, file_name + '.txt')

def getRandomData(data):
	""" Simulate the population in data with random data with 
	    the same distribution of each attributes.
	    Returns a synthetic population of the same size as data
	"""
	num_attrs = len(data[0])
	num_instances = len(data)
	counts = [{} for i in range(num_attrs)]

	for instance in data:
		for i in range(num_attrs):
			cnt = counts[i]
			val = instance[i]
			cnt[val] = cnt[val] + 1 if  val in cnt.keys() else 1

	for i in range(num_attrs):
		tot = sum([counts[i][k] for k in counts[i].keys()])
		for k in counts[i].keys():
			counts[i][k] = counts[i][k]/tot
		tot = sum([counts[i][k] for k in counts[i].keys()])
		assert(abs(tot - 1.0) < 1e-6)

	random_data = [None] * num_instances
	for row in range(num_instances):
		random_data[row] = [0] * num_attrs
		for i in range(num_attrs): 
			r = random.random()
			top = 0
			for k in counts[i].keys():
				top = top + counts[i][k]
				if r <= top:
					random_data[row][i] = k
					break

	return  random_data

def preprocessSoybeanData():
	""" Pre-process the Soybean data set downloaded from http://archive.ics.uci.edu/ml/machine-learning-databases/soybean/
	"""

	""" Read the data files """
	training_data = csv.readCsvRaw(os.path.join(dir, training_file))
	test_data = csv.readCsvRaw(os.path.join(dir, test_file))
	
	""" Combined data file """
	combined_data = test_data + training_data
	print 'combined data', len(combined_data), len(combined_data[0])

	""" Random data file where the percentage of each class and attribute
		matches the combined data """
	random_data = getRandomData(combined_data)

	""" Find the duplicate instances in each data set
		The number of duplicates in random_data provides an estimate
		of the number of duplicates that would occur in the real
		data sets by pure chance """
	training_duplicates = getDuplicates(training_data)
	print 'training_duplicates =', len(training_duplicates)
	test_duplicates = getDuplicates(test_data)
	print 'test_duplicates =', len(test_duplicates)
	combined_duplicates = getDuplicates(combined_data)
	print 'combined_duplicates =', len(combined_duplicates)
	random_duplicates = getDuplicates(random_data)
	duplicates_warning = '*** Data files should not contain duplicates!' if len(random_duplicates) == 0 else ''
	print 'random_duplicates =', len(random_duplicates), duplicates_warning
	
	""" Remove duplicate instances within each data set 
		We know removing duplicates is valid if len(random_duplicates) is zero """
	filtered_training_data = removeDuplicates(training_data, training_duplicates, False)
	filtered_test_data = removeDuplicates(test_data, test_duplicates, False)
	filtered_combined_data = removeDuplicates(combined_data, combined_duplicates, False)
	filtered_random_data = removeDuplicates(random_data, random_duplicates, False)

	""" Remove the instances in duplicate-free test data that duplicate instances 
		in duplicate-free training data """
	all_duplicates = getDuplicates(filtered_training_data + filtered_test_data)
	filtered_test_data = removeDuplicates(filtered_test_data, all_duplicates, True)

	""" Sanity check """
	assert(len(filtered_test_data) + len(filtered_training_data) + len(combined_duplicates) == len(combined_data))

	""" Write out the intermediate .csv files with duplicates marked for debugging """
	csv.writeCsv(appendDescription(dir, training_file, 'sorted'), markDuplicates(training_data))
	csv.writeCsv(appendDescription(dir, test_file, 'sorted'), markDuplicates(test_data))
	csv.writeCsv(appendDescription(dir, combined_file, 'sorted'), markDuplicates(combined_data))
	csv.writeCsv(appendDescription(dir, random_file, 'sorted'), markDuplicates(random_data))

	""" Read the names of the classes and attributes from downloaded files """
	classes = parseClasses(os.path.join(dir, classes_file))
	attrs = parseAttrs(os.path.join(dir, attrs_file))

	""" Add class and attribute names to original data, for comparison with filter data """
	original_named_training_data = applyAttrs(training_data, attrs)
	original_named_test_data = applyAttrs(test_data, attrs)
	original_named_combined_data = applyAttrs(combined_data, attrs)
	
	""" Add class and attribute names to filtered data """
	named_training_data = applyAttrs(filtered_training_data, attrs)
	named_test_data = applyAttrs(filtered_test_data, attrs)
	named_combined_data = applyAttrs(filtered_combined_data, attrs)
	named_random_data = applyAttrs(filtered_random_data, attrs)
	
	""" Get the class distribution """
	class_distribution_training = getClassDistribution(named_training_data)
	class_distribution_test = getClassDistribution(named_test_data)
	class_distribution_combined = getClassDistribution(named_combined_data)

	named_training_data = removeClassesWithFewerInstances(named_training_data, class_distribution_training,2)

	""" Create a header row for the .csv file """
	header = makeHeaderRow(attrs)

	""" Write out the .csv files """
	
	csv.writeCsv(appendDescription(dir, training_file, 'distribution'), dictToMatrix(class_distribution_training), ['Class', 'Number'])
	
	csv.writeCsv(appendDescription(dir, training_file, 'orig'), named_training_data, header)
	csv.writeCsv(appendDescription(dir, test_file, 'orig'), named_test_data, header)
	csv.writeCsv(appendDescription(dir, combined_file, 'orig'), named_combined_data, header)

	csv.writeCsv(appendDescription(dir, training_file, 'named'), original_named_training_data, header)
	csv.writeCsv(appendDescription(dir, test_file, 'named'), original_named_test_data, header)
	csv.writeCsv(appendDescription(dir, combined_file, 'named'), original_named_combined_data, header)

	""" Write out the .arff files """
	writeArff(buildPath(dir, training_file, '.arff', 'orig'), 'soybean', classes, attrs, original_named_training_data)
	writeArff(buildPath(dir, test_file, '.arff', 'orig'), 'soybean', classes, attrs, original_named_test_data)
	writeArff(buildPath(dir, combined_file, '.arff', 'orig'), 'soybean', classes, attrs, original_named_combined_data)
	
	writeArff(buildPath(dir, training_file, '.arff'), 'soybean', classes, attrs, named_training_data)
	writeArff(buildPath(dir, test_file, '.arff'), 'soybean', classes, attrs, named_test_data)
	writeArff(buildPath(dir, combined_file, '.arff'), 'soybean', classes, attrs, named_combined_data)
	writeArff(buildPath(dir, random_file, '.arff'), 'soybean', classes, attrs, named_random_data)

if __name__ == '__main__':
	preprocessSoybeanData()

	