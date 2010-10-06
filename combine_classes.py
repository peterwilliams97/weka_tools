from __future__ import division
"""
A module to produce an .arff file with the same instances as file A
but with only the attributes in file B

Created on 2/10/2010

@author: peter
"""

import sys, os, copy, arff

if __name__ == '__main__':

	if len(sys.argv) < 5:
		print 'Usage: jython combine_attributes.py  <base-arff-file> <class_index> <class1> <class2> ...'
		sys.exit()

	base_filename = sys.argv[1]
	class_index = int(sys.argv[2])
	classes_to_combine = sorted(set(sys.argv[3:]))

	group_name = '$'.join(classes_to_combine)
	combine_filename  = os.path.splitext(base_filename)[0] + '.' + group_name + '.combine' + os.path.splitext(base_filename)[1]
	separate_filename = os.path.splitext(base_filename)[0] + '.' + group_name + '.separate' + os.path.splitext(base_filename)[1]

	print 'original:', base_filename
	print classes_to_combine
	print 'combined :', combine_filename
	print 'separated:', separate_filename

	relation, comments, attributes, data = arff.readArff(base_filename)

	for val in classes_to_combine:
		assert(val in attributes[class_index]['vals'])

	attrs_to_combine = []
	remaining_attrs = []
	for val in attributes[class_index]['vals']:
		if val in classes_to_combine:
			attrs_to_combine.append(val)
		else:
			remaining_attrs.append(val)

	combine_attributes = copy.deepcopy(attributes)
	combine_attributes[class_index]['vals'] = [group_name] + remaining_attrs

	separate_attributes = copy.deepcopy(attributes)
	separate_attributes[class_index]['vals'] = attrs_to_combine

	data_to_combine = []
	separate_data = []
	remaining_data = []
	for d in data:
		if d[class_index] in classes_to_combine:
			separate_data.append(d)
			d2 = copy.deepcopy(d)
			d2[class_index] = group_name
			data_to_combine.append(d2)
		else:
			remaining_data.append(d)
	combine_data = data_to_combine + remaining_data

	arff.writeArff(combine_filename, comments, relation, combine_attributes, combine_data)
	arff.writeArff(separate_filename, comments, relation, separate_attributes, separate_data)

