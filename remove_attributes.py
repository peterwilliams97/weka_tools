from __future__ import division
"""
A module to remove attributes from a Weka .arff file

Created on 7/10/2010

@author: peter
"""

import sys, os, arff


if __name__ == '__main__':

	if len(sys.argv) < 3:
		print 'Usage: jython remove_attributes.py <base-arff-file> <attr-to-remove-1> <attr-to-remove-2> ...'
		sys.exit()

	base_filename = sys.argv[1]
	excluded_attr_indexes = sorted([int(x) for x in sys.argv[2:]])

	indexes_name = '-'.join([str(x) for x in excluded_attr_indexes])
	out_filename = os.path.splitext(base_filename)[0] + '.' + indexes_name + os.path.splitext(base_filename)[1] 
	
	print base_filename
	print excluded_attr_indexes
	print out_filename

	relation, comments, attributes, data = arff.readArff(base_filename)

	print [(i,attributes[i]['name']) for i in excluded_attr_indexes]

	indexes_subset = [i for i in range(len(attributes)) if not i in excluded_attr_indexes]
	out_attributes = [attributes[i] for i in indexes_subset]
	out_data = [[d[i] for i in indexes_subset] for d in data]

	arff.writeArff(out_filename, comments, relation, out_attributes, out_data) 

