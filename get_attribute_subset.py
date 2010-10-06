from __future__ import division
"""
A module to produce an .arff file with the same instances as file A
but with only the attiributes in file B

Created on 2/10/2010

@author: peter
"""

import sys, os, arff


if __name__ == '__main__':

	if len(sys.argv) < 3:
		print 'Usage: jython get_attribute_subset.py  <base-arff-file> <attrs-arff-file>'
		sys.exit()
	

	base_filename = sys.argv[1]
	attrs_filename = sys.argv[2]
	out_filename = os.path.splitext(base_filename)[0] + '.attr_subset' + os.path.splitext(base_filename)[1] 
	
	print base_filename
	print attrs_filename
	print out_filename

   	relation, comments, attributes, data = arff.readArff(base_filename)
	_, _, attributes_subset, _ = arff.readArff(attrs_filename)

	attribute_index_map = {}
	for i,a in enumerate(attributes):
		attribute_index_map[a['name']] = i
		
	names_subset = [a['name'] for a in attributes_subset]
		
	indexes_subset = []
	for name in attribute_index_map.keys():
		if name in names_subset:
			indexes_subset.append(attribute_index_map[name])
	
	out_attributes = [attributes[i] for i in indexes_subset]
	out_data = [[d[i] for i in indexes_subset] for d in data]
	
	arff.writeArff(out_filename, comments, relation, out_attributes, out_data) 
		
		

                
    
