from __future__ import division
"""
A module to produce an .arff file with the same instances as file A
but with only the attiributes in file B

Created on 7/10/2010

@author: peter
"""

import sys, os, arff, csv


if __name__ == '__main__':

	if len(sys.argv) < 2:
		print 'Usage: jython find_duplicate_attributes.py  <arff-file>'
		sys.exit()

	base_filename = sys.argv[1]
	
	print base_filename

	relation, comments, attributes, data = arff.readArff(base_filename)

	sorted_data = sorted(data, key = lambda x: x[1:] + [x[0]])

	csv.writeCsv('temp.csv', sorted_data, [a['name'] for a in attributes])

	duplicates = []
	for i in range(1, len(sorted_data)):
		if sorted_data[i] == sorted_data[i-1]:
			duplicates.append(i)

	print 'duplicates', len(duplicates), duplicates
	
	num_attrs = len(attributes)
	def getHamming(d1, d2):
		hamming = 0
		for i in range(1, num_attrs):
			if d1[i] != d2[i]:
				hamming += 1
		return hamming
	
	def makeHammingStats(same_class):
		hamming_histogram = [0] * num_attrs
		total_hamming = 0
		num_hammings = 0
		for i in range(1, len(sorted_data)):
			for j in range(i):
				if same_class:
					do_it = (sorted_data[i][0] == sorted_data[j][0])
				else:
					do_it = (sorted_data[i][0] != sorted_data[j][0])
				if do_it:
					h = getHamming(sorted_data[i], sorted_data[j])
					hamming_histogram[h] += 1
					total_hamming += h
					num_hammings += 1
		print 'Hamming Histogram', same_class
		for i,h in enumerate(hamming_histogram):
			if h > 0:
				print '%2d' % i, '%6d' % h
		print 'Average', '%.1f' % (total_hamming/num_hammings)
		print '----------------------------'
		
	for same_class in [False, True]:
	 	makeHammingStats(same_class)
		

