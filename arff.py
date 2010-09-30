from __future__ import division
"""
Operations on Weka .arff files

Created on 28/09/2010

@author: peter
"""

import sys, re, os, datetime

def writeArff(file_name, comments, relation, attrs, data, make_copies = False):
    """ Write a Weka .arff file """
    #print 'writeArff:', file_name, len(data), len(data[0])
    f = file(file_name, 'w')
    f.write('\n')
    f.write('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
    f.write('%% %s \n' % os.path.basename(file_name))
    f.write('%\n')
    f.write('% Created by ' + os.path.basename(sys.argv[0]) + ' on ' + datetime.date.today().strftime("%A, %d %B %Y") + '\n')
    f.write('% Code at http://bit.ly/b7Kkqt\n')
    f.write('%\n')
    f.write('% Constructed from raw data in http://archive.ics.uci.edu/ml/machine-learning-databases/soybean/\n')
    f.write('%% %d instances\n' % len(data))
    f.write('%% %d attributes + 1 class = %d columns\n' % (len(data[0]) - 1, len(data[0])))
    f.write('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
    f.write('\n')
    f.write('% Original comments\n')
    for c in comments:
        f.write(c + '\n')
    f.write('@RELATION ' + relation + '\n\n')
    for a in attrs:
        f.write('@ATTRIBUTE %-15s {%s}\n' % (a['name'], ','.join([x for x in a['vals'] if not x == '?'])))
    f.write('\n@DATA\n\n')
    for instance in data:
        f.write(', '.join(instance) + '\n')
    f.close()

    if make_copies:
        """ Copy .arff files to .arff.txt so they can be viewed from Google docs """
        print 'writeArff:', file_name + '.txt', '-- duplicate'
        shutil.copyfile(file_name, file_name + '.txt')

def quote(s):
    return '<<' + s + '>>'

def getRe(pattern, text):
    print '  getRe', quote(text)
    vals = re.findall(pattern, text)
    print '  ++', vals
    return vals

relation_pattern = re.compile(r'@RELATION\s*(\S+)\s*$', re.IGNORECASE)
attr_name_pattern = re.compile(r'@ATTRIBUTE\s*(\S+)\s*\{', re.IGNORECASE)
attr_vals_pattern = re.compile(r'\{\s*(.+)\s*\}', re.IGNORECASE)
csv_pattern = re.compile(r'(?:^|,)(\"(?:[^\"]+|\"\")*\"|[^,]*)', re.IGNORECASE)
    
def readArff(file_name):
    lines = file(file_name).readlines()
    lines = [l.rstrip('\n').strip() for l in lines]
    lines = [l for l in lines if len(l)]

    comments = [l for l in lines if l[0] == '%']
    lines = [l for l in lines if not l[0] == '%']
    
    relation = [l for l in lines if '@RELATION' in l.upper()]
    attributes = [l for l in lines if '@ATTRIBUTE' in l.upper()]
    
    data = []
    in_data = False
    for l in lines:
        if in_data:
            data.append(l)
        elif '@DATA' in l.upper():
            in_data = True

    out_relation = getRe(relation_pattern, relation[0])[0]

    out_attrs = []

    for l in attributes:
        name = getRe(attr_name_pattern, l)[0]
        vals_string = getRe(attr_vals_pattern, l)[0]
        vals = [x.strip() for x in vals_string.split(',')]
        out_attrs.append({'name':name, 'vals':vals})

    out_data = []
    for l in data:
        out_data.append([x.strip() for x in getRe(csv_pattern, l)])

    return (out_relation, comments, out_attrs, out_data)

def testCsv():
    if len(sys.argv) != 2:
        print "Usage: arff.py <arff-file>"
        sys.exit()

    in_file_name = sys.argv[1]
    out_file_name = os.path.splitext(in_file_name)[0] + '.copy' + os.path.splitext(in_file_name)[1]

    print 'Reading', in_file_name
    print 'Writing', out_file_name

    relation, comments, attrs, data = readArff(in_file_name)
    writeArff(out_file_name, comments, relation, attrs, data)
        
if __name__ == '__main__':
    if True:
        line = '1,a,"x,y",q'
        pattern = '(?:^|,)(\\\"(?:[^\\\"]+|\\\"\\\")*\\\"|[^,]*)'
        patter2 = r'(?:^|,)(\"(?:[^\"]+|\"\")*\"|[^,]*)'
        print pattern
        print patter2
        assert(patter2 == pattern)
        vals = re.findall(pattern, line)
        print pattern
        print line
        print vals
        
    if True:
        testCsv()