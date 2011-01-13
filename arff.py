from __future__ import division
"""
Operations on WEKA .arff files

Created on 28/09/2010

@author: peter
"""

import sys, re, os, datetime

def getAttributeByName_(attributes, name):
    """ Return attributes member with name <name> """
    for a in attributes:
        if a['name'] == name:
            return a
    return None

def showAttributeByName_(attributes, name, title):
    print '>>>', title, ':', getAttributeByName(attributes, name)

def debugAttributes(attributes, title):
    pass
    # showAttributeByName(attributes, 'Number.of.Successful.Grant', title)

def writeArff2(filename, comments, relation, attr_keys, attrs, data, make_copies = False):
    """ Write a WEKA .arff file 
    Params:
        filename: name of .arff file
        comments: free text comments 
        relation: name of data set
        attr_keys: gives order of keys in attrs to match columns
        attrs: dict of attribute: all values of attribute
        data: the actual data
    """
    assert(len(attr_keys) == len(attrs))
    assert(len(data[0]) == len(attrs))
    assert(len(attrs) >= 2)
    f = file(filename, 'w')
    f.write('\n')
    f.write('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
    f.write('%% %s \n' % os.path.basename(filename))
    f.write('%\n')
    f.write('% Created by ' + os.path.basename(sys.argv[0]) + ' on ' + datetime.date.today().strftime("%A, %d %B %Y") + '\n')
    f.write('% Code at http://bit.ly/read_arff\n')
    f.write('%\n')
    f.write('%% %d instances\n' % len(data))
    f.write('%% %d attributes + 1 class = %d columns\n' % (len(data[0]) - 1, len(data[0])))
    f.write('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
    f.write('\n')
    if comments:
        f.write('% Original comments\n')
        for c in comments:
            f.write(c + '\n')
    f.write('@RELATION ' + relation + '\n\n')
    for name in attr_keys:
        vals = attrs[name]
        if type(vals) is str:
            attrs_str = vals
        else:
            attrs_str = '{%s}' % ','.join([x for x in vals if not x == '?'])
        f.write('@ATTRIBUTE %-15s %s\n' % (name, attrs_str))
    f.write('\n@DATA\n\n')
    for instance in data:
        instance = ['?' if x == '' else x for x in instance]
        for i,name in enumerate(attr_keys):
            if type(attrs) is list:
                assert(instance[i] in attrs[name]+ ['?'])
        f.write(', '.join(instance) + '\n')
        #print ', '.join(instance)
    f.close()

    #print attr_keys[0], attrs[attr_keys[0]]
    #exit()

    if make_copies:
        """ Copy .arff files to .arff.txt so they can be viewed from Google docs """
        print 'writeArff:', filename + '.txt', '-- duplicate'
        shutil.copyfile(filename, filename + '.txt')

def writeArff(filename, comments, relation, attrs_list, data, make_copies = False, test = True):
    """ Write a WEKA .arff file 
    Params:
        filename: name of .arff file
        comments: free text comments 
        relation: name of data set
        attrs_list: list of dicts of attribute: all values of attribute
        data: the actual data
    """
    assert(len(attrs_list) > 0)
    assert(len(data) > 0)
    debugAttributes(attrs_list, 'writeArff')
    attr_keys = [x['name'] for x in attrs_list]
    attrs_dict = {}
    for x in attrs_list:
        attrs_dict[x['name']] = x['vals']
    writeArff2(filename, comments, relation, attr_keys, attrs_dict, data, make_copies)

    if test:
        out_relation, out_comments, out_attrs_list, out_data = readArff(filename)
        if out_attrs_list != attrs_list:
            print 'len(out_attrs_list) =', len(out_attrs_list), ', len(attrs_list) =', len(attrs_list)
            if len(out_attrs_list) == len(attrs_list):
                for i in range(len(attrs_list)):
                    print '%3d:' % i, out_attrs_list[i], attrs_list[i]
        assert(out_relation == relation)
        assert(out_attrs_list == attrs_list)
        assert(out_data == data)

def getRe(pattern, text):
    return re.findall(pattern, text)

relation_pattern = re.compile(r'@RELATION\s*(\S+)\s*$', re.IGNORECASE)
attr_name_pattern = re.compile(r'@ATTRIBUTE\s*(\S+)\s*', re.IGNORECASE)
attr_type_pattern = re.compile(r'@ATTRIBUTE\s*\S+\s*(\S+)', re.IGNORECASE)
attr_vals_pattern = re.compile(r'\{\s*(.+)\s*\}', re.IGNORECASE)
csv_pattern = re.compile(r'(?:^|,)(\"(?:[^\"]+|\"\")*\"|[^,]*)', re.IGNORECASE)

def readArff(filename):
    """ Read a WEKA .arff file
    Params: 
        filename: name of .arff file
    Returns:
        comments: free text comments 
        relation: name of data set
        attrs: list of attributes
        data: the actual data
    """
    print 'readArff(%s)' % filename

    lines = file(filename).readlines()
    lines = [l.rstrip('\n').strip() for l in lines]
    lines = [l for l in lines if len(l)]

    comments = [l for l in lines if l[0] == '%']
    lines = [l for l in lines if not l[0] == '%']
    
    relation = [l for l in lines if '@RELATION' in l.upper()]
    attributes = [l for l in lines if '@ATTRIBUTE' in l.upper()]
    
    #for i,a in enumerate(attributes[8:12]):
    #    print '%4d' % (8+i), a

    data = []
    in_data = False
    for l in lines:
        if in_data:
            data.append(l)
        elif '@DATA' in l.upper():
            in_data = True

    #print 'relation =', relation
    out_relation = getRe(relation_pattern, relation[0])[0]

    out_attrs = []

    for l in attributes:
        name = getRe(attr_name_pattern, l)[0]
        if not '{' in l:
            vals_string = getRe(attr_type_pattern, l)[0].strip()
            vals = vals_string.strip()
        else:
            vals_string = getRe(attr_vals_pattern, l)[0]
            vals = [x.strip() for x in vals_string.split(',')]
        out_attrs.append({'name':name, 'vals':vals})
        if False:
            print name, vals
            if name == 'Number.of.Successful.Grant':
                exit()

    #print 'out_attrs:', out_attrs
    out_data = []
    for l in data:
        out_data.append([x.strip() for x in getRe(csv_pattern, l)])
    for d in out_data:
        assert(len(out_attrs) == len(d))

    debugAttributes(out_attrs, 'readArff')

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