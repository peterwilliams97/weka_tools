from __future__ import division
"""
Explore the Kaggle unimelb data set

Created on 1/1/2011

@author: peter
"""

import sys, os, collections, datetime, optparse, csv, arff , misc

def removeSpaces(x):
    return x.replace(' ', '') 

def cleanDictKeysAndVals(a_dict):
    out_dict = {}
    for k,v in a_dict.items():
        out_dict[removeSpaces(k)] = [removeSpaces(x) for x in v]
    return out_dict

NO_VALUES = ['']

def getNumElements(vector):
    return len([x for x in vector if x not in NO_VALUES])

def getFreqHisto(vector):
    """ Return the frequency histogram of values in vector """
    uniques = set(vector)
    for x in NO_VALUES:
        if x in uniques:
            uniques.remove(x)
    histo = collections.defaultdict(int)
    for x in vector:
        if x not in NO_VALUES:
            histo[x] += 1
    keys = list(uniques)
    keys.sort(key = lambda x: -histo[x])
    return keys, histo

def getNumElementsWithFreq(vector, min_freq):
    """ Returns number of values that are repeated <min_freq> or more times in <vector> """
    keys, histo = getFreqHisto(vector)
    num_with_freq = 0
    for k,v in histo.items():
        if v >= min_freq:
            num_with_freq += 1
    return num_with_freq
    
    
def moveKey(from_dict, to_dict, key):
    to_dict[key] = from_dict[key]
    del from_dict[key]

def filterDict(in_dict, filter_func):
    out_dict = {}
    for k,v in in_dict.items():
        if filter_func(k,v):
            out_dict[k] = v
    return out_dict

def stringToDate(string):
    """ Convert string of the form 19/11/05 to days since 1 Jan 2000 """
    parts = string.split('/')
    day, month, year = [int(x) for x in parts]
    if year < 20:
        year += 2000
    else:
        year += 1900
    date = datetime.date(year, month, day)
    base = datetime.date(2000, 1, 1)
    string2 = '%d/%02d/%02d' % (date.day, date.month, date.year % 100)
    #print string, string2
    assert(string == string2)
    return (date - base).days/365.25

def showStats(data_dict):
    """ Analyse a data dict by frequency stats """
    # Select the most interesting elements
    def sortFunc(name): 
        vals = data_dict[name]
        return getNumElementsWithFreq(vals,3)
        return getNumElements(vals) * getNumElementsWithFreq(vals,2)
    
    for i,name in enumerate(sorted(data_dict.keys(), key = sortFunc)):
        vals = data_dict[name]
        keys, histo = getFreqHisto(data_dict[name])
        freq_vals = [getNumElementsWithFreq(vals,freq) for freq in range(2,5)]
        print '%3d:%20s: %4d' % (i, name, getNumElements(vals)), freq_vals, ['%s:%d' % (k, histo[k]) for k in keys[:5]]
    
if __name__ == '__main__':

    parser = optparse.OptionParser('usage: python ' + sys.argv[0] + ' [options] <input file>')
    
    #parser.add_option('-f', '--first', dest='first_col', default='0', help='first column')
    parser.add_option('-k', '--keys', action='store_true', dest='show_keys', default=False, help='display all keys')
    #parser.add_option('-v', '--verbose', action='store_true', dest='verbose', default=False, help='show details in output')
    parser.add_option('-s', '--subset', dest='subset_keys', default='', help='comma separated list of keys to include')
        
    (options, args) = parser.parse_args()
    if len(args) < 1:
        print parser.usage
        print 'options:', options
        print 'args', args
        exit()

    inname = args[0]
    subset = None
    if len(options.subset_keys):
        subset = [x.strip() for x in options.subset_keys.split(',') if len(x.strip()) > 0]
    if subset:
        outname = os.path.splitext(inname)[0] + '.subset[%s]' % ','.join(subset)
    else:
        outname = os.path.splitext(inname)[0] + '.many'

    data_dict = cleanDictKeysAndVals(csv.readCsvAsDict(inname))
    data_dict_many = filterDict(data_dict, lambda k,v: getNumElements(v) >= len(data_dict['Person.ID'])/2)
    data_dict_many = filterDict(data_dict_many, lambda k,v: getNumElementsWithFreq(v,3) >= 2)
    if subset:
        data_dict_many = filterDict(data_dict, lambda k,v: k in subset + ['Grant.Status'])
    data_dict_many['Grant.Status'] = data_dict['Grant.Status']

    #showStats(data_dict)
    showStats(data_dict_many)

    date_strings = data_dict['Start.date']
    dates = ['%.2f' % stringToDate(x) for x in date_strings]
    print 'dates', sorted(dates)[::100]
    # Convert dates to numbers that Weka can understand
    data_dict_many['Start.date'] = dates

    numeric_keys = ['SEO.Percentage.1', 
                    'RFCD.Percentage.1',
                    'Number.of.Unsuccessful.Grant',
                    'SEO.Percentage.2',
                    'Number.of.Successful.Grant',
                    'Start.date']

    def makeAttrs(data_dict, numeric_keys):
        header = sorted(data_dict.keys(), key = lambda x: ' ' if x == 'Grant.Status' else x )
        attrs = {}
        for k,v in data_dict.items():
            if k in numeric_keys:
                attrs[k] = 'numeric'
            else:
                attrs[k] = sorted(set(x for x in v if x not in NO_VALUES))
        return header, attrs

    header, attrs = makeAttrs(data_dict_many, numeric_keys)
    columns_many = [data_dict_many[k] for k in header]
    data_many = misc.transpose(columns_many)
    data_many.sort(key = lambda x: -getNumElements(x))

    print header

    arff.writeArff2(outname + '.arff', None, 'relation', header, attrs, data_many[:10000])
    csv.writeCsv(outname + '.csv', data_many, header)

    if False:
        name = 'SEO.Code.4'
        keys, histo = getFreqHisto(data_dict[name])
        print name, ['%s:%d' % (k, histo[k]) for k in keys]
