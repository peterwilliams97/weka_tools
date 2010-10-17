from __future__ import division
"""
Functions that belong in any other module

Created on 15/10/2010

@author: peter
"""
import os

def rm(filename):
    try:
        os.remove(filename)
    except:
        pass

def mkDir(dir):
    try:
        os.mkdir(dir)
    except:
        pass

def detectNumberCpus():
    """
    Detects the number of CPUs on a system.
    
    From  http://codeliberates.blogspot.com/2008/05/detecting-cpuscores-in-python.html
    """
    # Linux, Unix and MacOS:
    if hasattr(os, 'sysconf'):
        if os.sysconf_names.has_key('SC_NPROCESSORS_ONLN'):
            # Linux & Unix:
            ncpus = os.sysconf('SC_NPROCESSORS_ONLN')
            if isinstance(ncpus, int) and ncpus > 0:
                return ncpus
        else: # OSX:
            return int(os.popen2('sysctl -n hw.ncpu')[1].read())
 
    # Windows:
    if os.environ.has_key('NUMBER_OF_PROCESSORS'):
        ncpus = int(os.environ['NUMBER_OF_PROCESSORS']);
        if ncpus > 0:
            return ncpus
        
    return 1 # Default

def removeDuplicates(a_list):
    """ Remove duplicates from a list """
    if not a_list:
        return None
    out = [a_list[0]]
    for a in a_list[1:]:
        if not a in out:
            out.append(a)
    return out

if __name__ == '__main__':
    pass

