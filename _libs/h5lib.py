'''
============================================
h5dump
--------------------------------------------
copied from Phil Austin's ATSC-301 classcode
'''
#!/usr/bin/env python
from __future__ import print_function
import h5py
import argparse
import types

def print_attrs(name, obj):
    print("item name: ",name,repr(obj))
    for key, val in obj.attrs.iteritems():
        print("    %s: %s" % (key, val))

def h5dump(filename):
    if isinstance(filename,types.StringType):
        infile = h5py.File(filename,'r')
    elif isinstance(filename,h5py._hl.files.File):
        infile=filename
    else:
        raise IOError, "need an h5 filename or h5.File instance"
    infile.visititems(print_attrs)
    print('-------------------')
    print("attributes for the root file")
    print('-------------------')
    for key,value in infile.attrs.items():
        print("attribute name: ",key,"--- value: ",value)
    return None
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('h5_file',type=str,help='name of h5 file')
    args=parser.parse_args()
    filename=args.h5_file
    dumph5(filename)

    
    
