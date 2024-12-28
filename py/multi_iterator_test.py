#! /usr/bin/env python3
import numpy as np

def multi_iterator_test():
    x=np.arange(4).reshape(1,4).astype(float)
    y=np.arange(24).reshape(3,2,4)
    z=np.arange(0,240,10).reshape(1,2,3,4)

    op_axes=[[ -1,  0,  1, -1],
             [  0, -1,  2,  1],
             [  2,  0,  3,  1]]
    it=np.nditer((x,y,z),op_axes=op_axes,order='C')
    for w in it:
        print("{0} {1} {2}".format(*w))

if __name__=="__main__":
    multi_iterator_test()
