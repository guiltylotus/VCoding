import numpy as np
import cv2
from Block import *


# set width and height of video
width = 352
height = 288
size = 352*288*1.5
pix = 4
macro = 8 # pix + macro * 2 = size of seachArea block
sample = 2 # 4:2:2
qb = 10


def quantise(a, qb = 3.0):
    """
        return quantisation of 'a' matrix with QB = 10
    """ 
    np.savetxt('test.out', np.round(a / qb), fmt='%f')
    return np.round(a / qb)

def rescale(a, qb = 3.0):
    """
        rescale = re-quantise
    """
    return (a * qb)

def current_reference(Y1,Y2,cout, pixel):
    
    _block = Block()
            
    # convert uint8 to int32
    Y1 = np.int32(Y1)
    Y2 = np.int32(Y2)

    Mvector, Ry = _block.Intercoding(Y1,Y2,pixel,macro)
    # print Ry
    DCT_Ry = _block.DCT(Ry,pixel)
    # print DCT_Ry.dtype
    Quantised_Ry = quantise(DCT_Ry)
    Dequantised_Ry = rescale(Quantised_Ry)
    # print DCT_Ry
    IDCT_Ry = _block.IDCT(Dequantised_Ry,pixel)
    # print IDCT_Ry.dtype
    # print IDCT_Ry
    D_img = _block.Reconstruct(Y2,IDCT_Ry, Mvector,pixel)
    
    return D_img

import timeit
import pickle
#read YUV file 

def writefile(Y, U, V, outstream):
    def inverse_repeat(a, repeats, axis):
        if isinstance(repeats, int):
            indices = np.arange(a.shape[axis] / repeats, dtype=np.int) * repeats
        else:  # assume array_like of int
            indices = np.cumsum(repeats) - 1
        return a.take(indices, axis)
    data = []
    for x in Y.reshape((1, -1))[0].astype(dtype=np.uint8):
        data.append(x)
    for x in inverse_repeat(inverse_repeat(U, 2, 0), 2, 1).reshape((1, -1))[0].astype(dtype=np.uint8):
        data.append(x)  
    for x in inverse_repeat(inverse_repeat(V, 2, 0), 2, 1).reshape((1, -1))[0].astype(dtype=np.uint8):
        data.append(x)
    # print len(data), size
    # break
    for x in data:
        outstream.write(x)
def readfile():

    cout = 0
    dd = False
    while True:
        cout += 1
        if cout < 6:
            continue
        if cout > 7:
            break
        stream.seek(cout*int(size))   #skip all value before cout*size
        
        # Load the Y (luminance) data from the stream
        Y = np.fromfile(stream, dtype=np.uint8, count=width*height)
        if (len(Y) == 0):
            break
        Y = Y.reshape((height, width))
        U = np.fromfile(stream, dtype=np.uint8, count=(width//2)*(height//2)).reshape((height//2, width//2)).repeat(2, axis=0).repeat(2, axis=1)
        V = np.fromfile(stream, dtype=np.uint8, count=(width//2)*(height//2)).reshape((height//2, width//2)).repeat(2, axis=0).repeat(2, axis=1)
        
        if (cout == 6):
            Y1 = Y.copy()
            U1 = U.copy()
            V1 = V.copy()
            
        if (cout >= 2):
            Y2 = Y.copy()
            U2 = U.copy()
            V2 = V.copy()
            Y1 = current_reference(Y2,Y1,cout,pix)
            U1 = current_reference(U2,U1,cout,pix)
            V1 = current_reference(V2,V1,cout,pix)
        # cv2.imwrite('image' + str(cout) + ".png", Y1)
        writefile(Y1, U1, V1, outstream)
        
    print cout
        




stream = open('xyz.yuv', 'rb')  #rb to open non-text file
outstream = open('res.yuv', 'wb')
readfile()

