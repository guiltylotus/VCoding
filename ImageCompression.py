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

def current_reference(Y1,cout, pixel):
    
    _block = Block()
            
    # convert uint8 to int32
    Y1 = np.int32(Y1)

    DCT_Ry = _block.DCT(Y1,pixel)
    DCT_Ry_Quantised = _block.quantise(DCT_Ry)
    DCT_Ry_Requantised = _block.rescale(DCT_Ry_Quantised)
    IDCT_Ry = _block.IDCT(DCT_Ry_Requantised,pixel)
  
    # D_img = _block.Reconstruct(Y2,IDCT_Ry, Mvector,pixel)
    
    return IDCT_Ry

import timeit
#read YUV file 
def readfile():

    cout = 0
    while True:
        start_time = timeit.default_timer()
        cout += 1
        
        stream.seek(cout*size)   #skip all value before cout*size

        # Load the Y (luminance) data from the stream
        Y = np.fromfile(stream, dtype=np.uint8, count=width*height)
        if (len(Y) == 0):
            break
        Y = Y.reshape((height, width))
        
        # Load the UV (chrominance) data from the stream, and double its size
        U = np.fromfile(stream, dtype=np.uint8, count=(width//2)*(height//2)).reshape((height//2, width//2)).repeat(2, axis=0).repeat(2, axis=1)
        V = np.fromfile(stream, dtype=np.uint8, count=(width//2)*(height//2)).reshape((height//2, width//2)).repeat(2, axis=0).repeat(2, axis=1)

        Y2 = Y[:,:]
        U2 = U[:,:]
        V2 = V[:,:]
        Y1 = current_reference(Y2,cout,pix)
        U1 = current_reference(U2,cout,pix/2)
        V1 = current_reference(V2,cout,pix/2)



        # # Stack the YUV channels together, crop the actual resolution, convert to
        # # floating point for later calculations, and apply the standard biases
        # YUV = np.dstack((Y1, U1, V1))[:height, :width, :].astype(np.float)
        YUV = np.dstack((Y1, U2, V2))[:height, :width, :].astype(np.float)
        # YUV[:, :, 0]  = YUV[:, :, 0]  - 16   # Offset Y by 16
        YUV[:, :, 1:] = YUV[:, :, 1:] - 128  # Offset UV by 128
        # # YUV conversion matrix from ITU-R BT.601 version (SDTV)
        # # Note the swapped R and B planes!
        # #              Y       U       V
        M = np.array([[1.164,  2.017,  0.000],    # B
                    [1.164, -0.392, -0.813],    # G
                    [1.164,  0.000,  1.596]])   # R
        # # Take the dot product with the matrix to produce BGR output, clamp the
        # # results to byte range and convert to bytes
        BGR = YUV.dot(M.T).clip(0, 255).astype(np.uint8)
        # # Display the image with OpenCV

        # cv2.imwrite('YRimage' + str(cout) + ".png", Y)
        # cv2.imwrite('URimage' + str(cout) + ".png", U)
        # cv2.imwrite('VRimage' + str(cout) + ".png", V)
        print('Y2 ', Y2)
        print('Y1 ', Y1)
        cv2.imwrite('BGRimage' + str(cout) + ".png", BGR)
        print (timeit.default_timer() - start_time)
            



stream = open('xyz.yuv', 'rb')  #rb to open non-text file
readfile()

