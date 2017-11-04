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

def current_reference(Y1,Y2,cout, pixel):
    
    _block = Block()
            
    # convert uint8 to int32
    Y1 = np.int32(Y1)
    Y2 = np.int32(Y2)

    Mvector, Ry = _block.Intercoding(Y1,Y2,pixel,macro)

    DCT_Ry = _block.DCT(Ry,pixel)
    IDCT_Ry = _block.IDCT(DCT_Ry,pixel)

    D_img = _block.Reconstruct(Y2,IDCT_Ry, Mvector,pixel)
    
    return D_img

#read YUV file 
def readfile():

    cout = 0
    while True:
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

        if (cout == 1):
            Y1 = Y[:,:]
            U1 = U[:,:]
            V1 = V[:,:]
            
        if (cout >= 2):
            Y2 = Y[:,:]
            U2 = U[:,:]
            V2 = V[:,:]
            Y1 = current_reference(Y2,Y1,cout,pix)
            # U1 = current_reference(U2,U1,cout,pix/2)
            # V1 = current_reference(V2,V1,cout,pix/2)


        # # Stack the YUV channels together, crop the actual resolution, convert to
        # # floating point for later calculations, and apply the standard biases
        # YUV = np.dstack((Y1, U1, V1))[:height, :width, :].astype(np.float)
        # YUV[:, :, 0]  = YUV[:, :, 0]  - 16   # Offset Y by 16
        # YUV[:, :, 1:] = YUV[:, :, 1:] - 128  # Offset UV by 128
        # # YUV conversion matrix from ITU-R BT.601 version (SDTV)
        # # Note the swapped R and B planes!
        # #              Y       U       V
        # M = np.array([[1.164,  2.017,  0.000],    # B
        #             [1.164, -0.392, -0.813],    # G
        #             [1.164,  0.000,  1.596]])   # R
        # # Take the dot product with the matrix to produce BGR output, clamp the
        # # results to byte range and convert to bytes
        # BGR = YUV.dot(M.T).clip(0, 255).astype(np.uint8)
        # # Display the image with OpenCV

        # print(Y1)
        cv2.imwrite('image' + str(cout) + ".png", Y1)
        # cv2.imshow('image', Y1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            

    cv2.destroyAllWindows()



stream = open('akiyo_cif.yuv', 'rb')  #rb to open non-text file
readfile()

