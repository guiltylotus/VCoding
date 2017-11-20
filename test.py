import numpy as np
import cv2
from Block import *
from Nghia import *
import json
from ast import literal_eval as make_tuple
# set width and height of video
width = 352
height = 288
size = 352*288*1.5
pix = 8
macro = 8 # pix + macro * 2 = size of seachArea block
sample = 2 # 4:2:2
prob_residual = {}
prob_motion = {}

# def quantise(a, qb = 30):
#     """
#         return quantisation of 'a' matrix with QB = 10
#     """ 
#     # np.savetxt('test.out', np.round(a / qb), fmt='%d')
#     return np.int32(np.sign(a) * np.round(np.abs(a)/qb))

# def rescale(a, qb = 30):
#     """
#         rescale = re-quantise
#     """
#     return a * qb
compressed = '' 
def current_reference(Y1,Y2,cout, pixel, prob_residual, prob_motion, codebook_residual = None, codebook_motion = None):

    _block = Block()
    global compressed
    # convert uint8 to int32
    Y1 = np.int32(Y1)
    Y2 = np.int32(Y2)

    Mvector, Ry = _block.Intercoding(Y1,Y2,pixel,macro)
    for vec in Mvector:
        prob_motion[vec] = 0 if vec not in prob_motion else prob_motion[vec] + 1
    if codebook_motion:
        compressed += "".join(entropyCoding(codebook_motion, Mvector)) 
    # print Ry
    DCT_Ry = _block.DCT(Ry,pixel)
    # print DCT_Ry.dtype

    Rys = blockshaped(DCT_Ry, pixel, pixel)

    IRys = Rys.copy()
    for i, Ry in enumerate(Rys):
        Quantised_Ry = quantise(Ry)

        reordered_Ry = reorder(prob_residual, Quantised_Ry)

        if codebook_residual:
            compressed += "".join(entropyCoding(codebook_residual, reordered_Ry))

        inverse_reordered_Ry = inverseReorder(reordered_Ry)

        Dequantised_Ry = rescale(Quantised_Ry)

        IRys[i] = Dequantised_Ry
    IRys = mergeshaped(IRys, height, width)
    # print DCT_Ry
    IDCT_Ry = _block.IDCT(IRys,pixel)
    # print IDCT_Ry.dtype
    # print IDCT_Ry
    D_img = _block.Reconstruct(Y2,IDCT_Ry, Mvector,pixel)
    
    return D_img

def encode():
    pass


#read YUV file 

def writefile(Y, U, V, outstream):
    def inverse_repeat(a, repeats, axis):
        if isinstance(repeats, int):
            indices = np.arange(a.shape[axis] / repeats, dtype=np.int) * repeats
        else:  # assume array_like of int
            indices = np.cumsum(repeats) - 1
        return a.take(indices, axis)
    data = []
    for x in Y.reshape((1, -1))[0].clip(0, 255).astype(dtype=np.uint8):
        data.append(x)
    for x in inverse_repeat(inverse_repeat(U, 2, 0), 2, 1).reshape((1, -1))[0].clip(0, 255).astype(dtype=np.uint8):
        data.append(x)  
    for x in inverse_repeat(inverse_repeat(V, 2, 0), 2, 1).reshape((1, -1))[0].clip(0, 255).astype(dtype=np.uint8):
        data.append(x)
    # print len(data), size
    # break
    for x in data:
        outstream.write(x)
def readfile():
   
    codebook_residual = None
    codebook_motion = None
    for done_prob in [False, True]:
        cout = 0
        Y1 = np.zeros((height, width), dtype=np.uint8)
        U1 = Y1.copy()
        V1 = Y1.copy()
        if done_prob:
            codebook_residual = doHuffman(prob_residual)
            codebook_motion = doHuffman(prob_motion)
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
            
            Y2 = Y.copy()
            U2 = U.copy()
            V2 = V.copy()
            Y1 = current_reference(Y2, Y1, cout, pix, prob_residual, prob_motion, codebook_residual, codebook_motion)
            U1 = current_reference(U2, U1, cout, pix, prob_residual, prob_motion, codebook_residual, codebook_motion)
            V1 = current_reference(V2, V1, cout, pix, prob_residual, prob_motion, codebook_residual, codebook_motion)
            # cv2.imwrite('image' + str(cout) + ".png", Y1)
            # if done_prob:
            #     writefile(Y1, U1, V1, outstream)
    f = open('compressed.txt','w')
    f.write(compressed)
    f.close()

    with open('huffman_residual.txt', 'w') as f:
        json.dump(swapDict(codebook_residual), f)

    with open('huffman_motion.txt', 'w') as f:
        json.dump(swapDict(codebook_motion), f)


stream = open('xyz.yuv', 'rb')  #rb to open non-text file
outstream = open('res.yuv', 'wb')
readfile()

def decodeFrame(ref, Mvector, residual):
    blocks = []
    for resi in residual:
        inverse_reordered_Ry = inverseReorder(tuple(resi))

        Dequantised_Ry = rescale(Quantised_Ry)

        blocks.append(Dequantised_Ry)

    IRys = mergeshaped(np.array(blocks), height, width)
    IDCT_Ry = _block.IDCT(IRys,pix)
    D_img = _block.Reconstruct(ref,IDCT_Ry, Mvector,pix)
    
    return D_img

def decode(height, width, pix):
    with open('huffman_residual.txt','r') as f:
        codebook_residual = json.load(f)

    with open('huffman_motion.txt','r') as f:
        codebook_motion = json.load(f)

    with open('compressed.txt','r') as f:
        compressed = f.read()

    video = []
    idx = 0
    Yr = np.zeros((width, height), dtype=np.uint8)
    Ur = Yr.copy()
    Vr = Yr.copy()
    while idx < len(compressed):
        idx, Mvector, residual = huffmanDecode(height, width, pix, codebook_residual, codebook_motion, compressed, idx)
        Y = decodeFrame(Yr, Mvector, residual)
        idx, Mvector, residual = huffmanDecode(height, width, pix, codebook_residual, codebook_motion, compressed, idx)
        U = decodeFrame(Ur, Mvector, residual)
        idx, Mvector, residual = huffmanDecode(height, width, pix, codebook_residual, codebook_motion, compressed, idx)
        V = decodeFrame(Vr, Mvector, residual)
        writefile(Y, U, V, outstream)
        Yr, Ur, Vr = Y, U, V       


decode(height, width, pix)
