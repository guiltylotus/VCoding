import numpy as np
import cv2
from Search import *

class Block():

#_______________________________________INTERCODING______________________________________________________________________________________

    # Yc current illuminance
    # Yr reference illuminance
    # pix : size of block

    def motionEs(self, Yc, Yr, pix):

        _search = Search()
        cpoint = _search.full_search(Yc, Yr, pix)

        return cpoint

    # residual_block = Current_block - Reference_block 
    def motionCompensation(self, Yc, Yr):
        Y = np.zeros_like(Yc)

        height, width = np.shape(Yc)

        # Y = Yc[:,:] - Yr[:,:]
        for i in range(height):
            for j in range(width):
                Y[i,j] = Yc[i,j] - Yr[i,j]

        return(Y)

    # the area Y1 will be searched in  # (x,y) is top_left of Y1
    def searchArea(self,Y2, x, y, pix, macro):
        
        height, width = np.shape(Y2)

        top_left_x = x - macro
        if (top_left_x < 0):
            top_left_x = 0

        top_left_y = y - macro
        if (top_left_y < 0):
            top_left_y = 0 

        bottom_right_x = x + pix + macro
        if (bottom_right_x > height ):
            bottom_right_x = height
        
        bottom_right_y = y + pix + macro
        if (bottom_right_y > width):
            bottom_right_y  = width

        return(Y2[top_left_x:bottom_right_x, top_left_y: bottom_right_y], top_left_x, top_left_y)

        
    #check result of motionES
    def motionES_check(self,Y1, Y2, pix, res):
        
        height, width = np.shape(Y2)
        Ys = Y2[res[0]: res[0]+pix, res[1]:res[1]+pix]
        
        return
    
    # temporal coding
    def Intercoding(self, Y1,Y2,pix,macro):
        # input current image and reference image, size of subblock, size of search area on 4 sides

        # get width, height
        height, width = np.shape(Y2)

        residual_Y = np.zeros_like(Y1)
        Mvector = []

        # for each 8 pixel
        for i in range(0,height,pix):
            for j in range(0,width,pix):
                
                Ys, top_left_x, top_left_y = self.searchArea(Y2,i,j,pix,macro)

                res = self.motionEs(Y1[i:i+pix, j:j+pix], Ys, pix)
                Mvector.append((res[0] + top_left_x -i, res[1] + top_left_y-j))  
                # self.motionES_check(Y1[i:i+pix, j:j+pix], Ys, pix, res)
                newY = self.motionCompensation(Y1[i:i+pix, j:j+pix], Ys[res[0]: res[0]+pix, res[1]:res[1]+pix])
                residual_Y[i:i+pix, j:j+pix] = newY[:,:]


        print("intercoding Done!")
        return([Mvector,residual_Y])
        
        # output : motion vector (of current block -> reference block ) , residual image
    
#_______________________________________DCT_______________________________________________________________________________________________

    # Count C of matrix A
    def Count_C(self,x,n):
        
        if (x == 0):
            return(np.sqrt(np.float(1)/n))
        else: 
            return(np.sqrt(np.float(2)/n))

    # Transform matrix
    def Create_transform_matrix(self,n):
        
        A = np.full((n,n), np.float(0.))

        for i in range(n):
            for j in range(n):
                
                C = self.Count_C(i,n)
                r = np.float(2*j+1)*i*180 / (2*n)
                A[i,j] = C * np.cos(np.deg2rad(r))

        return A

    
    # DCT on X. X is  a small block (ex : 8x8) of residual image, Y = A * X * A.T
    def DCT_a_block_of_img(self, X , pix):
        
        Y = np.full((pix, pix), np.float(0.))
        A = self.Create_transform_matrix(pix)

        Y = np.round(A.dot(X.dot(A.T)) , decimals=3) # take 3 value of decimals
        return Y

    # block DCT (tranformation)
    # resi_img is residual image, pix is size of block
    def DCT(self, resi_img, pix):
        
        height , width = np.shape(resi_img)
        Yimg = np.full((height,width), np.float(0.0))

        for i in range(0,height,pix):
            for j in range(0,width,pix):
                
                X = resi_img[i:i+pix, j:j+pix]

                R = self.DCT_a_block_of_img(X, pix)
                Yimg[i:i+pix, j:j+pix] =  R

        print("DCT Done!")
        return Yimg

#_______________________________________Reconstruct_______________________________________________________________________________________

    def motion_vector_to_block(self, i, pix, height, width, vector):
        # input : current postion of vector in list, size of sub-block, height and width of image, vector at i

        x1 = (i / (width / pix)) * pix
        y1 = (i % (width / pix)) * pix

        
        x = x1 + vector[0]
        y = y1 + vector[1]

        return ([x1,y1],[x,y])

        # output: position topleft of reference block 

    # reconstruct img = IDCT img + decoded reference img 
    def Reconstruct(self, y2, idctImg, mvector, pix):
        # input : decoded reference img, idct image, list of motion vector ((x1,y1),(x2,y2)...), size of sub-block

        height , width = np.shape(y2)
        re_img = np.full((height,width), np.int32(0))    

        for i in range(len(mvector)):
            
            cp , rp = self.motion_vector_to_block(i, pix, height, width, mvector[i]) # cp: topleft current point of current imamge, rp: topleft point of best match in Reference image 
            
            x = cp[0]
            y = cp[1]
            cx = x + pix
            cy = y + pix

            u = rp[0]
            v = rp[1]
            ru = u + pix
            rv = v + pix

            re_img[x:cx,y:cy] = idctImg[x:cx,y:cy] + y2[u:ru, v:rv]

        print("Reconstruct Done!")
        return re_img
        # output: reconstruct image 



#_______________________________________IDCT______________________________________________________________________________________________

    # X = A.T * Y * A
    def IDCT_a_block_of_img(self, X , pix):
        
        Y = np.full((pix, pix), np.float(0.))
        A = self.Create_transform_matrix(pix)

        Y = np.round((A.T).dot(X.dot(A)) , decimals=3) # take 3 value of decimals
        return Y

    def IDCT(self, resi_img, pix):
        
        height , width = np.shape(resi_img)
        Yimg = np.full((height,width), np.float(0.0))

        for i in range(0,height,pix):
            for j in range(0,width,pix):
                
                X = resi_img[i:i+pix, j:j+pix]

                R = self.IDCT_a_block_of_img(X, pix)
                Yimg[i:i+pix, j:j+pix] =  R

        print("IDCT Done!")
        return Yimg