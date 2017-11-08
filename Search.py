import numpy as np

class Search():
    
    # the difference between 2 block
    def distance_MAD(self, Yc, Yr):
        
        # MAD = 0
        # height, width = np.shape(Yr) # take shape

        # for i in range(height):
        #     for j in range(width):
        #         MAD += abs(Yc[i,j] - Yr[i,j] )

        return np.sum(np.abs(Yc - Yr))


    # Yc current illuminance
    # Yr reference illuminance
    # pix : size of block
    def full_search(self, Yc, Yr, pix):

        height, width = np.shape(Yr)
        # print(width)
        # print(height)

        res = np.iinfo(np.int32).max  #get max int
        cpoint = (0,0)

        for i in range(height):
            for j in range(width):
                
                if (i + pix <= height and j + pix <= width):
                    newYr = Yr[i:i+pix, j:j+pix ]
                    r = self.distance_MAD(Yc, newYr)

                    if (r < res):
                        res = r
                        cpoint = (i,j)

        return cpoint
                        

        

