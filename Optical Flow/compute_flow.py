import numpy as np
import pdb


def constrain(val,low,high):
    """
    Constrain Function to constrain between low and high values
    @Input: 
        val (int,float) : Value that is to be constrained
        low (int,float) : Lower bound
        high (int,float) : Upper bound
    @Return:
        val (int,float): Constrained value
    """
    if val<low:
        return low
    if val>high:
        return high
    return val

def flow_lk_patch(Ix, Iy, It, x, y, size=5):
    """
    params:
        @Ix: np.array(h, w)
        @Iy: np.array(h, w)
        @It: np.array(h, w)
        @x: int
        @y: int
    return value:
        flow: np.array(2,)
        conf: np.array(1,)
    """

    #Since x,y are centered around, we need to calculate start and end index of path
    #For the edge case, we also need to constrain them to the size of the image
    #Also int conversion is done

    #Calculating Offset from center
    offset = int(np.floor(size/2))

    startx = constrain(  x-offset, 0 , Ix.shape[0]-1)
    starty = constrain( y-offset, 0 , Ix.shape[0]-1)
    endx = constrain( x+offset, 0 , Ix.shape[0]-1)+1
    endy = constrain ( y+offset, 0 , Ix.shape[0]-1)+1
    
    startx,starty,endx,endy = int(startx),int(starty),int(endx),int(endy)

    #Calculate Matrix of Ax = B for optical flow
    #|Ix Iy||u| + It = 0 
    #       |v|

    #Gradient    
    #Traverse columns first and then rows
    A = np.column_stack( (np.array(Ix[starty:endy,startx:endx].flatten()).reshape(-1,1), np.array(Iy[starty:endy,startx:endx].flatten()).reshape(-1,1)) )

    #Difference
    B = -np.array(It[starty:endy,startx:endx].flatten()).reshape(-1,1)

    #Flow Calculation using KLT
    flow,_,_,singularvals = np.linalg.lstsq(A,B,rcond=None)

    #Minimum Singular Value
    conf = min(singularvals)

    # hmmmm...
    flow = [float(i) for i in flow]
    

    return flow, conf

import matplotlib.pyplot as plt

def flow_lk(Ix, Iy, It, size=5):
    """
    params:
        @Ix: np.array(h, w)
        @Iy: np.array(h, w)
        @It: np.array(h, w)
    return value:
        flow: np.array(h, w, 2)
        conf: np.array(h, w)
    """
    # plt.imshow(It)
    # plt.show()
    image_flow = np.zeros([Ix.shape[0], Ix.shape[1], 2])
    confidence = np.zeros([Ix.shape[0], Ix.shape[1]])
    for x in range(Ix.shape[1]):
        for y in range(Ix.shape[0]):
            flow, conf = flow_lk_patch(Ix, Iy, It, x, y)
            image_flow[y, x, :] = flow
            confidence[y, x] = conf
    return image_flow, confidence

    

