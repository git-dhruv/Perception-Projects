import numpy as np

def est_pixel_world(pixels, R_wc, t_wc, K):
    """
    Estimate the world coordinates of a point given a set of pixel coordinates.
    The points are assumed to lie on the x-y plane in the world.
    Input:
        pixels: N x 2 coordiantes of pixels
        R_wc: (3, 3) Rotation of camera in world
        t_wc: (3, ) translation from world to camera
        K: 3 x 3 camara intrinsics
    Returns:
        Pw: N x 3 points, the world coordinates of pixels
    """

    #Forget everything, get it back to the inital frame
    R_cw = np.linalg.inv(R_wc)
    
    #Negate the transformation to invert the vector but also then align the frame
    t_cw = np.dot(R_cw,-t_wc)

    #Make the [R|t]
    Homogenous = np.column_stack((R_cw,t_cw))

    #the points follow homography since z = 0
    Homography = np.delete(Homogenous, 2, 1)

    #This is homography with callibration. 
    Calib_H = np.linalg.inv(np.dot(K,Homography))
    #Normalize
    Calib_H = Calib_H/Calib_H[2,2]
    

    #Prepare the pixels - [x,y,1]
    pixels_prepared = np.row_stack((pixels.T,np.ones((1,len(pixels[:,0])))))
    #-- Transform to World  --#
    P = np.dot(Calib_H,pixels_prepared)


    #Normalize the world coordinate
    P = P/P[2,:]
    #Make the scalar a 0 since Z axis = 0
    P = P - np.array([[0],[0],[1]])

    Pw = P.T
    return Pw
