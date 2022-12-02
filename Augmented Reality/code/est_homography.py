import numpy as np

def est_homography(X, Y):
    """
    Calculates the homography H of two planes such that Y ~ H*X
    If you want to use this function for hw5, you need to figure out
    what X and Y should be.
    Input:
        X: 4x2 matrix of (x,y) coordinates
        Y: 4x2 matrix of (x,y) coordinates
    Returns:
        H: 3x3 homogeneours transformation matrix s.t. Y ~ H*X

    """    

    A = None
    #Make the matrix 
    for i in range(4):
        ax = np.array([-X[i][0],-X[i][1], -1, 0,0,0, X[i][0]*Y[i][0], X[i][1]*Y[i][0], Y[i][0] ])
        ay = np.array([0,0,0, -X[i][0],-X[i][1], -1, X[i][0]*Y[i][1], X[i][1]*Y[i][1], Y[i][1] ]) 
        if A is not None:
            A = np.concatenate([A,[ax],[ay]])
        else:
            A = np.concatenate([[ax],[ay]])

    #SVD
    [U, S , Vt ] = np.linalg.svd(A)
    #Directly taking rows instead of transposing
    H = Vt[-1,:].reshape((3, 3))

    return H/H[2,2]
