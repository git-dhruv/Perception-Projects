import numpy as np


def est_homography(X, X_prime):
    """
    Calculates the homography of two planes, from the plane defined by X
    to the plane defined by X_prime. In this assignment, X are the coordinates of the
    four corners of the soccer goal while X_prime are the four corners of the penn logo

    Input:
        X: 4x2 matrix of (x,y) coordinates of goal corners in video frame
        X_prime: 4x2 matrix of (x,y) coordinates of logo corners in penn logo
    Returns:
        H: 3x3 homogeneours transformation matrix s.t. X_prime ~ H*X

    """
    A = None
    #Make the matrix 
    for i in range(4):
        ax = np.array([-X[i][0],-X[i][1], -1, 0,0,0, X[i][0]*X_prime[i][0], X[i][1]*X_prime[i][0], X_prime[i][0] ])
        ay = np.array([0,0,0, -X[i][0],-X[i][1], -1, X[i][0]*X_prime[i][1], X[i][1]*X_prime[i][1], X_prime[i][1] ]) 
        if A is not None:
            A = np.concatenate([A,[ax],[ay]])
        else:
            A = np.concatenate([[ax],[ay]])

    #SVD
    [U, S , Vt ] = np.linalg.svd(A)
    #Directly taking rows instead of transposing
    Vt = Vt[-1,:].reshape((3, 3))


    return Vt

