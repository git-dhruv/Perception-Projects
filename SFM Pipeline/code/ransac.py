from random import sample
from matplotlib import test
from lse import least_squares_estimation
import numpy as np
from copy import deepcopy

def ransac_estimator(X1, X2, num_iterations=60000):
    sample_size = 8

    eps = 10**-4

    best_num_inliers = -1
    best_inliers = None
    best_E = None

    for i in range(num_iterations):
        # permuted_indices = np.random.permutation(np.arange(X1.shape[0]))
        permuted_indices = np.random.RandomState(seed=(i*10)).permutation(np.arange(X1.shape[0]))
        sample_indices = permuted_indices[:sample_size]
        test_indices = permuted_indices[sample_size:]


        #given random indices
        x1 = deepcopy(X1[sample_indices,:]) #better safe than sorry
        x2 = deepcopy(X2[sample_indices,:])


        #estimate
        E = least_squares_estimation(x1,x2)
        
        #Z direction skew symmetric
        sk_z = np.array([[0,-1,0],[1,0,0],[0,0,0]])

        inliers = sample_indices

        for i in test_indices:
            #Calculating distances
            dx2 = np.square(X2[i,:].T@(E@X1[i,:]))/(np.linalg.norm(sk_z@(E@X1[i,:]))**2) 
            dx1 = np.square(X1[i,:].T@(E.T@X2[i,:]))/(np.linalg.norm(sk_z@(E.T@X2[i,:]))**2) 

            #D1 plus D2 < tolerance
            if dx1+dx2<eps:
                inliers = np.append(inliers,i)
        

        if inliers.shape[0] > best_num_inliers:
            best_num_inliers = inliers.shape[0]
            best_E = E
            best_inliers = deepcopy(inliers)


    return best_E, best_inliers