import numpy as np
def epipole(u,v,smin,thresh,num_iterations = 1000):
    ''' Takes flow (u,v) with confidence smin and finds the epipole using only the points with confidence above the threshold thresh 
        (for both sampling and finding inliers)
        params:
            @u: np.array(h,w)
            @v: np.array(h,w)
            @smin: np.array(h,w)
        return value:
            @best_ep: np.array(3,)
            @inliers: np.array(n,) 
        
        u, v and smin are (h,w), thresh is a scalar
        output should be best_ep and inliers, which have shapes, respectively (3,) and (n,) 
    '''

   
    #Make linear vectors
    xp = np.arange(-256,256)
    yp = np.arange(-256,256)
    #Make a meshgrid like a (x,y) coordinate system
    xp,yp = np.meshgrid(xp,yp)
    #Flatten them 
    xp = xp.flatten()
    yp = yp.flatten()

    #Get indices for ransac - for sampling
    indices_for_flow = np.argwhere(smin.flatten()>thresh).flatten()

    #Make the Xp vector like [x,y,1]
    Xp = np.column_stack((xp,yp,np.ones((len(smin)**2,1)))).T
    #Make Flow vector [u,v,0]
    U = np.column_stack( (u.flatten(), v.flatten(), np.zeros((len(smin)**2,1)))   ).T


    sample_size = 2

    eps = 10**-2

    best_num_inliers = -1
    best_inliers = None
    best_ep = None

    for i in range(num_iterations): #Make sure to vectorize your code or it will be slow! Try not to introduce a nested loop inside this one
        permuted_indices = np.random.RandomState(seed=(i*10)).permutation(np.arange(0,np.sum((smin>thresh))))
        sample_indices = permuted_indices[:sample_size] #indices for thresholded arrays you find above
        test_indices = permuted_indices[sample_size:] #indices for thresholded arrays you find above
        
        # Based on sample indices calculate
        Sample_Indices = indices_for_flow[sample_indices]
        #Get the current inliers
        inliers = [int(i) for i in Sample_Indices]


        #Get the Xp x U matrix
        E = np.cross(U[:,Sample_Indices].T,Xp[:,Sample_Indices].T)

        # We are doing ET(Xp x U) So for epipole, we have to do (Xp x U).T E = 0
        _,_,Vt = np.linalg.svd(E)
    
        #Get the epipole
        ep = Vt[-1,:] 
        
        # Based on test indices calculate
        Test_Indices = indices_for_flow[test_indices]
        #Distances
        distances = abs( ( np.cross(U[:,Test_Indices].T,Xp[:,Test_Indices].T) )@ep   )
        #Add to the list of inliers        
        inliers.extend([int(i) for i in Test_Indices[distances<eps]])
        #Convert them to array for compatibility
        inliers = np.array(inliers)        

        #NOTE: inliers need to be indices in flattened original input (unthresholded), 
        #sample indices need to be before the test indices for the autograder
        if inliers.shape[0] > best_num_inliers:
            best_num_inliers = inliers.shape[0]
            best_ep = ep
            best_inliers = inliers

    return best_ep, best_inliers
