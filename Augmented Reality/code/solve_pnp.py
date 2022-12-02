from est_homography import est_homography
import numpy as np

def PnP(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-N-Point problem with collineation assumption, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3, ) numpy array describing camera translation in the world (t_wc)

    """

    # Homography Approach

    #Modified Pw
    pw_without_z = np.delete(Pw,(2), axis=1)
    
    H = est_homography(pw_without_z,Pc) #Homography that maps world to camera

    #We will normalize homography with sum of squares 
    #Reference: https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html
    sum_of_sqr = 0
    for i in range(3):
        sum_of_sqr += H[i][0]**2 + H[i][1]**2 + H[i][2]**2

    H = H/sum_of_sqr #Normalizing

    #Get h'
    H_prime = np.dot(np.linalg.inv(K),H)

    #r3 = h1xh2
    r3_from_h = np.cross(H_prime[:,0], H_prime[:,1])
    #Rotation Matrix = h1 h2 h3
    R_from_H = np.array([H_prime[:,0],H_prime[:,1],r3_from_h]).T
    
    #Get the SVD
    U,S,Vt = np.linalg.svd(R_from_H)

    #Make the Determinant
    det_UVt = np.linalg.det(np.dot(U,Vt))
    intermediate_diagonal_matrix = np.array([[1,0,0],[0,1,0],[0,0,det_UVt]])
    #Estimated rotation matrix
    R = np.dot(U,np.dot(intermediate_diagonal_matrix,Vt))

    #Inverting to change the reference frame
    R = np.linalg.inv(R)
    
    #Get the Transformation matrix
    t = H_prime[:,2]/np.linalg.norm(H_prime[:,0])

    #Negate the transformation to invert the vector but also then align the frame
    t = np.dot(R,-t)

    return R, t
