import numpy as np
from math import *
from copy import copy, deepcopy
from est_pixel_world import est_pixel_world

def P3P(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-3-Point problem, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """
    #Spagetti time
    #Small p is World coordinate
    #u,v is Pixel

    #Single Shot Callibration based on pinhole camera model
    P = (np.column_stack((Pc,np.ones((4,1)))).T)*K[0][0]


    j = np.dot(np.linalg.inv(K),P)
    j1,j2,j3 = j[:,0],j[:,1],j[:,2]
    
    #Everything below is derived from paper - Review and Analysis of Solutions of the Three Point Perspective Pose Estimation Problem
    j1 = j1/np.linalg.norm(j1)
    j2 = j2/np.linalg.norm(j2)
    j3 = j3/np.linalg.norm(j3)
    
    #Get cosines for cosine law
    cos_alpha = np.dot(j2,j3.T)
    cos_beta = np.dot(j1,j3.T)
    cos_gamma = np.dot(j1,j2.T)

    #
    a = np.linalg.norm(Pw[1,:]-Pw[2,:]) #p2-p3
    b = np.linalg.norm(Pw[0,:]-Pw[2,:])
    c = np.linalg.norm(Pw[0,:]-Pw[1,:])


    #Variable for something that was occuring a lot of times
    inter = (a*a - c*c)/b**2

    #Coeffs
    A4 = (inter-1)**2 - 4*c*c*(cos_alpha**2)/(b*b)
    A3 = 4*( (inter)*(1 - inter)*cos_beta - (1 - (a**2+c**2)/b**2)*cos_alpha*cos_gamma + 2*c*c*cos_alpha*cos_alpha*cos_beta/b**2 )


    
    A2 = 2*( (inter)**2 - 1 + 2*(inter*inter)*cos_beta*cos_beta + 2*(b**2 - c**2)*cos_alpha*cos_alpha/b**2 - 4*((a*a + c*c)/b**2)*cos_alpha*cos_beta*cos_gamma + 2*cos_gamma*cos_gamma*(b*b - a*a)/b**2 )
    A1 = 4*(   -(inter)*(1+inter)*cos_beta + 2*a*a*cos_gamma*cos_gamma*cos_beta/b**2 - (1 - (a*a + c*c)/b**2)*cos_alpha*cos_gamma    )
    A0 =  (1+inter)**2 - 4*a*a*cos_gamma**2/b**2

    #Get the roots
    all_roots = np.roots([A4, A3, A2, A1, A0])

    #Get real roots
    all_roots_real = all_roots.real[abs(all_roots.imag)<1e-5]

    #Get positive real roots
    roots_real_postive = all_roots_real[all_roots_real>0]

    #Minimization of error
    min_error = 0
    i = 0 #lets track index as well

    #Get rotations and translations
    R_final = np.eye(3)
    t_final = np.ones((3,))

    #For each root remaininig - should be 2
    for v in roots_real_postive:
        u = (-1+inter)*v*v - 2*cos_beta*v*(inter) + 1 + inter
        u = u/(2*(cos_gamma-v*cos_alpha))
        
        #get depth 1
        denom = u*u + v*v - 2*u*v*cos_alpha
        s1 = sqrt(a*a/denom)
        #other depths
        s2 = u*s1
        s3 = v*s1

        #Reproject - Recalculated again for debugging purpose
        norm_p1 = np.linalg.norm(np.array([Pc[0,0] - K[0][2],Pc[0,1] - K[1][2],K[0][0]]))
        norm_p2 = np.linalg.norm(np.array([Pc[1,0] - K[0][2],Pc[1,1] - K[1][2],K[0][0]]))
        norm_p3 = np.linalg.norm(np.array([Pc[2,0] - K[0][2],Pc[2,1] - K[1][2],K[0][0]]))
        norm_p4 = np.linalg.norm(np.array([Pc[3,0] - K[0][2],Pc[3,1] - K[1][2],K[0][0]]))


        #Points
        A = np.array([[s1*(Pc[0,0]-K[0][2])/norm_p1,s1*(Pc[0,1]-K[1][2])/norm_p1,s1*K[0][0]/norm_p1],
                    [s2*(Pc[1,0]-K[0][2])/norm_p2,s2*(Pc[1,1]-K[1][2])/norm_p2,s2*K[0][0]/norm_p2],
                    [s3*(Pc[2,0]-K[0][2])/norm_p3,s3*(Pc[2,1]-K[1][2])/norm_p3,s3*K[0][0]/norm_p3]])

        #Get R and t
        R,t = Procrustes(A,Pw[0:3])


        #
        Pc_projected = Pw[-1,:]

        #Calibrated coordinates
        Pc_real = K[0][0]*np.dot(np.linalg.inv(K),np.array([Pc[-1,0], Pc[-1,1], 1]).T)
        Pc_real = np.dot(R,Pc_real)+t

        #Error between pixels and world         
        error = np.linalg.norm((Pc_projected - Pc_real))

        if i==0:
            min_error = error
            R_final = deepcopy(R)
            t_final = deepcopy(t)
        else:
            if error<min_error:
                error = min_error
                R_final = deepcopy(R)
                t_final = deepcopy(t)

        i += 1

    #Take the final Rs and ts
    R = R_final
    t = t_final


    return R, t

def Procrustes(X, Y):
    """
    Solve Procrustes: Y = RX + t

    Input:
        X: Nx3 numpy array of N points in camera coordinate (returned by your P3P)
        Y: Nx3 numpy array of N points in world coordinate
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """

    A = deepcopy(Y)
    B = deepcopy(X)

    B_bar = np.mean(B, axis = 0)
    A_bar = np.mean(A, axis = 0)

    B = np.transpose(B - B_bar)
    A = np.transpose(A - A_bar)

    U,_,Vt = np.linalg.svd(np.dot(A,B.T))

    det_UVt = np.linalg.det(np.dot(Vt.T,U.T))
    intermediate_diagonal_matrix = np.array([[1,0,0],[0,1,0],[0,0,det_UVt]])
    
    R = np.dot(U,np.dot(intermediate_diagonal_matrix,Vt))

    t = A_bar - np.dot(R,B_bar)



    return R, t
