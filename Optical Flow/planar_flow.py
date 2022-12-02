import numpy as np

def compute_planar_params(flow_x, flow_y, K,
                                up=[256, 0], down=[512, 256]):
    """
    params:
        @flow_x: np.array(h, w)
        @flow_y: np.array(h, w)
        @K: np.array(3, 3)
        @up: upper left index [i,j] of image region to consider.
        @down: lower right index [i,j] of image region to consider.
    return value:
        sol: np.array(8,)
    """

    #Normalizing flow vec and then cropping     
    xdot = flow_x[up[0]:down[0],up[1]:down[1]].flatten()/K[0,0]
    ydot = flow_y[up[0]:down[0],up[1]:down[1]].flatten()/K[1,1]

    #Make linear vectors
    xp = np.arange(0,len(flow_x))
    yp = np.arange(0,len(flow_x))
    #Make a meshgrid like a (x,y) coordinate system
    xp,yp = np.meshgrid(xp,yp)

    #Flatten them, crop them and reshape them
    xp = xp[up[0]:down[0], up[1]:down[1]].reshape(len(xdot),1)
    yp = yp[up[0]:down[0], up[1]:down[1]].reshape(len(xdot),1)

    #Callibrating the points    
    p = np.column_stack((xp,yp,np.ones((len(xp),1)))).T
    p = np.linalg.inv(K)@p

    #Again getting them back
    xp = p[0,:].flatten()
    yp = p[1,:].flatten()

    #Make an Ax=b matrix
    """
    B should be intended as | xdot1 |
                            | ydot1 |
                            | xdot2 |

    A should be all those x2 and y2
    """
    #We are reshaping again 
    xdot = xdot.reshape(len(xdot),1)
    ydot = ydot.reshape(len(xdot),1)
    xp = xp.reshape(len(xp),1)
    yp = yp.reshape(len(xp),1)

    #preparing matrices for Linear system of equation

    # a1x2 + a2xy + a3x + a4y + a5
    # a2y2 + a1xy + a6y + a7x + a8
    b = np.column_stack((xdot,ydot)).flatten()
    #Making A matrix
    row1 = np.column_stack( (xp**2,xp*yp,xp,yp,np.ones((len(xp),1)),np.zeros((len(xp),1)),np.zeros((len(xp),1)),np.zeros((len(xp),1))   ) )
    row2 = np.column_stack( (xp*yp,yp**2,np.zeros((len(xp),1)),np.zeros((len(xp),1)),np.zeros((len(xp),1)),yp,xp,np.ones((len(xp),1))   ) )
    A = np.column_stack((row1,row2)).flatten()
    A = A.reshape(len(b),8)
    #Answer 
    ans = np.linalg.lstsq(A,b,rcond=False)[0]
    # ans = np.dot(np.linalg.pinv(A),b)
    sol = np.array([float(i) for i in ans])


    return sol
    
    
