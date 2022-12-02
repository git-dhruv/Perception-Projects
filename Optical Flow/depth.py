import numpy as np
import numpy as np

def depth(flow, confidence, ep, K, thres=10):
    """
    params:
        @flow: np.array(h, w, 2)
        @confidence: np.array(h, w, 2)
        @K: np.array(3, 3)
        @ep: np.array(3,) the epipole you found epipole.py note it is uncalibrated and you need to calibrate it in this function!
    return value:
        depth_map: np.array(h, w)
    """
    depth_map = np.zeros_like(confidence)


    #Make linear vectors
    xp = np.arange(0,512)
    yp = np.arange(0,512)
    #Make a meshgrid like a (x,y) coordinate system
    xp,yp = np.meshgrid(xp,yp)
    #Flatten them 
    xp = xp.flatten()
    yp = yp.flatten()
    
    #Make the p vector like [x,y,1]
    p = np.column_stack((xp,yp,np.ones((len(xp),1)))).T
    
    #Callibration first
    ep = np.linalg.inv(K)@ep
    p = np.linalg.inv(K)@p

    #Calibrating flow vector -- doesn't work if I remove offset
    Vx = (flow[:,:,0].flatten())/K[0,0]
    Vy = (flow[:,:,1].flatten())/K[1,1]
    #Getting Vzx and Vzy - ideally should be close
    Vzx = Vx/ep[0]
    Vzy = Vy/ep[1]
    #Flow vector for norm calculation
    flow_vec = np.column_stack((Vx,Vy)).T

    #This is the Numerator
    num = ((p[0,:]-ep[0]))**2 + ((p[1,:]-ep[1]))**2
    #Denometer 
    den = np.linalg.norm(flow_vec,axis=0)
    #Depth Calculation
    depth = np.sqrt(num.T)/den
    _sizex = confidence.shape[0]
    _sizey = confidence.shape[1]

    confidence = confidence.flatten()

    #Adjustments
    depth = np.array([float(depth[i]) if confidence[i]>thres else 0 for i in range(len(depth))])
    #Reshape
    depth_map = depth.reshape(_sizex, _sizey)
    

    truncated_depth_map = np.maximum(depth_map, 0)
    valid_depths = truncated_depth_map[truncated_depth_map > 0]
    # You can change the depth bound for better visualization if your depth is in different scale
    depth_bound = valid_depths.mean() + 10 * np.std(valid_depths)
    # print(f'depth bound: {depth_bound}')

    truncated_depth_map[truncated_depth_map > depth_bound] = 0
    truncated_depth_map = truncated_depth_map / truncated_depth_map.max()
    

    return truncated_depth_map
