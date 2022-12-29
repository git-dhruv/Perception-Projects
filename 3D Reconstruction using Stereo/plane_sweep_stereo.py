import numpy as np
import cv2


EPS = 1e-8


def backproject_corners(K, width, height, depth, Rt):
    """
    Backproject 4 corner points in image plane to the imaginary depth plane using intrinsics and extrinsics
    
    Hint:
    depth * corners = K @ T @ y, where y is the output world coordinates and T is the 4x4 matrix of Rt (3x4)

    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        width -- width of the image
        heigh -- height of the image
        depth -- depth value of the imaginary plane
    Output:
        points -- 2 x 2 x 3 array of 3D coordinates of backprojected points, here 2x2 correspinds to 4 corners
    """

    points = np.array(
        (
            (0, 0, 1),
            (width, 0, 1),
            (0, height, 1),
            (width, height, 1),
        ),
        dtype=np.float32,
    ).reshape(2, 2, 3)

    #Points is 3D coords - get it to X,y,Z
    points = points.reshape(4,3) 
    Lastrow = np.array([[0,0,0,1]])
    T = np.vstack((Rt,Lastrow))

    #inv(T)
    Tinv = np.linalg.inv(T)
    #Temp variable so that it looks cleaner
    tmp = Tinv[0:3,0:3]@(depth*np.linalg.inv(K))
    points = tmp@points.T + Tinv[0:3,-1].reshape(3,1)
    points = points.T
    points = points.reshape((2,2,3))
    
    return points


def project_points(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    
    Hint:
    Z * projections = K @ T @ p, where p is the input points and projections is the output, T is the 4x4 matrix of Rt (3x4)
    
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- points_height x points_width x 3 array of 3D points
    Output:
        projections -- points_height x points_width x 2 array of 2D projections
    """
    #make it a col vector of XYZ
    savedshape = points.shape
    points = points.reshape(points.shape[0]*points.shape[1],3).T
    points = np.vstack( (points,np.ones((1,points.shape[1]))) )
    
    projections = K@Rt@points

    points = projections/projections[-1,:] #This will be 3 rows and N cols

    points = np.delete(points,2,axis=0) #We delete Z row as it is one!!!!!

    points = (points.T).reshape(savedshape[0],savedshape[1],2)


    
    return points


def warp_neighbor_to_ref(
    backproject_fn, project_fn, depth, neighbor_rgb, K_ref, Rt_ref, K_neighbor, Rt_neighbor
):
    """
    Warp the neighbor view into the reference view
    via the homography induced by the imaginary depth plane at the specified depth

    Make use of the functions you've implemented in the same file (which are passed in as arguments):
    - backproject_corners
    - project_points

    Also make use of the cv2 functions:
    - cv2.findHomography
    - cv2.warpPerspective
    
    ! Note, when you use cv2.warpPerspective, you should use the shape (width, height), NOT (height, width)
    
    Hint: you should do the follows:
    1.) apply backproject_corners on ref view to get the virtual 3D corner points in the virtual plane
    2.) apply project_fn to project these virtual 3D corner points back to ref and neighbor views
    3.) use findHomography to get teh H between neighbor and ref
    4.) warp the neighbor view into the reference view

    Input:
        backproject_fn -- backproject_corners function
        project_fn -- project_points function
        depth -- scalar value of the depth at the imaginary depth plane
        neighbor_rgb -- height x width x 3 array of neighbor rgb image
        K_ref -- 3 x 3 camera intrinsics calibration matrix of reference view
        Rt_ref -- 3 x 4 camera extrinsics calibration matrix of reference view
        K_neighbor -- 3 x 3 camera intrinsics calibration matrix of neighbor view
        Rt_neighbor -- 3 x 4 camera extrinsics calibration matrix of neighbor view
    Output:
        warped_neighbor -- height x width x 3 array of the warped neighbor RGB image
    """

    height, width = neighbor_rgb.shape[:2]
    

    
    # cv2.findHomography and cv2.warpPerspective
    
    #Hint 1
    virtual3Dpts = backproject_fn(K_ref,width,height,depth,Rt_ref)

    #hint 2
    projectpointsNebor = project_fn(K_neighbor,Rt_neighbor,virtual3Dpts).reshape(-1,2)
    projectpointsRef = project_fn(K_ref,Rt_ref,virtual3Dpts).reshape(-1,2)

    #hint 3
    Homography,_ = cv2.findHomography(projectpointsNebor,projectpointsRef)
    #hint 4
    warped_neighbor = cv2.warpPerspective(neighbor_rgb,Homography,(width,height))

    
    
        
    
    return warped_neighbor


def zncc_kernel_2D(src, dst):
    """
    Compute the cost map between src and dst patchified images via the ZNCC metric

    IMPORTANT NOTES:
    - Treat each RGB channel separately but sum the 3 different zncc scores at each pixel

    - When normalizing by the standard deviation, add the provided small epsilon value,
    EPS which is included in this file, to both sigma_src and sigma_dst to avoid divide-by-zero issues

    Input:
        src -- height x width x K**2 x 3
        dst -- height x width x K**2 x 3
    Output:
        zncc -- height x width array of zncc metric computed at each pixel
    """
    assert src.ndim == 4 and dst.ndim == 4
    assert src.shape[:] == dst.shape[:]

    
    W1bar = np.mean(src,axis=2)
    W2bar = np.mean(dst,axis=2)
    
    W1std = np.std(src,axis=2)+EPS
    W2std = np.std(dst,axis=2)+EPS

    #Had to get the (x,y,1) shape rather than (x,y,) for adding the src and W1bar
    setting = (src.shape[0],src.shape[1],1)

    #Simply Set the RGB channels apart and then add them
    zncc = np.sum((src[:,:,:,0]-W1bar[:,:,0].reshape(setting))*(dst[:,:,:,0]-W2bar[:,:,0].reshape(setting)),axis=2)/(W1std[:,:,0]*W2std[:,:,0]) 
    zncc += np.sum((src[:,:,:,1]-W1bar[:,:,1].reshape(setting))*(dst[:,:,:,1]-W2bar[:,:,1].reshape(setting)),axis=2)/(W1std[:,:,1]*W2std[:,:,1]) 
    zncc += np.sum((src[:,:,:,2]-W1bar[:,:,2].reshape(setting))*(dst[:,:,:,2]-W2bar[:,:,2].reshape(setting)),axis=2)/(W1std[:,:,2]*W2std[:,:,2]) 
    
    

    return zncc  # height x width


def backproject(dep_map, K):
    """
    Backproject image points to 3D coordinates wrt the camera frame according to the depth map

    Input:
        K -- camera intrinsics calibration matrix
        dep_map -- height x width array of depth values
    Output:
        points -- height x width x 3 array of 3D coordinates of backprojected points
    """
    _u, _v = np.meshgrid(np.arange(dep_map.shape[1]), np.arange(dep_map.shape[0]))

    
    #borrowed from two_view_stereo
    xyz_cam = np.zeros((dep_map.shape[0],dep_map.shape[1],3))    
    
    x, y = np.meshgrid(np.arange(dep_map.shape[1]), np.arange(dep_map.shape[0]))

    x_cam =  ((x.flatten() - K[0,2])*dep_map.flatten()/K[0,0])
    y_cam = (y.flatten() - K[1,2])*dep_map.flatten()/K[1,1]

    joinedimage = np.stack([x_cam,y_cam,dep_map.flatten()]).T
    xyz_cam = joinedimage.reshape((dep_map.shape[0],dep_map.shape[1],3))

    
    return xyz_cam
