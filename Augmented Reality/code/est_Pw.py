import numpy as np

def est_Pw(s):
    """
    Estimate the world coordinates of the April tag corners, assuming the world origin
    is at the center of the tag, and that the xy plane is in the plane of the April
    tag with the z axis in the tag's facing direction. See world_setup.jpg for details.
    Input:
        s: side length of the April tag

    Returns:
        Pw: 4x3 numpy array describing the world coordinates of the April tag corners
            in the order of a, b, c, d for row order. See world_setup.jpg for details.

    """

    half_length = s/2

    #Calculating corners in order of a,b,c,d
    Pw = np.array([[-half_length,-half_length,0],
                [half_length,-half_length,0],
                [half_length,half_length,0],
                [-half_length,half_length,0]])                

    return Pw
