import numpy as np
import cv2
import pdb
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_flow(image, flow_image, confidence, threshmin=10):
    """
    params:
        @img: np.array(h, w)
        @flow_image: np.array(h, w, 2)
        @confidence: np.array(h, w)
        @threshmin: confidence must be greater than threshmin to be kept
    return value:
        None
    """

    """
    STUDENT CODE BEGINS
    """
    #Size of Image
    imgy = image.shape[0]
    imgx = image.shape[1]

    #Data loggers
    flow_x,flow_y = [],[]
    x = []
    y = []

    
    for Y in range(imgy):
        for X in range(imgx):
            #If confidence exceeds threshold
            if confidence[Y,X]>threshmin:
                #Get the Flow 
                xFlow,yFlow = flow_image[Y,X]

                #Store the flow
                flow_x.append(xFlow)
                flow_y.append(yFlow)

                #Store the index of flow
                x.append(X)
                y.append(Y)

    #Go with the flow
    flow_x = np.array(flow_x)
    flow_y = np.array(flow_y)
    """
    STUDENT CODE ENDS
    """
    
    plt.imshow(image, cmap='gray')
    plt.quiver(x, y, (flow_x*10).astype(int), (flow_y*10).astype(int), 
                    angles='xy', scale_units='xy', scale=1., color='red', width=0.001)
    
    return





    

