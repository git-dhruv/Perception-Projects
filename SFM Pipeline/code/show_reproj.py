import numpy as np
import matplotlib.pyplot as plt

def show_reprojections(image1, image2, uncalibrated_1, uncalibrated_2, P1, P2, K, T, R, plot=True):


  """
  We have imag1, imag2, their uncallibrated points - > aka 
  P1, P2 are 3D pts with depths I guess

  """

  # Homogenous Coordinates
  P1 = np.vstack((P1.T,np.ones((1,len(P1)))))
  P2 = np.vstack((P2.T,np.ones((1,len(P2)))))

  #Extrinsic for first cam
  extrinsic_1 = np.column_stack((R,T))
  P_1 = K@extrinsic_1

  #Extrinsic for second cam
  extrinsic_2 = np.column_stack((R.T,-R.T@T))
  P_2 = K@extrinsic_2
  
  #Reproject
  P2proj = (P_2@P2).T
  P1proj =( P_1@P1).T
  

  if (plot):
    plt.figure(figsize=(6.4*3, 4.8*3))
    ax = plt.subplot(1, 2, 1)
    ax.set_xlim([0, image1.shape[1]])
    ax.set_ylim([image1.shape[0], 0])
    plt.imshow(image1[:, :, ::-1])
    plt.plot(P2proj[:, 0] / P2proj[:, 2],
           P2proj[:, 1] / P2proj[:, 2], 'bs')
    plt.plot(uncalibrated_1[0, :], uncalibrated_1[1, :], 'ro')

    ax = plt.subplot(1, 2, 2)
    ax.set_xlim([0, image1.shape[1]])
    ax.set_ylim([image1.shape[0], 0])
    plt.imshow(image2[:, :, ::-1])
    plt.plot(P1proj[:, 0] / P1proj[:, 2],
           P1proj[:, 1] / P1proj[:, 2], 'bs')
    plt.plot(uncalibrated_2[0, :], uncalibrated_2[1, :], 'ro')
    
  else:
    return P1proj, P2proj