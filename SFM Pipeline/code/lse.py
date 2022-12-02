import numpy as np

def least_squares_estimation(X1, X2):
  """
  Just to clarify things
  X1,X2 callibrated camera coordinates: aka 2D without intrinsics
  Essential matrix is for camera callibrated coords so we are good.

  """
  
  #Let's make Ax=0
  A = np.zeros((len(X1),9))

  # @TODO: Vectorize
  # Inefficient way to populate 
  for i in range(len(X1)):
      A[i][0] = X1[i,0] * X2[i,0]
      A[i][1] = X1[i,0] * X2[i,1]
      A[i][2] = X1[i,0] * X2[i,2]
      A[i][3] = X1[i,1] * X2[i,0]
      A[i][4] = X1[i,1] * X2[i,1]
      A[i][5] = X1[i,1] * X2[i,2]
      A[i][6] = X1[i,2] * X2[i,0]
      A[i][7] = X1[i,2] * X2[i,1]
      A[i][8] = X1[i,2] * X2[i,2]

  #Get a least square estimate of A
  U,S,Vt = np.linalg.svd(A)
  E_est = (Vt[-1,:].reshape(3,3)).T

  #Conditioning - take singular values and condition the estimate
  U,S,Vt = np.linalg.svd(E_est)
  Sing_vals = np.eye(3)
  Sing_vals[-1,-1] = 0
  E = U@Sing_vals@Vt

  return E
