import numpy as np
from numpy import sin,cos,pi

def rotz(a):
  return np.array([[cos(a), -sin(a), 0],[sin(a), cos(a), 0],[0,0,1]])

def pose_candidates_from_E(E):
  transform_candidates = []


  U,_,Vt = np.linalg.svd(E)
  
  transform_candidates=[
    {
    'T': U[:,-1],
    'R': U@(rotz(pi/2).T@Vt)
    },
    {
      'T': U[:,-1],
      'R':  U@(rotz(-pi/2).T@Vt)
    },
    {
      'T':-U[:,-1],
      'R': U@(rotz(pi/2).T@Vt)
    },
    {
      'T':-U[:,-1],
      'R': U@(rotz(-pi/2).T@Vt)

    }
  ]
  

  return transform_candidates
