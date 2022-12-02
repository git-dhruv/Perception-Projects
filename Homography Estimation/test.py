import numpy as np

X = np.array([[,],[,],[,],[,]])
X_prime = np.array([[,],[,],[,],[,]])
A = None
#Make the matrix 
for i in range(4):
    ax = np.array([-X[i][0],-X[i][1], -1, 0,0,0, X[i][0]*X_prime[i][0], X[i][1]*X_prime[i][0], X_prime[i][0] ])
    ay = np.array([0,0,0, -X[i][0],-X[i][1], -1, X[i][0]*X_prime[i][1], X[i][1]*X_prime[i][1], X_prime[i][1] ]) 
    if A is not None:
        A = np.concatenate([A,[ax],[ay]])
    else:
        A = np.concatenate([[ax],[ay]])

#SVD
[U, S , Vt ] = np.linalg.svd(A)
#Directly taking rows instead of transposing
Vt = Vt[-1,:].reshape((3, 3))

print(Vt)