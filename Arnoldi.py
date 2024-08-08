import numpy as np
import scipy
from scipy.linalg import expm, norm
from scipy.sparse import diags, csc_matrix, csr_matrix, lil_matrix, lil_array
import scipy.sparse
from scipy.sparse.linalg import expm, expm_multiply

def Arnoldi(A, v_1, m):
    v = np.zeros((np.shape(v_1)[-1], m + 1))
    v[:,0] = v_1/np.linalg.norm(v_1)    
    h = lil_array((m + 1, m))
    for j in range(0,m):
        w = A @ v[:,j]
        for i in range(0,j):
            h[i,j] = np.dot(w, v[:,i])
            w -= h[i,j]*v[:,i]
            
        if j < m:
            h[j+1,j] = norm(w)
            v[:,j+1] = w/h[j+1,j]
        else:
            break

    return csc_matrix(h[0:m,0:m]), csc_matrix(v[:,0:m])

def ArnoldiExp(A, v_1, m):
    H, V = Arnoldi(A, v_1, m)
    e_1 = np.zeros((m))
    e_1[0] = 1
    return V@expm_multiply(H,e_1)*np.linalg.norm(v_1)


if __name__ == "__main__":
    n = 50
    m = 20

    v = np.random.rand(n)
    print(np.shape(v))

    A = diags([-1,2,-1], [-1,0,1], shape=(n,n))
    A = csc_matrix(A)
    print(np.shape(A))        

    print(ArnoldiExp(A, v, m))
    print(expm_multiply(A,v))