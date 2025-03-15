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
    for j in range(1,m+1):
        w = A @ v[:,j-1]
        for i in range(0,j):
            h[i,j-1] = np.dot(w, v[:,i])
            w -= h[i,j-1]*v[:,i]
        h[j,j-1] = norm(w)
    #    if h[j+1,j] < 1e-10:
    #        break
        v[:,j] = w/h[j,j-1]

    # return csc_matrix(h[0:j,0:j]), csc_matrix(v[:,0:j]), norm(v_1)
    return csc_matrix(h[0:m,0:m]), csc_matrix(v[:,0:m]), norm(v_1)

def ArnoldiExp(A, v_1, m):
    H, V, beta = Arnoldi(A, v_1, m)
    if H.shape[0] == 0:
        return np.zeros_like(v_1)
    e_1 = np.zeros((H.shape[0]))
    e_1[0] = 1
    ret = V@expm_multiply(H,e_1)*beta
    # print(v_1.dot(v_1), ret.dot(ret))
    return ret # V@expm_multiply(H,e_1)*np.linalg.norm(v_1)


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
