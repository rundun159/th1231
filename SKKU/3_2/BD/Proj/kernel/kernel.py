import numpy as np
import math
from scipy.spatial import distance
def get_Hamming_Dist(x1,x2):
    ret = np.zeros(shape=(len(x1), len(x2)), dtype=np.float)
    for idx1, _x1 in enumerate(x1):
        for idx2, _x2 in enumerate(x2):
            ret[idx1][idx2]=np.sum(_x1==_x2)
    return ret

def ret_custom_rbf(gamma):
    def custom_rbf(x1, x2):
        ret = np.zeros(shape=(len(x1),len(x2)),dtype=np.float)
        for idx1, _x1 in enumerate(x1):
            for idx2, _x2 in enumerate(x2):
                diff = _x1-_x2
                ret[idx1][idx2]=math.exp(gamma*-1*np.matmul(diff,np.transpose(diff)))
        return ret
    return custom_rbf

def ret_custom_poly(gamma,coef,degree):
    def custom_poly(x1, x2):
        return np.power(gamma*np.matmul(x1,np.transpose(x2)+coef),degree)
    return custom_poly

def ret_TH_custom_kernel(gamma1,gamma2,coef,degree,prac):
    def TH_custom_kernel(x1,x2):
        TH_rbf = ret_custom_rbf(gamma=gamma1)
        TH_poly = ret_custom_poly(gamma=gamma2,coef=coef,degree=degree)
        return TH_rbf(x1,x2)*prac + (1-prac)*TH_poly(x1,x2)
    return TH_custom_kernel

def ret_Cosine_Dist(x1,x2):
    ret = np.zeros(shape=(len(x1), len(x2)), dtype=np.float)
    for idx1, _x1 in enumerate(x1):
        for idx2, _x2 in enumerate(x2):
            ret[idx1][idx2]=distance.cosine(_x1,_x2)
    return 1-ret

def interpolation_HAM_COS(prac=0.5):
    def ret_interpolation(x1,x2):
        return prac*get_Hamming_Dist(x1,x2)+(1-prac)*ret_Cosine_Dist(x1,x2)
    return ret_interpolation

