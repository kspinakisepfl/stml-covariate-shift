import numpy as np
import scipy.linalg as sp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

Gaussian = lambda x,y: 1/np.sqrt(2*np.pi)*np.exp(-(x-y)**2/2)
Default_psi = lambda x, de_sample: [Gaussian(x, i) for i in de_sample]



def KMM_inf_approach_w_nonneg(de_data, nu_data, method, K=Gaussian):
    if method== "L-BFGS-B":
        reg = 10e-3
    else:
        reg = 1
    n_de = len(de_data)
    n_nu = len(nu_data)
    
    # Kernel matrices
    K_de_de = np.array([[K(de_data[i], de_data[j]) for j in range(n_de)] for i in range(n_de)]) * reg
    K_de_nu = np.array([[K(de_data[i], nu_data[j]) for j in range(n_nu)] for i in range(n_de)]) * reg
    
    one_nu = np.ones((n_nu, 1))
    
    # Solve the constrained optimization
    def objective(w):
        w = np.array(w).reshape(-1, 1)
        diff = ((1 / n_de**2) * np.transpose(w) @ K_de_de @ w) - ((2 /( n_nu*n_de)) * np.transpose(w) @ K_de_nu @ one_nu)
        return diff

    # Initial weights
    w_init = np.ones(n_de) / n_de
    
    # Constraints: non-negativity
    bounds = [(0, None) for _ in range(n_de)]
    
    # Optimization
    result = minimize(objective, w_init, bounds=bounds, method=method)
    print(result)

    return result.x
