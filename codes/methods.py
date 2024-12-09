import numpy as np
import scipy.linalg as sp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

Gaussian = lambda x,y: 1/np.sqrt(2*np.pi)*np.exp(-(x-y)**2/2)
Default_psi = lambda x, de_sample: [Gaussian(x, i) for i in de_sample]


def KMM_simple(de_data, nu_data, K = Gaussian):
    reg = 10e-2
    n_de = len(de_data)     
    n_nu = len(nu_data) 

    K_de_de = np.zeros((n_de, n_de))
    K_de_nu = np.zeros((n_de, n_nu))

    #for i in range(n_de):
    #    for j in range(n_de):
    #        K_de_de[i, j] = K(de_data[i], de_data[j])
           
    #for i in range(n_de):
    #    for j in range(n_nu):
    #        K_de_nu[i, j] = K(de_data[i], nu_data[j])
    K_de_de = [[K(de_data[i], de_data[j]) for j in range(n_de)] for i in range(n_de)]
    K_de_nu = [[K(de_data[i], nu_data[j]) for j in range(n_nu)] for i in range(n_de)]
    
    one_nu = np.ones(n_nu).reshape(-1, 1)
    scalar = n_de/n_nu

    K_inv = sp.inv(K_de_de + reg * np.eye(n_de)) 

    tmp_computation = K_inv @ K_de_nu
    tmp_computation = tmp_computation @ one_nu
    r_de_estim = scalar * tmp_computation

    return r_de_estim


def IWCV(parametric_family, dim_theta, loss_function, x_test, training_set, gamma , lamb):
    # implementation of LOO IWCV with density ratio estimation using infinite-order KMM 

    n_tr = len(training_set[0,:])
    x_tr, y_tr = training_set[0, :], training_set[1, :]
    x_te = x_test

    r_estim = KMM_simple(x_te, x_tr)

    def get_generalization_error(theta, gamma, lamb):
        sum = 0
        loss_at_i = 0

        #flattened sum
        for i in range(n_tr):
            loss_at_i = loss_function(parametric_family(x_tr[i], theta), y_tr[i])
            sum += (np.abs(r_estim[i])**gamma) * loss_at_i

        #regularization term
        reg = lamb * np.linalg.norm(theta)**2
        result = sum/n_tr + reg
        if np.isnan(result): 
            print(f"got reg = {reg}, sum = {sum}, loss_at_i = {loss_at_i}")
            raise TypeError(f"Unexpected NaN in get_generalization_error: got reg = {reg}, sum = {sum}, loss_at_i ={loss_at_i}")
        return result
    
    def minimize_given_parameters(gamma, lamb):
        return lambda theta: get_generalization_error(theta, gamma, lamb)
    
    G_theta = minimize_given_parameters(gamma, lamb)     #depends only on theta

    #optim_result = minimize(G_theta, np.ones(dim_theta))
    optim_result = minimize(G_theta, np.zeros(dim_theta))

    print(optim_result)
    theta_optimal = optim_result.x
    f_opti= lambda x: parametric_family(x, theta_optimal)
    return theta_optimal, f_opti







def KMM_fixed_design(phi, de_data, nu_data):
    phi_nu = np.array(np.zeros((1, len(nu_data))))   # 1 x n_nu
    phi_de = np.array(np.zeros((1, len(de_data))))   # 1 x n_de

    for i in range(len(de_data)):
        phi_de[:, i] = phi(de_data[i])

    for i in range(len(nu_data)):
        phi_nu[:, i] = phi(nu_data[i])   

    one_nu = np.array(np.ones((len(nu_data), 1)))  # n_nu x 1

    scalar = len(de_data) / len(nu_data) # 1 x 1
    
    reg = 10e-4
    phi_inv = np.linalg.inv(phi_de.T @ phi_de + reg * np.eye(len(de_data)))  # ( n_de x 1) x (1 x n_de) = n_de x n_de
    
    # Compute remaining terms
    phi_trans_mult = phi_de.T @ phi_nu  # (n_de x 1) x (1 x n_nu) = n_de x n_nu 
    r_de = scalar * phi_inv @ phi_trans_mult @ one_nu   # (n_de x n_de) x (n_de x n_nu) x (n_nu x 1) = n_de x 1
    
    return r_de


    

