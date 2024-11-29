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

    for i in range(n_de):
        for j in range(n_de):
            K_de_de[i, j] = K(de_data[i], de_data[j])
            

    for i in range(n_de):
        for j in range(n_nu):
            K_de_nu[i, j] = K(de_data[i], nu_data[j])

    one_nu = np.ones(n_nu).reshape(-1, 1)
    scalar = n_de/n_nu

    K_inv = sp.inv(K_de_de + reg * np.eye(n_de)) 

    tmp_computation = K_inv @ K_de_nu
    tmp_computation = tmp_computation @ one_nu
    r_de_estim = scalar * tmp_computation

    return r_de_estim


#def IWCV(parametric_family, dimension_of_parameter, loss_function, x_test, training_set, gamma , lamb):
def IWCV(parametric_family, loss_function, x_test, training_set, gamma , lamb):
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
            print(f"got reg = {reg}, sum = {sum}, loss_at_i ={loss_at_i}")
            raise TypeError(f"Unexpected NaN in get_generalization_error: got reg = {reg}, sum = {sum}, loss_at_i ={loss_at_i}")
        return result
    
    def minimize_given_parameters(gamma, lamb):
        return lambda theta: get_generalization_error(theta, gamma, lamb)
    
    G_theta = minimize_given_parameters(gamma, lamb)     #depends only on theta

    optim_result = minimize(G_theta, [0])

    print(optim_result)
    theta_optimal = optim_result.x
    return theta_optimal