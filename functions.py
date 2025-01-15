import numpy as np
import scipy.linalg as sp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

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


def KMM_inf_approach_w_nonneg(de_data, nu_data, method, K=Gaussian):
    
    reg = 10e-1
    n_de = len(de_data)
    n_nu = len(nu_data)
    
    # Kernel matrices
    K_de_de = np.array([[K(de_data[i], de_data[j]) for j in range(n_de)] for i in range(n_de)]) +np.eye(n_de)* reg
    K_de_nu = np.array([[K(de_data[i], nu_data[j]) for j in range(n_nu)] for i in range(n_de)])
    
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


def IWERM(parametric_family, dim_theta, loss_function, x_test, training_set, gamma , lamb):
    # implementation of LOO IWCV with density ratio estimation using infinite-order KMM 

    n_tr = len(training_set[0,:])
    x_tr, y_tr = training_set[0, :], training_set[1, :]
    x_te = x_test

    r_estim = KMM_inf_approach_w_nonneg(x_te, x_tr)

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


def IWCV(parametric_family, dim_theta, loss_function, x_test, training_set, n_folds,gamma):
    # implementation of k-fold/LOO IWCV with density ratio estimation using infinite-order KMM 

    x_tr, y_tr = training_set[0, :], training_set[1, :]
    x_te = x_test
    kf = KFold(n_splits = n_folds)

    r_estim = KMM_simple(x_te, x_tr)


    def f_estim(x,y,loss_function):

        def loss(theta,x,y,loss_function):
            predictions = parametric_family(x, theta)
            return loss_function(predictions,y)
        
        def loss_minim(x,y,loss_function):
            return lambda theta: loss(theta,x,y,loss_function)
        
        loss_minimz = loss_minim(x,y,loss_function)

        optim_result = minimize(loss_minimz, np.zeros(dim_theta))
        theta_optimal = optim_result.x
        return theta_optimal

    sum = 0

    #flattened sum
    for (t_train,t_test) in kf.split(x_tr):
        abs_Z_i = len(t_test)
        theta_f_hat = f_estim(x_tr[t_train],y_tr[t_train],loss_function)

        loss_x_ratio_Zi = np.array([np.abs(r_estim[j])**gamma * loss_function(parametric_family(x_tr[j], theta_f_hat), y_tr[j]) for j in t_test])
        sum += np.sum(loss_x_ratio_Zi)/abs_Z_i
    
    G_hat = sum/n_folds
    return G_hat
