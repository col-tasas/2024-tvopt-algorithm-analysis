import numpy as np
import control as ctrl


def gradient_descent(m,L):
    alpha = 2/(m+L)

    A = 1
    B = -alpha
    C = 1
    D = 0

    p, q = 1, 0

    return ctrl.ss(A,B,C,D,dt=1), p, q

def heavy_ball(m,L):
    alpha = (2/(np.sqrt(L)+np.sqrt(m)))**2
    beta = (np.sqrt(L/m)-1) / (np.sqrt(L/m)+1)

    A = np.asarray([[1+beta, -beta], [1, 0]])
    B = np.asarray([[-alpha], [0]])
    C = np.asarray([[1, 0]])
    D = 0

    p, q = 1, 0

    return ctrl.ss(A,B,C,D,dt=1), p, q


def nesterov(m,L): 
    alpha = 1/L
    beta = (np.sqrt(L/m)-1) / (np.sqrt(L/m)+1)

    A = np.asarray([[1+beta, -beta], [1, 0]])
    B = np.asarray([[-alpha], [0]])
    C = np.asarray([[1+beta, -beta]])
    D = 0

    p, q = 1, 0

    return ctrl.ss(A,B,C,D,dt=1), p, q


def triple_momentum(m,L):
    rho = 1 - 1/np.sqrt(L/m)

    alpha = (1+rho)/L
    beta = rho**2/(2-rho)
    gamma = rho**2/((1+rho)*(2-rho))

    A = np.asarray([[1+beta, -beta], 
                    [1,          0]])
    B = np.asarray([[-alpha], 
                    [0     ]])
    C = np.asarray([[1+gamma, -gamma]])
    D = 0

    p, q = 1, 0

    return ctrl.ss(A,B,C,D,dt=1), p, q


def multi_step_gradient(m,L,K):
    alpha = 2/(m+L)

    A = 1
    B = np.full((1, K), -alpha)
    C = np.full((K,1), 1)
    D = np.tril(-alpha * np.ones((K, K)), k=-1)

    p, q = K, 0

    return ctrl.ss(A,B,C,D,dt=1), p, q