import numpy as np
import control as ctrl


def proximal_gradient(m,L):
    alpha = 2/(m+L)

    A = 1
    B = np.asarray([[-alpha, -alpha]])
    C = np.asarray([[1],[1]])
    D = np.asarray([[0,0],[-alpha,-alpha]])

    p, q = 1, 1

    return ctrl.ss(A,B,C,D,dt=1), p, q


def proximal_heavy_ball(m,L):
    alpha = (2/(np.sqrt(L)+np.sqrt(m)))**2
    beta = (np.sqrt(L/m)-1) / (np.sqrt(L/m)+1)

    A = np.asarray([[1+beta, -beta], 
                    [1,          0]])
    B = np.asarray([[-alpha, -alpha], 
                    [0,       0]])
    C = np.asarray([[1,          0],
                    [1+beta, -beta]])
    D = np.asarray([[0,       0],
                    [-alpha, -alpha]])
    
    p, q = 1, 1

    return ctrl.ss(A,B,C,D,dt=1), p, q


def proximal_nesterov(m,L): 
    alpha = 1/L
    beta = (np.sqrt(L/m)-1) / (np.sqrt(L/m)+1)

    A = np.asarray([[1+beta, -beta], 
                    [1,          0]])
    B = np.asarray([[-alpha, -alpha], 
                    [0,           0]])
    C = np.asarray([[1+beta, -beta],
                    [1+beta, -beta]])
    D = np.asarray([[0,       0],
                    [-alpha, -alpha]])
    
    p, q = 1, 1

    return ctrl.ss(A,B,C,D,dt=1), p, q


def proximal_triple_momentum(m,L):
    rho = 1 - 1/np.sqrt(L/m)

    alpha = (1+rho)/L
    beta = rho**2/(2-rho)
    gamma = rho**2/((1+rho)*(2-rho))

    A = np.asarray([[1+beta, -beta], 
                    [1,          0]])
    B = np.asarray([[-alpha, -alpha], 
                    [0,           0]])
    C = np.asarray([[1+gamma, -gamma],
                    [1+gamma, -gamma]])
    D = np.asarray([[0,       0],
                    [-alpha, -alpha]])

    p, q = 1, 1

    return ctrl.ss(A,B,C,D,dt=1), p, q


def accelerated_ogd(m,L):
    alpha = 1/L
    gamma = 1/L
    tau   = gamma*alpha

    A = np.asarray([[tau, 1-tau], 
                    [0,       1]])
    
    B = np.asarray([[-gamma, -gamma, 0], 
                    [-alpha, 0, -alpha]])
    
    C = np.asarray([[tau, 1-tau], 
                    [tau, 1-tau], 
                    [0,       1]])
    
    D = np.asarray([[0, 0, 0], 
                    [-gamma, -gamma, 0],
                    [-alpha, 0, -alpha]])

    p, q = 1, 2

    return ctrl.ss(A,B,C,D,dt=1), p, q


def multi_step_ogd(m,L,K):
    alpha = 2/(m+L)

    block1 = np.tril(-alpha * np.ones((K, K)), k=-1)
    block2 = np.tril(-alpha * np.ones((K, K)), k=0)

    A = 1
    B = np.full((1, 2*K), -alpha)
    C = np.full((2*K,1), 1)
    D = np.block([[block1, block1],
                  [block2, block2]])
    
    p, q = K, K

    return ctrl.ss(A,B,C,D,dt=1), p, q
