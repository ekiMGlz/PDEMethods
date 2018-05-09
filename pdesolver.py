import numpy as np
from scipy.sparse.linalg import inv
from scipy.sparse import diags, bmat, eye

def explicitHeat(eq, Ix, It, M = 1e3, N = 1e3):
    """
    DESCRICPCION DEL METODO
    
    Parameters
    ----------
    eq : dictionary
        Description of the equation, containing:
            =========  =============================================
            Key        Value
            =========  =============================================
            D          int - Diffusion Coeficient
            ic         function - Initial Condition
            bcL        funciton - Left Boundry Condition (Dirichlet)
            bcR        function - Right Boundry Condition (Neumann)
            =========  =============================================
        
    Ix : tuple (x0 xf)
        The interval (x0, xf) of distance over which to approximate the solution.
    It : tuple (t0 tf)
        The interval (t0, tf) of time over which to approximate the solution.
    M : int, optional
        Number of steps for space in the approximation. Default is 1e3.
    N : int, optional
        Number of steps for time in the approximation. Default is 1e3.
    
    Returns
    -------
    W : (M + 1, N + 1) double ndarray
        Matrix with the approximations of the PDE.
    X : (M + 1) double ndarray
        Times for x
    T : (N + 1) double ndarray
        Times for t
    """
    
    X = np.linspace(Ix[0],Ix[1], M + 1)
    h = (Ix[1]-Ix[0])/M
    
    T = np.linspace(It[0],It[1], N + 1)
    k = (It[1]-It[0])/N
    
    
    W = np.zeros((M + 1, N + 1))
    W[:,0] = eq['ic'](X)
    
    
    sigma = eq['D']*k/(h**2)
    
    
    #Matriz A del lado derecho
    d = [[0]+[1-2*sigma]*(M-1)+[0], \
         [sigma]*(M-1) + [0], \
         [0] + [sigma]*(M-1)]
    A = diags(d, [0,-1,1], (M+1, M+1))
    
    
    #Invesa de la matriz I del lado izquierdo de la igualdad
    I = np.eye(M+1)
    I[M, M-2] = -1/3
    I[M, M-1] = 4/3
    I[M, M] = 2*h/3
    
    for i in range(N):
        b = np.zeros(M+1)
        b[0] = eq['bcL'](T[i+1])
        b[-1] = eq['bcR'](T[i+1])
        
        b = b + A.dot(W[:,i])
        W[:,i+1] = I.dot(b)
        
    return W, X, T

def implicitHeat(eq, Ix, It, M = 1e3, N = 1e3):
    """
    DESCRICPCION DEL METODO
    
    Parameters
    ----------
    eq : dictionary
        Description of the equation, containing:
            =========  =============================================
            Key        Value
            =========  =============================================
            D          int - Diffusion Coeficient
            ic         function - Initial Condition
            bcL        funciton - Left Boundry Condition (Dirichlet)
            bcR        function - Right Boundry Condition (Neumann)
            =========  =============================================
        
    Ix : tuple (x0 xf)
        The interval (x0, xf) of distance over which to approximate the solution.
    It : tuple (t0 tf)
        The interval (t0, tf) of time over which to approximate the solution.
    M : int, optional
        Number of steps for space in the approximation. Default is 1e3.
    N : int, optional
        Number of steps for time in the approximation. Default is 1e3.
    
    Returns
    -------
    W : (M + 1, N + 1) double ndarray
        Matrix with the approximations of the PDE.
    X : (M + 1) double ndarray
        Times for x
    T : (N + 1) double ndarray
        Times for t
    """
    
    X = np.linspace(Ix[0],Ix[1], M + 1)
    h = (Ix[1]-Ix[0])/M
    
    T = np.linspace(It[0],It[1], N + 1)
    k = (It[1]-It[0])/N
    
    
    W = np.zeros((M + 1, N + 1))
    W[:,0] = eq['ic'](X)
    
    sigma = eq['D']*k/(h**2)
    
    #Matriz del lado izquierdo
    di = [[1]+[1+2*sigma]*(M-1)+[3/(2*h)], \
         [-sigma]*(M-1) + [-2/h], \
         [0] + [-sigma]*(M-1), \
         [0]*(M-2) + [1/(2*h)]]
    A = diags(di, [0,-1,1, -2], (M+1, M+1), format = 'csc')
    Ainv = inv(A)
    
    for i in range(N):
        b = np.zeros(M+1)
        b[0] = eq['bcL'](T[i+1])
        b[-1] = eq['bcR'](T[i+1])
        b[1:M] = W[1:M, i]
        
        W[:,i+1] = Ainv.dot(b)
        
    return W, X, T
