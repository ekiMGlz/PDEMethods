import numpy as np
from scipy.sparse.linalg import inv
from scipy.sparse import diags, bmat, eye

def explicitHeat(eq, Ix, It, M = 1e3, N = 1e3):
    """
    Explicit method for solving the heat equation:
        u_{t} = D * u_{xx}
    With an initial condition for x, a Dirichlet left boundry condition and a Neumann 
    right boundry condition for t, over an interval of space Ix = [a,b] and an interval
    of time It = [0,t]:
        u(x,0)     =    b(x)
        u(a,b)     =    l(t)
        u_{x}(b,t) =    r(t)
    The method uses forward finite differences over N points to approximate u_{t} and
    centered finite differences over M points to approximate u_{xx}. We also approximate
    the values at the right boundry through the Taylor series:
        u_{t}(x,t)  = (u(x,t+k) - u(x,t)) / k
        u_{xx}(x,t) = (u(x+h,t) - 2u(x,t) + u(x-h,t)) / (h^2)
        u_{x}(x,t)  = (u(x-2h,t) - 4u(x-h,t) + 3u(x,t)) / 2h
    This results in a linear system:
        Cw_{:,j+1} = Aw{:,j}+b
    Where:
        - A is a tridiagonal matrix with the coefficients from the approximations
        - C is the identity, except for the last row where it is the approximation for the right boundry
        - b is a zero vector with the left boundry condition on the first entry and the right boundry condition on the last.
        - w_{:,j} is the vector with the approximations for x at time j
    
    
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
    
    #Generate equally spaced intervals for X and T
    X = np.linspace(Ix[0],Ix[1], M + 1)
    h = (Ix[1]-Ix[0])/M
    
    T = np.linspace(It[0],It[1], N + 1)
    k = (It[1]-It[0])/N
    
    #Initialize the approximation matrix with zeros, and the initial conditions of x in the first column
    W = np.zeros((M + 1, N + 1))
    W[:,0] = eq['ic'](X)
    
    #Define sigma as Dk/h**2
    sigma = eq['D']*k/(h**2)
    
    
    #Create the matrix A as a tridiagonal sparse matrix
    d = [[0]+[1-2*sigma]*(M-1)+[0], \
         [sigma]*(M-1) + [0], \
         [0] + [sigma]*(M-1)]
    A = diags(d, [0,-1,1], (M+1, M+1))
    
    
    #Create Cinv, the inverse of the semi-identity matrix C.
    Cinv = np.eye(M+1)
    Cinv[M, M-2] = -1/3
    Cinv[M, M-1] = 4/3
    Cinv[M, M] = 2*h/3
    
    #Iterate to fill all the columns of the matrix with the appropriate solutions
    for i in range(N):
        #Define b as a 0 vector with b[0] = l(t) and b[-1] = r(t)
        b = np.zeros(M+1)
        b[0] = eq['bcL'](T[i+1])
        b[-1] = eq['bcR'](T[i+1])
        
        #Solve CW[:,i+1] = AW[:,i] + b
        b = b + A.dot(W[:,i])
        W[:,i+1] = Cinv.dot(b)
    
    return W, X, T

def implicitHeat(eq, Ix, It, M = 1e3, N = 1e3):
    """
    Implicit method for solving the heat equation:
        u_{t} = D * u_{xx}
    With an initial condition for x, a Dirichlet left boundry condition and a Neumann 
    right boundry condition for t, over an interval of space Ix = [a,b] and an interval
    of time It = [0,t]:
        u(x,0)     =    b(x)
        u(a,b)     =    l(t)
        u_{x}(b,t) =    r(t)
    The method uses backwards finite differences over N points to approximate u_{t} and
    centered finite differences over M points to approximate u_{xx}. We also approximate
    the values at the right boundry through the Taylor series:
        u_{t}(x,t)  = (u(x,t) - u(x,t-k)) / k
        u_{xx}(x,t) = (u(x+h,t) - 2u(x,t) + u(x-h,t)) / (h^2)
        u_{x}(x,t)  = (u(x-2h,t) - 4u(x-h,t) + 3u(x,t)) / 2h
    This results in a linear system:
        Aw_{:,j+1} = b
    Where:
        - A is a tridiagonal matrix with the coefficients from the approximations.
        - b is a vector with the left boundry condition on the first entry, the right boundry condition
            on the last, and the respective approcimations of x in every other entry.
    
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
    
    #Generate equally spaced intervals for X and T
    X = np.linspace(Ix[0],Ix[1], M + 1)
    h = (Ix[1]-Ix[0])/M
    
    T = np.linspace(It[0],It[1], N + 1)
    k = (It[1]-It[0])/N
    
    #Initialize the approximation matrix with zeros, and the initial conditions of x in the first column
    W = np.zeros((M + 1, N + 1))
    W[:,0] = eq['ic'](X)
    
    #Define sigma as Dk/h**2
    sigma = eq['D']*k/(h**2)
    
    #Create the semi-tridiagonal sparse matrix, then calculate the inverse in order to save time when solving the system
    di = [[1]+[1+2*sigma]*(M-1)+[3/(2*h)], \
         [-sigma]*(M-1) + [-2/h], \
         [0] + [-sigma]*(M-1), \
         [0]*(M-2) + [1/(2*h)]]
    A = diags(di, [0,-1,1, -2], (M+1, M+1), format = 'csc')
    Ainv = inv(A)
    
    for i in range(N):
        #Define b as a 0 vector with b[0] = l(t), b[-1] = r(t), and the previous approximations in every other entry
        b = np.zeros(M+1)
        b[0] = eq['bcL'](T[i+1])
        b[-1] = eq['bcR'](T[i+1])
        b[1:M] = W[1:M, i]
        
        #Solve AW[:,i+1] = b
        W[:,i+1] = Ainv.dot(b)
    
    return W, X, T
