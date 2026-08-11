"""
only part A done, will get onto part B later in the week, could you give it a quick check over make sure im not reading anything wrong?

also i saw you changed up Q2, if you want to do the same here go ahead, im just a bit lost with that so cant really help there
"""
import numpy as np
import matplotlib.pyplot as plt

def forwardSub(L, b):
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        s = b[i]
        for k in range(i):
            s -= L[i, k] * y[k]
        y[i] = s / L[i, i]
    return y

#changing the gauss seidel stuff from learn, unsure if the equation is right
def sor (A , b , x0 , w , max_iter =1000 , tol =1e-8 ): 
    n = x0.shape[0]
    x = x0.copy()
    results = []
    converged = False
    for k in range(max_iter):
        x_new = np.empty_like(x)
        for i in range(n):
                x_new[i] =(1-w) * x[i] + w * (b[i] - (A[i,:i].dot(x_new[:i])) - (A[i,i+1:].dot(x[i+1:])))/A[i,i]        
        results.append([k+1,*x_new])
        err = np.linalg.norm(x_new - x,np.inf)/np.linalg.norm(x_new,np.inf)
        if err < tol:
            converged = True
            break
        x = x_new.copy()
    return np.array(results), converged

#from learn page 
def gauss_seidel(A, b, x0, max_iter=100, tol=1e-3): # adding in to compare to SOR (a. iii)

    P = np.tril(A)
    solver = forwardSub

    Q = P - A
    converged = False
    x = x0.copy()
    results = []
    for k in range(max_iter):
        rhs = Q @ x + b
        x_new = solver(P,rhs)
        results.append([k+1,*x_new])
        err = np.linalg.norm(x_new - x,np.inf)/np.linalg.norm(x_new,np.inf)
        if err < tol:
            converged = True
            break
        x = x_new.copy()
    return np.array(results), k 

def good_matrix (n , d ):
    rng = np . random . default_rng ( seed = 211 )
    B = rng . uniform ( size =(n , n ))
    return B . T @ B + d*n*np .eye( n )

if __name__ == "__main__":
    """
    a. i)
    """
   
    n= 20
    d = .5
    A = good_matrix(n, d)
    b = np.ones(n)
    x0 = b.copy()

    w = np.linspace(0.01, 1.99, 300)
    itters = []
    ws = []
    for i in w:
        x, f = sor(A, b, x0, i, max_iter=1000)
        itters.append(x[-1, 0])
        ws.append(i)
    """
    a. ii)
    """
    plt.title("itterations vs w value")
    plt.ylabel('w')
    plt.xlabel('itterations')
    plt.plot(itters, ws)
    plt.show()

"""
a. iii)

when plotting both graphs together, SOR does not seem to be more efficeint than gauss seidel, 
SOR is best for this matrix when w ~= .5-.6, but never better than gauss seidel in terms of itterations
"""
_, k = gauss_seidel(A, b, x0)

plt.plot(ws, itters, label='SOR')
plt.axhline(y=k, color='r', linestyle='--', label='Gauss-Seidel') #axhline to get a single line to compare against SOR curve
plt.xlabel('w')
plt.ylabel('iterations')
plt.title('SOR iterations vs w, compared to Gauss-Seidel')
plt.legend()
plt.show()
