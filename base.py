# Example scripts for plotting Argand diagram for the base case
# Windows 10 64-bit 22H2 / Python 3.9.5 / NumPy 1.26.2 / SciPy 1.11.3 / Matplotlib 3.4.3
# Author: Shaoguang Wang
# Date: June 5, 2025
# License: Apache 2.0
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

#Transcendental equation for a cantilever beam
beamfun = lambda x: 1 + np.cos(x) * np.cosh(x)

#Eigenfunctions and derivatives
def ModeShape(x, r):
    const = (np.cos(r)+np.cosh(r)) / (np.sin(r)+np.sinh(r))
    w = np.cosh(r*x)-np.cos(r*x) + const * (np.sin(r*x)-np.sinh(r*x))
    return w
def ModeShape_prime(x, r):
    const = (np.cos(r)+np.cosh(r)) / (np.sin(r)+np.sinh(r))
    w = np.sinh(r*x)+np.sin(r*x) + const * (np.cos(r*x)-np.cosh(r*x))
    return w*r
def ModeShape_2prime(x, r):
    const = (np.cos(r)+np.cosh(r)) / (np.sin(r)+np.sinh(r))
    w = np.cosh(r*x)+np.cos(r*x) - const * (np.sin(r*x)+np.sinh(r*x))
    return w*r**2

#Define coef matrix for solving the integral eqn
g = lambda x, ksee, d: (x-ksee) / ((x-ksee)**2 + d**2)
h = lambda x, t: 1 / (2*x - t-1)
def CoMatrix(n,x,t,d):
    A = np.zeros((2*n,2*n))
    for i in range(n-1):
        for j in range(n):
            A[i, j] = h(x[i], t[j])
            A[i+n-1, j+n] = h(x[i], t[j])
            A[i, j+n] = g(x[i], (t[j]+1)/2, d) /2
            A[i+n-1, j] = g(x[i], (t[j]+1)/2, d) /2
    A[-1, n:] = np.ones(n)
    A[-2, :n] = np.ones(n)
    return A/(2*n)

#Force vector in the integral eqn
def FMatrix(n,x,r,m,Q):
    fm = np.zeros(2*n)
    fg = np.zeros(2*n)
    fk = np.zeros(2*n)
    if m < Q:
        fm[:n-1] = ModeShape(x, r)
        fg[:n-1] = ModeShape_prime(x, r)
        fk[:n-1] = ModeShape_2prime(x, r)
    else:
        fm[n-1:2*n-2] = ModeShape(x, r)
        fg[n-1:2*n-2] = ModeShape_prime(x, r)
        fk[n-1:2*n-2] = ModeShape_2prime(x, r)
    return fm, fg, fk

#Pressure gradient, including 3 components
def SolvePhi(n,x,r,m,Q,A):
    bM, bG, bK = FMatrix(n,x,r,m,Q)
    Ainv = np.linalg.inv(A)
    PhiM = Ainv @ bM
    PhiG = Ainv @ bG
    PhiK = Ainv @ bK
    return PhiM, PhiG, PhiK

#Fully recovered pressure across flag 1 and 2
def pressure(n, pgrad,t):
    Phi1 = pgrad[:n]
    Phi2 = pgrad[n:]
    co1 = np.polynomial.chebyshev.chebfit(t, Phi1, 8)
    co2 = np.polynomial.chebyshev.chebfit(t, Phi2, 8)
    changevar = lambda x,t: x-1 + x*t
    ChebPhi1 = lambda x,t: np.polynomial.chebyshev.chebval(changevar(x,t),co1)
    ChebPhi2 = lambda x,t: np.polynomial.chebyshev.chebval(changevar(x,t),co2)
    q1 = lambda x,s: x * ChebPhi1(x,s) * np.sqrt(1-s**2) / np.sqrt(1-changevar(x,s)**2)
    q2 = lambda x,s: x * ChebPhi2(x,s) * np.sqrt(1-s**2) / np.sqrt(1-changevar(x,s)**2)
    x = (1+t)/2
    p1, p2 = [],[]
    for j in range(n):
        I1, I2 = 0, 0
        for i in range(n):
            I1 += q1(x[j], t[i])
            I2 += q2(x[j], t[i])
        p1.append(np.pi/n*I1/2)
        p2.append(np.pi/n*I2/2)
    return p1, p2

# Solve eigenvalues and eigenvectors
def solve_eig(n, Mass,U, M, K, FM, FG, FK):
    MatrixM = M + Mass * FM
    MatrixG = -2 * Mass * FG * U
    MatrixK = K + Mass * FK*U**2
    for i in range(Q):
        for j in range(Q):
            Cij = ModeShape(1,roots[j]) * ModeShape(1,roots[i])
            Kij = ModeShape_prime(1,roots[j]) * ModeShape(1,roots[i])
            MatrixG[i,j] += np.sqrt(Mass) * U * Cij
            MatrixK[i,j] -= U**2 * Kij
            MatrixG[i+Q,j+Q] += np.sqrt(Mass) * U * Cij
            MatrixK[i+Q,j+Q] -= U**2 * Kij
    Minv = np.linalg.inv(MatrixM)
    MatA = -Minv @ MatrixK
    MatB = -Minv @ MatrixG
    Gamma = np.block([[np.zeros((n,n)), np.identity(n)], [MatA, MatB]])
    return np.linalg.eig(Gamma)

# Main scripts start from here
# To compute the Cauchy type singular integrals    
N = 200 # Number of Gauss-Chebyshev nodes discretized on t
i = np.arange(1,N+1)
j = np.arange(1,N)
t_i = np.cos((i-0.5)*np.pi/N) # Discretizing nodes
x_j = (1+np.cos(j*np.pi/N))/2 # Abscissa points
d = 2/3 # Gap
A = CoMatrix(N,x_j,t_i,d)

# Roots of transcendental eqn
roots = []
Q = 6 # number of modes for the flag approximated solution
for i in np.arange(0.5*np.pi, Q*np.pi, np.pi):
    rt = fsolve(beamfun, i).item()
    roots.append(rt)
roots *= 2

# Matrices in the eigenvalue problem
M = np.identity(2*Q)
K = np.identity(2*Q)
F_M = np.zeros((2*Q,2*Q))
F_G = np.zeros((2*Q,2*Q))
F_K = np.zeros((2*Q,2*Q))
for j in range(2*Q):
    PhiM, PhiG, PhiK = SolvePhi(N, x_j, roots[j], j, Q, A)
    p1M, p2M = pressure(N, PhiM, t_i)
    p1G, p2G = pressure(N, PhiG, t_i)
    p1K, p2K = pressure(N, PhiK, t_i)
    K[j, j] = roots[j]**4
    for i in range(Q):
        for k in range(N):
            F_M[i, j] += p1M[k] * ModeShape(0.5+0.5*t_i[k], roots[i]) * np.sqrt(1-t_i[k]**2)
            F_G[i, j] += p1G[k] * ModeShape(0.5+0.5*t_i[k], roots[i]) * np.sqrt(1-t_i[k]**2)
            F_K[i, j] += p1K[k] * ModeShape(0.5+0.5*t_i[k], roots[i]) * np.sqrt(1-t_i[k]**2)
            F_M[i+Q, j] += p2M[k] * ModeShape(0.5+0.5*t_i[k], roots[i]) * np.sqrt(1-t_i[k]**2)
            F_G[i+Q, j] += p2G[k] * ModeShape(0.5+0.5*t_i[k], roots[i]) * np.sqrt(1-t_i[k]**2)
            F_K[i+Q, j] += p2K[k] * ModeShape(0.5+0.5*t_i[k], roots[i]) * np.sqrt(1-t_i[k]**2)
F_M *= np.pi/(2*N)
F_G *= np.pi/(2*N)
F_K *= np.pi/(2*N)

# Example for the base case, return Argand diagram
mu = 0.3 # mass ratio
f1 = 3.21 # track mode 1
f1im = 0.005
re, im = [], []
Uarray = np.arange(0, 1.6, 0.01)
for U in Uarray:
    eigval, eigvec = solve_eig(2*Q, mu, U, M, K, F_M, F_G, F_K)
    frequency = -1j * eigval
    idx = np.argmin(abs(frequency.real - f1) + abs(frequency.imag-f1im))
    re.append(frequency[idx].real)
    im.append(frequency[idx].imag)
    f1 = frequency[idx].real
    f1im = frequency[idx].imag
plt.figure()
plt.plot(re, im, '-o', color='k')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.savefig('CHANGE TO YOUR PATH/test.pdf')
plt.close()