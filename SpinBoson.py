# -*- coding: utf-8 -*-
"""
Weak-coupling spin-boson model solution
written in Python 2.7
"""
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
#import ctypes

def J_superohm(omega, omega_c=2.2, alpha=5.):
    #return alpha*(omega**3)*np.exp(-(omega/omega_c)**2)
    return alpha*(omega**3)*np.exp(-(omega/omega_c)**2)
def J_ohm(omega, omega_c=2.2, alpha=0.025):
    return alpha*(omega)*np.exp(-(omega/omega_c))

def J_overdamped(omega, omega_c=2.2, alpha=5.):
    return alpha*omega*omega_c/(omega**2 + omega_c**2)


def coth(x):
    return (np.exp(2*x)+1)/(np.exp(2*x)-1)

def cauchyIntegrands(omega, beta, J, ver):
    # Function which will be called within another function where J, beta and the eta are defined locally
    F = 0
    if ver == 1:
        F = J(omega)*(coth(beta*omega/2.)+1)
    elif ver == -1:
        F = J(omega)*(coth(beta*omega/2.)-1)
    elif ver == 0:
        F = J(omega)
    return F

def Gamma(omega, beta, J):
    G = 0
    # Here I define the functions which "dress" the integrands so they have only 1 free parameter for Quad.
    F_p = (lambda x: (cauchyIntegrands(x, beta, J, 1)))
    F_m = (lambda x: (cauchyIntegrands(x, beta, J, -1)))
    F_0 = (lambda x: (cauchyIntegrands(x, beta, J, 0)))
    n = 21
    print "Cauchy int. convergence checks: ", F_0(4*n), F_m(4*n), F_p(4*n)
    w='cauchy'
    if omega>0.:
        # These bits do the Cauchy integrals too
        G = (np.pi/2)*(coth(beta*omega/2.)-1)*J(omega)

        G += (1j/2.)*(integrate.quad(F_m, 0, n, weight=w, wvar=omega)[0]
                    +integrate.quad(F_m, n, 2*n, weight=w, wvar=omega)[0]
                    +integrate.quad(F_m, 2*n, 3*n, weight=w, wvar=omega)[0]
                    +integrate.quad(F_m, 3*n, 4*n, weight=w, wvar=omega)[0]
                    - integrate.quad(F_p, 0, n, weight=w, wvar=-omega)[0]
                    -integrate.quad(F_p, n, 2*n, weight=w, wvar=-omega)[0]
                    -integrate.quad(F_p, 2*n, 3*n, weight=w, wvar=-omega)[0]
                    -integrate.quad(F_p, 3*n, 4*n, weight=w, wvar=-omega)[0])
        #print integrate.quad(F_m, 0, n, weight='cauchy', wvar=omega), integrate.quad(F_p, 0, n, weight='cauchy', wvar=-omega)
    elif omega==0.:
        # The limit as omega tends to zero is zero for superohmic case?
        G = -(1j)*(integrate.quad(F_0, -1e-12, n, weight=w, wvar=0)[0]
                    +integrate.quad(F_0, n, 2*n, weight=w, wvar=0)[0]
                    +integrate.quad(F_0, 2*n, 3*n, weight=w, wvar=0)[0]
                    +integrate.quad(F_0, 3*n, 4*n, weight=w, wvar=0)[0])
        #print (integrate.quad(F_0, -1e-12, 20, weight='cauchy', wvar=0)[0])
    elif omega<0.:
        G = (np.pi/2)*(coth(beta*abs(omega)/2.)+1)*J(abs(omega))
        G += (1j/2.)*(integrate.quad(F_m, 0, n, weight=w, wvar=-abs(omega))[0]
                    +integrate.quad(F_m, n, 2*n, weight=w, wvar=-abs(omega))[0]
                    +integrate.quad(F_m, 2*n, 3*n, weight=w, wvar=-abs(omega))[0]
                    +integrate.quad(F_m, 3*n, 4*n, weight=w, wvar=-abs(omega))[0]
                    - integrate.quad(F_p, 0, n, weight=w, wvar=abs(omega))[0]
                    -integrate.quad(F_p, n, 2*n, weight=w, wvar=abs(omega))[0]
                    -integrate.quad(F_p, 2*n, 3*n, weight=w, wvar=abs(omega))[0]
                    -integrate.quad(F_p, 3*n, 4*n, weight=w, wvar=abs(omega))[0])
        #print integrate.quad(F_m, 0, n, weight='cauchy', wvar=-abs(omega)), integrate.quad(F_p, 0, n, weight='cauchy', wvar=abs(omega))
    return G


def liouvillian(epsilon, delta, J, T):
    L = 0 # Initialise liouvilliian
    Z = 0 # initialise operator Z
    #beta = 1 /(T* 0.695)
    beta = 7.683/T
    eta = np.sqrt(epsilon**2 + delta**2)
    # Here I define the eigenstates of the H_s
    H = qt.Qobj([[-epsilon/2., delta/2],[delta/2, epsilon/2.]])
    eVecs = H.eigenstates()[1]
    psi_p = (1/np.sqrt(2*eta))*(np.sqrt(eta-epsilon)*qt.basis(2,0) + np.sqrt(eta+epsilon)*qt.basis(2,1))
    psi_m = (-1/np.sqrt(2*eta))*(np.sqrt(eta+epsilon)*qt.basis(2,0) - np.sqrt(eta-epsilon)*qt.basis(2,1))

    #print H.eigenstates()
    # Jake's eigenvectors
    #psi_p = (1/np.sqrt(2*eta))*(np.sqrt(eta+epsilon)*qt.basis(2,0) - np.sqrt(eta-epsilon)*qt.basis(2,1))
    #psi_m = (1/np.sqrt(2*eta))*(np.sqrt(eta-epsilon)*qt.basis(2,0) + np.sqrt(eta+epsilon)*qt.basis(2,1))

    sigma_z = (1/eta)*(epsilon*(psi_p*psi_p.dag()-psi_m*psi_m.dag()) + delta*(psi_p*psi_m.dag() + psi_m*psi_p.dag()))

    Z = (1/eta)*(epsilon*(psi_p*psi_p.dag()-psi_m*psi_m.dag() )*Gamma(0, beta, J) + delta*(Gamma(eta, beta, J)*psi_p*psi_m.dag() + Gamma(-eta,beta, J)*psi_m*psi_p.dag()))

    L +=  qt.spre(sigma_z*Z) - qt.sprepost(Z, sigma_z)
    L += qt.spost(Z.dag()*sigma_z) - qt.sprepost(sigma_z, Z.dag())
    return -L

"""
plt.figure()
omega= np.linspace(0,50, 1000)
plt.plot(omega,J_superohm(omega))
plt.title("Spectral density")
"""
"""
epsilon = 1.
delta = 2*np.pi #*10**(-12)
T = 10.

L = liouvillian(epsilon, delta, J_overdamped, T)

H = qt.Qobj([[-epsilon/2., delta/2.],[delta/2., epsilon/2.]])


rho = qt.fock_dm(2,1)
timelist = np.linspace(0.,16., 10000)
expect_list = [qt.fock_dm(2,0), qt.fock_dm(2,1)]
DATA = qt.mesolve(H, rho, timelist, [L], expect_list)
DATA_A = (open("Data/pop5d2pi.dat").read()).split('\n')
pop_A, time_A = [], []
for item in DATA_A:
    li = item.split('\t')
    time_A.append(float(li[0]))
    pop_A.append(float(li[1]))

plt.figure()
plt.plot(timelist, DATA.expect[1], label='H')
plt.plot(time_A, pop_A, label="Ahsan's data")
plt.legend()

"""
"""
fo = open("WCSB_0p025.dat", "w")
fo.write("% Parameters: \n alpha=0.025, omega_c=2.2, T=10, delta = pi, epsilon=1% \n")
for item in DATA.expect[1]:
    fo.write(item)
"""
#fo.write(DATA.expect[1])
#fo.write("\n")
#fo.write(timelist)

#DATA_J = qt.mesolve(H, rho, timelist, [L_Jake], expect_list)
#plt.plot(timelist, DATA.expect[0])



#plt.plot(timelist, DATA_J.expect[1], label='J')
#plt.ylim(0,1)
#print liuo(100, 10, J, T=10.).eigenenergies()
#plt.legend()
#plt.hold()




#plt.show()
