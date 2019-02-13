import numpy as np
import scipy as sp
from scipy import integrate
import qutip as qt
from qutip import destroy, tensor, qeye, spost, spre, sprepost, basis
import time
from utils import (J_minimal, beta_f, J_multipolar, lin_construct, 
                    exciton_states, rate_up, rate_down, Occupation)
import sympy
from numpy import pi, sqrt

def coth(x):
    return float(sympy.coth(x))

def cauchyIntegrands(omega, beta, J, Gamma, w0, ver, alpha=0.):
    # J_overdamped(omega, alpha, wc)
    # Function which will be called within another function where J, beta and
    # the eta are defined locally
    F = 0
    if ver == 1:
        F = J(omega, Gamma, w0, alpha=alpha)*(coth(beta*omega/2.)+1)
    elif ver == -1:
        F = J(omega, Gamma, w0, alpha=alpha)*(coth(beta*omega/2.)-1)
    elif ver == 0:
        F = J(omega, Gamma, w0, alpha=alpha)
    return F

def int_conv(f, a, inc, omega, tol=1E-4):
        x = inc
        I = 0.
        while abs(f(x))>tol:
            #print inc, x, f(x), a, omega
            I += integrate.quad(f, a, x, weight='cauchy', wvar=omega)[0]
            a+=inc
            x+=inc
            #time.sleep(0.1)
        #print "Integral converged to {} with step size of {}".format(I, inc)
        return I # Converged integral

def integral_converge(f, a, omega, tol=1e-7):
    for inc in [300., 200., 100., 50., 25., 10, 5., 1, 0.5]:
        inc += np.random.random()
        try:
            return int_conv(f, a, inc, omega, tol=tol) 
        except:
            pass
    raise ValueError("Integrals couldn't converge")
                
    

def DecayRate(omega, beta, J, Gamma, w0, imag_part=True, alpha=0., tol=1e-4):
    G = 0
    # Here I define the functions which "dress" the integrands so they
    # have only 1 free parameter for Quad.
    F_p = (lambda x: (cauchyIntegrands(x, beta, J, Gamma, w0, 1 , alpha=alpha)))
    F_m = (lambda x: (cauchyIntegrands(x, beta, J, Gamma, w0, -1, alpha=alpha)))
    F_0 = (lambda x: (cauchyIntegrands(x, beta, J, Gamma, w0, 0,  alpha=alpha)))
    w='cauchy'

    if omega>0.:
        # These bits do the Cauchy integrals too
        G = (np.pi/2)*(coth(beta*omega/2.)-1)*J(omega, Gamma, w0, alpha=alpha)
        if imag_part:
            G += (1j/2.)*(integral_converge(F_m, 0,omega, tol=tol))
            G -= (1j/2.)*(integral_converge(F_p, 0,-omega, tol=tol))

        #print integrate.quad(F_m, 0, n, weight='cauchy', wvar=omega), integrate.quad(F_p, 0, n, weight='cauchy', wvar=-omega)
    elif omega==0.:
        G = (pi*alpha*Gamma)/(beta*(w0**2))
        if imag_part:
            G += -(1j)*integral_converge(F_0, -1e-12,0., tol=tol)
        #print (integrate.quad(F_0, -1e-12, 20, weight='cauchy', wvar=0)[0])
    elif omega<0.:
        G = (np.pi/2)*(coth(beta*abs(omega)/2.)+1)*J(abs(omega),Gamma, w0, alpha=alpha)
        if imag_part:
            G += (1j/2.)*integral_converge(F_m, 0, -abs(omega), tol=tol)
            G -= (1j/2.)*integral_converge(F_p, 0, abs(omega), tol=tol)
        #print integrate.quad(F_m, 0, n, weight='cauchy', wvar=-abs(omega)), integrate.quad(F_p, 0, n, weight='cauchy', wvar=abs(omega))
    return G

def _J_underdamped(omega, Gamma, omega_0, alpha=0.):
    return alpha*Gamma*(omega_0**2)*omega/(((omega_0**2)-(omega**2))**2+(Gamma*omega)**2)

def L_wc_analytic(detuning=0., Rabi=0, alpha=0., w0=0., Gamma=0., T=0., tol=1e-7):
    energies, states = exciton_states(detuning, Rabi)
    dark_proj = states[0]*states[0].dag()
    bright_proj = states[1]*states[1].dag()
    ct_p = states[1]*states[0].dag()
    ct_m = states[0]*states[1].dag()
    cross_term = (ct_p + ct_m)
    epsilon = -detuning
    V = Rabi/2
    
    eta = sqrt(epsilon**2 + 4*V**2)

    # Bath 1 (only one bath)
    G = (lambda x: (DecayRate(x, beta_f(T), _J_underdamped, 
                        Gamma, w0, imag_part=True, 
                        alpha=alpha, tol=tol)))
    G_0 = G(0.)
    G_p = G(eta)
    G_m = G(-eta)

    site_1 = (0.5/eta)*((eta+epsilon)*bright_proj + (eta-epsilon)*dark_proj + 2*V*cross_term)

    Z_1 = (0.5/eta)*(G_0*((eta+epsilon)*bright_proj + (eta-epsilon)*dark_proj) + 2*V*(ct_p*G_p + ct_m*G_m))

    L =  - qt.spre(site_1*Z_1) + qt.sprepost(Z_1, site_1)
    L += -qt.spost(Z_1.dag()*site_1) + qt.sprepost(site_1, Z_1.dag())

    return L