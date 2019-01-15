"""
In this script we have four methods.
1) Ham_RC builds the RC-frame Hamiltonian and system operators for both bath interactions.
    It takes in the system splitting, the RC frequency, system-RC coupling and the Hilbert space dimension.

2) RCME_operators builds the collapse operators for the RCME. It takes as input the
    Hamiltonian, system operator for phonon interaction, RC-residual coupling strength and beta (inv. T).

3) liouvillian_build... builds the RCME Liouvillian. Taking RC-residual coupling, RC-freq. and Temperature (for beta).
    It also takes a default parameter time_units which is set 'cm', giving frequencies in inverse cm.
    Can be set to 'ps' for inv. picoseconds.

4) RC_function_UD dresses up the Liouvillian with all the mapped RC frame parameters.
    It calculates these in accordance with Jake's initial paper.
"""
import numpy as np
import scipy as sp
from qutip import destroy, tensor, qeye, spre, spost, sprepost, Qobj
from utils import Coth, beta_f


#import pdb; pdb.set_trace()

def Ham_RC(sigma, eps, Omega, kappa, N, rotating=False):
    """
    Input: System splitting, RC freq., system-RC coupling and Hilbert space dimension
    Output: Hamiltonian, sigma_- and sigma_z in the vibronic Hilbert space
    """

    a = destroy(N)
    shift = (kappa**2)/Omega
    I_sys = Qobj(qeye(sigma.shape[0]),dims=sigma.dims)
    sys_energy = (eps+shift)
    if rotating:
        sys_energy=0.
    H_S = sys_energy*tensor(sigma.dag()*sigma, qeye(N)) + kappa*tensor(sigma.dag()*sigma, (a + a.dag())) + tensor(I_sys,Omega*a.dag()*a)
    A_em = tensor(sigma, qeye(N))
    A_nrwa = tensor(sigma+sigma.dag(), qeye(N))
    A_ph = tensor(I_sys, (a + a.dag()))
    return H_S, A_em, A_nrwa, A_ph

def Ham_RC_gen(H_sub, sigma, Omega, kappa, N, rotating=False, shift = True, shift_op=None, w_laser=0.):
    """
    will only work for spin-boson like models
    Input: System Hamiltonian, RC freq., system-RC coupling and Hilbert space dimension
    Output: Hamiltonian, sigma_- and sigma_z in the vibronic Hilbert space
    """
    a = destroy(N)
    energy_shift = (kappa**2)/Omega 
    I_sys = Qobj(qeye(H_sub.shape[0]),dims=sigma.dims)
    if shift and shift_op is not None:
        H_sub += shift_op*energy_shift
    if rotating:
        # Hopefully removes energy scale. Shift operator should be the same as
        # the site energy-scale operator.
        H_sub -= shift_op*H_sub*(shift_op.dag())
    H_S = tensor(H_sub, qeye(N)) + kappa*tensor(shift_op, (a + a.dag()))
    H_S += tensor(I_sys, Omega*a.dag()*a)
    A_em = tensor(sigma, qeye(N))
    A_nrwa = tensor(sigma+sigma.dag(), qeye(N))
    A_ph = tensor(I_sys, (a + a.dag()))
    return H_S, A_em, A_nrwa, A_ph


def RCME_operators(H_0, A, gamma, beta):
    # This function will be passed a TLS-RC hamiltonian, RC operator, spectral density and beta
    # outputs all of the operators needed for the RCME (underdamped)
    dim_ham = H_0.shape[0]
    Chi = 0 # Initiate the operators
    Xi = 0
    eVals, eVecs = H_0.eigenstates()
    ground_list = []
    excited_list = []
    for i in range(len(eVals)):
        is_ground = sum(eVecs[i])[0][0].real == 1.
        if is_ground:
            ground_list.append(i)
        else:
            excited_list.append(i)

    #print H_0
    #ti = time.time()
    for j in range(dim_ham):
        for k in range(dim_ham):
            e_jk = eVals[j] - eVals[k] # eigenvalue difference
            A_jk = A.matrix_element(eVecs[j].dag(), eVecs[k])
            outer_eigen = eVecs[j] * (eVecs[k].dag())
            if sp.absolute(A_jk) > 0:
                if sp.absolute(e_jk) > 0:
                    #print e_jk
                    # If e_jk is zero, coth diverges but J goes to zero so limit taken seperately
                    """
                    if (np.pi*gamma*A_jk/beta) >0:
                        print j, k
                        print j in ground_list, k in ground_list
                        print e_jk"""
                    Chi += 0.5*np.pi*e_jk*gamma * Coth(e_jk * beta / 2)*A_jk*outer_eigen # e_jk*gamma is the spectral density
                    Xi += 0.5*np.pi*e_jk*gamma * A_jk * outer_eigen
                else:
                    """
                    if (np.pi*gamma*A_jk/beta) >0:
                        print j, k
                        print j in ground_list, k in ground_list
                        print e_jk"""

                    Chi += (np.pi*gamma*A_jk/beta)*outer_eigen # Just return coefficients which are left over
                    #Xi += 0 #since J_RC goes to zero

    return H_0, A, Chi, Xi

def liouvillian_build(H_0, A, gamma, wRC, T_C):
    # Now this function has to construct the liouvillian so that it can be passed to mesolve
    H_0, A, Chi, Xi = RCME_operators(H_0, A, gamma, beta_f(T_C))
    L = 0
    L-=spre(A*Chi)
    L+=sprepost(A, Chi)
    L+=sprepost(Chi, A)
    L-=spost(Chi*A)

    L+=spre(A*Xi)
    L+=sprepost(A, Xi)
    L-=sprepost(Xi, A)
    L-=spost(Xi*A)

    return L, Chi+Xi

def liouvillian_build_new(H_0, A, gamma, wRC, T_C):
    # Now this function has to construct the liouvillian so that it can be passed to mesolve
    H_0, A, Chi, Xi = RCME_operators(H_0, A, gamma, beta_f(T_C))
    Z = Chi+Xi
    Z_dag = Z.dag()
    L=0
    #L+=spre(A*Z_dag)
    #L-=sprepost(A, Z)
    #L-=sprepost(Z_dag, A)
    #L+=spost(Z_dag*A)

    L-=spre(A*Z_dag)
    L+=sprepost(A, Z)
    L+=sprepost(Z_dag, A)
    L-=spost(Z*A)

    print("new L built")
    return L, Z

def RC_function_UD(sigma, eps, T_ph, Gamma, wRC, alpha_ph, N, silent=False,
                                            residual_off=False, rotating=False,
                                            new=False):
    # we define all of the RC parameters by the underdamped spectral density
    gamma = Gamma / (2. * np.pi * wRC)  # coupling between RC and residual bath
    if residual_off:
        gamma=0
    kappa= np.sqrt(np.pi * alpha_ph * wRC / 2.)  # coupling strength between the TLS and RC

    if not silent:
        print "w_RC={} | TLS splitting = {} | RC-res. coupling={:0.2f} | TLS-RC coupling={:0.2f} | Gamma_RC={:0.2f} | alpha_ph={:0.2f} | N={} |".format(wRC, eps, gamma,  kappa, Gamma, alpha_ph, N)
    H, A_em, A_nrwa, A_ph = Ham_RC(sigma, eps, wRC, kappa, N, rotating=rotating)
    if new:
        L_RC, Z =  liouvillian_build_new(H, A_ph, gamma, wRC, T_ph)
    else:
        L_RC, Z =  liouvillian_build(H, A_ph, gamma, wRC, T_ph)
    return L_RC, H, A_em, A_nrwa, Z, wRC, kappa, Gamma


def RC_function_gen(H_sub, sigma, T_ph, Gamma, wRC, alpha_ph, N, silent=False,
                                            residual_off=False, rotating=False,
                                            shift_op = None, shift=True, new=False, w_laser=0.):
    
    # we define all of the RC parameters by the underdamped spectral density
    gamma = Gamma / (2. * np.pi * wRC)  # coupling between RC and residual bath
    if residual_off:
        gamma=0
    kappa= np.sqrt(np.pi * alpha_ph * wRC / 2.)  # coupling strength between the TLS and RC

    if not silent:
        print "w_RC={} | RC-res. coupling={:0.2f} | TLS-RC coupling={:0.2f} | Gamma_RC={:0.2f} | alpha_ph={:0.2f} | N={} |".format(wRC, gamma,  kappa, Gamma, alpha_ph, N)
    if shift_op is None:
        shift_op = sigma.dag()*sigma
    H, A_em, A_nrwa, A_ph = Ham_RC_gen(H_sub, sigma, wRC, kappa, N,
                                        rotating=rotating,
                                        shift_op=shift_op, shift=shift, 
                                        w_laser=w_laser)
    if not new:
        L_RC, Z =  liouvillian_build(H, A_ph, gamma, wRC, T_ph)
    else:
        L_RC, Z =  liouvillian_build_new(H, A_ph, gamma, wRC, T_ph)
    return L_RC, H, A_em, A_nrwa, Z, wRC, kappa, gamma


#### WEAK COUPLING CODE

# -*- coding: utf-8 -*-
"""
Weak-coupling spin-boson model solution
written in Python 2.7
"""
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from utils import *
import sympy
from qutip import basis
import time
#import ctypes



def cauchyIntegrands(omega, beta, J, alpha, Gamma, omega_0, ver):
    # J_overdamped(omega, alpha, wc)
    # Function which will be called within another function where J, beta and
    # the eta are defined locally
    F = 0
    if ver == 1:
        F = J(omega, alpha, Gamma, omega_0)*(coth(beta*omega/2.)+1)
    elif ver == -1:
        F = J(omega, alpha, Gamma, omega_0)*(coth(beta*omega/2.)-1)
    elif ver == 0:
        F = J(omega, alpha, Gamma, omega_0)
    return F

def integral_converge(f, a, omega):
    x = 30
    I = 0
    while abs(f(x))>0.001:
        #print a, x
        I += integrate.quad(f, a, x, weight='cauchy', wvar=omega)[0]
        a+=30
        x+=30
    return I # Converged integral

def Decay(omega, beta, J, alpha, Gamma, omega_0, imag_part=True):
    G = 0
    # Here I define the functions which "dress" the integrands so they
    # have only 1 free parameter for Quad.
    F_p = (lambda x: (cauchyIntegrands(x, beta, J, alpha, Gamma, omega_0, 1)))
    F_m = (lambda x: (cauchyIntegrands(x, beta, J, alpha, Gamma, omega_0, -1)))
    F_0 = (lambda x: (cauchyIntegrands(x, beta, J, alpha, Gamma, omega_0, 0)))
    w='cauchy'
    if omega>0.:
        # These bits do the Cauchy integrals too
        G = (np.pi/2)*(coth(beta*omega/2.)-1)*J(omega, alpha, Gamma, omega_0)
        if imag_part:
            G += (1j/2.)*(integral_converge(F_m, 0,omega))
            G -= (1j/2.)*(integral_converge(F_p, 0,-omega))

        #print integrate.quad(F_m, 0, n, weight='cauchy', wvar=omega), integrate.quad(F_p, 0, n, weight='cauchy', wvar=-omega)
    elif omega==0.:
        G = (np.pi/2)*(2*alpha/beta)
        # The limit as omega tends to zero is zero for superohmic case?
        if imag_part:
            G += -(1j)*integral_converge(F_0, -1e-12,0)
        #print (integrate.quad(F_0, -1e-12, 20, weight='cauchy', wvar=0)[0])
    elif omega<0.:
        G = (np.pi/2)*(coth(beta*abs(omega)/2.)+1)*J(abs(omega),alpha, Gamma, omega_0)
        if imag_part:
            G += (1j/2.)*integral_converge(F_m, 0,-abs(omega))
            G -= (1j/2.)*integral_converge(F_p, 0,abs(omega))
        #print integrate.quad(F_m, 0, n, weight='cauchy', wvar=-abs(omega)), integrate.quad(F_p, 0, n, weight='cauchy', wvar=abs(omega))
    return G

def commutate(A, A_i, anti = False):
    if anti:
        return qt.spre(A*A_i) - qt.sprepost(A_i,A) + qt.sprepost(A, A_i.dag()) - qt.spre(A_i.dag()*A)
    else:
        return qt.spre(A*A_i) - qt.sprepost(A_i,A) - qt.sprepost(A, A_i.dag()) + qt.spre(A_i.dag()*A)


def auto_L(PARAMS, A, T, alpha):
    eig = zip(*check.exciton_states(PARAMS))
    L = 0
    beta = beta_f(T)
    for eig_i in eig:
        for eig_j in eig:
            omega = eig_i[0]-eig_j[0]
            A_ij = eig_i[1]*eig_j[1].dag()*A.matrix_element(eig_i[1].dag(), eig_j[1])
            L += Gamma(omega, beta, J_underdamped, alpha, PARAMS['wc'], imag_part=False) * commutate(A, A_ij)
            # Imaginary part
            G = Gamma(omega, beta, J_underdamped, alpha, PARAMS['wc'], imag_part=True)
            print G
            L += G.imag * commutate(A, A_ij, anti=True)
    return -0.5*L


def exciton_states(PARS):
    w_1, w_2, V, bias = PARS['w_1'], PARS['w_2'],PARS['V'], PARS['bias']
    v_p, v_m = 0, 0
    eta = np.sqrt(4*(V**2)+bias**2)
    lam_p = w_2+(bias+eta)*0.5
    lam_m = w_2+(bias-eta)*0.5
    v_m = np.array([ -(w_1-lam_p)/V, -1])
    #v_p/= /(1+(V/(w_2-lam_m))**2)
    v_m/= np.sqrt(np.dot(v_m, v_m))
    v_p = np.array([V/(w_2-lam_m),1.])

    v_p /= np.sqrt(np.dot(v_p, v_p))
    #print  np.dot(v_p, v_m) < 1E-15
    return [lam_m, lam_p], [qt.Qobj(v_m), qt.Qobj(v_p)]

def L_weak_phonon_SES(PARAMS, silent=False):
    ti = time.time()
    w_1 = PARAMS['w_1']
    w_2 = PARAMS['w_2']
    OO = basis(3,0)

    eps = PARAMS['bias']
    V = PARAMS['V']

    energies, states = exciton_states(PARAMS)
    psi_m = states[0]
    psi_p = states[1]
    eta = np.sqrt(eps**2 + 4*V**2)

    PARAMS['beta_1'] = beta_1 = beta_f(PARAMS['T_1'])
    PARAMS['beta_2'] = beta_2 = beta_f(PARAMS['T_2'])
    MM = psi_m*psi_m.dag()
    PP = psi_p*psi_p.dag()
    MP = psi_m*psi_p.dag()
    PM = psi_p*psi_m.dag()
    J = J_underdamped
    site_1 = (0.5*((eta-eps)*MM + (eta+eps)*PP) +V*(PM + MP))/eta
    Z_1 = (Decay(0, beta_1, J, PARAMS['alpha_1'], PARAMS['Gamma_1'], PARAMS['w0_1'])*((eta-eps)*MM + (eta+eps)*PP))/(2.*eta)
    Z_1 += (V/eta)*Decay(eta, beta_1, J, PARAMS['alpha_1'], PARAMS['Gamma_1'], PARAMS['w0_1'])*PM
    Z_1 += (V/eta)*Decay(-eta, beta_1, J, PARAMS['alpha_1'], PARAMS['Gamma_1'], PARAMS['w0_1'])*MP
    site_2 = (0.5*((eta+eps)*MM + (eta-eps)*PP) -V*(PM + MP))/eta

    Z_2 = (Decay(0, beta_2, J, PARAMS['alpha_2'], PARAMS['Gamma_2'], PARAMS['w0_2'])*((eta+eps)*MM + (eta-eps)*PP))/(2.*eta)
    Z_2 -= (V/eta)*Decay(eta, beta_2, J, PARAMS['alpha_2'], PARAMS['Gamma_2'], PARAMS['w0_2'])*PM
    Z_2 -= (V/eta)*Decay(-eta, beta_2, J, PARAMS['alpha_2'], PARAMS['Gamma_2'], PARAMS['w0_2'])*MP
    # Initialise liouvilliian
    L =  qt.spre(site_1*Z_1) - qt.sprepost(Z_1, site_1)
    L += qt.spost(Z_1.dag()*site_1) - qt.sprepost(site_1, Z_1.dag())
    L +=  qt.spre(site_2*Z_2) - qt.sprepost(Z_2, site_2)
    L += qt.spost(Z_2.dag()*site_2) - qt.sprepost(site_2, Z_2.dag())
    # Second attempt
    #print site_1, site_2
    if not silent:
        print "Weak coupling Liouvillian took {:0.2f} seconds".format(time.time()-ti)
    return -L

def get_wc_H_and_L(H_sub, sigma, T_ph, Gamma, wRC, alpha_ph, N, w_laser=0.,silent=False):
    import optical as opt
    w_1 = PARAMS['w_1']
    w_2 = PARAMS['w_2']
    OO, XO, OX = basis(3,0), basis(3,1), basis(3,2)
    sigma_m1 =  OO*XO.dag()
    sigma_m2 =  OO*OX.dag()
    eps = PARAMS['bias']
    V = PARAMS['V']
    H = w_1*XO*XO.dag() + w_2*OX*OX.dag() + V*(OX*XO.dag() + XO*OX.dag())
    L = L_weak_phonon_SES(PARAMS, silent=False)
    N_1 = PARAMS['N_1']
    N_2 = PARAMS['N_2']
    exc = PARAMS['exc']
    mu = PARAMS['mu']

    sigma = sigma_m1 + mu*sigma_m2
    
    if abs(PARAMS['alpha_EM'])>0:
        L += opt.L_BMME(H, sigma, PARAMS, ME_type='nonsecular', site_basis=True, silent=silent)
    
    return H, L
