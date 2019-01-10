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

