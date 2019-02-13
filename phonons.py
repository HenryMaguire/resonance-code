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
from qutip import destroy, tensor, qeye, spre, spost, sprepost, Qobj, identity, thermal_dm, basis
from utils import Coth, beta_f, ev_to_inv_cm, Occupation
from numpy import sqrt, pi
import copy

#import pdb; pdb.set_trace()

def Ham_RC(sigma, eps, Omega, kappa, N, rotating=False):
    """
    Input: System splitting, RC freq., system-RC coupling and Hilbert space dimension
    Output: Hamiltonian, sigma_- and sigma_z in the vibronic Hilbert space
    """

    a = destroy(N)
    shift = 0.5*(kappa**2)/Omega
    I_sys = Qobj(qeye(sigma.shape[0]),dims=sigma.dims)
    sys_energy = (eps+shift)
    if rotating:
        sys_energy=0.
    H_S = sys_energy*tensor(sigma.dag()*sigma, qeye(N)) + kappa*tensor(sigma.dag()*sigma, (a + a.dag())) + tensor(I_sys,Omega*a.dag()*a)
    A_em = tensor(sigma, qeye(N))
    A_nrwa = tensor(sigma+sigma.dag(), qeye(N))
    A_ph = tensor(I_sys, (a + a.dag()))
    return H_S, A_em, A_nrwa, A_ph

def Ham_RC_gen(H_sub, sigma, Omega, kappa, N, rotating=False, shift = 0., shift_op=None, w_laser=0.):
    """
    will only work for spin-boson like models
    Input: System Hamiltonian, RC freq., system-RC coupling and Hilbert space dimension
    Output: Hamiltonian, sigma_- and sigma_z in the vibronic Hilbert space
    """
    a = destroy(N)
    I_sys = Qobj(qeye(H_sub.shape[0]),dims=sigma.dims)
    #print(( "shift is ", shift))
    H_sub += shift_op*shift # shift is zero if no shift is required
    if rotating:
        # Hopefully removes energy scale. Shift operator should be the same as
        # the site energy-scale operator.
        H_sub -= shift_op*H_sub*(shift_op.dag())
    H_S = tensor(H_sub, qeye(N)) + kappa*tensor(shift_op, (a + a.dag()))
    H_S += tensor(I_sys, Omega*a.dag()*a)
    A_em = tensor(sigma, qeye(N))
    A_nrwa = tensor(sigma+sigma.dag(), qeye(N))
    A_ph = tensor(I_sys, a)
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
        print("w_RC={} | TLS splitting = {} | RC-res. coupling={:0.2f} | TLS-RC coupling={:0.2f} | Gamma_RC={:0.2f} | alpha_ph={:0.2f} | N={} |".format(wRC, eps, gamma,  kappa, Gamma, alpha_ph, N))
    H, A_em, A_nrwa, A_ph = Ham_RC(sigma, eps, wRC, kappa, N, rotating=rotating)
    if new:
        L_RC, Z =  liouvillian_build_new(H, A_ph, gamma, wRC, T_ph)
    else:
        L_RC, Z =  liouvillian_build(H, A_ph, gamma, wRC, T_ph)
    return L_RC, H, A_em, A_nrwa, Z, wRC, kappa, Gamma

def underdamped_shift(alpha, Gamma, w0):
    sfactor = sqrt(Gamma**2 -4*w0**2)
    denom = sqrt(2)*(sqrt(Gamma**2 - 2*w0**2 - Gamma*sfactor)+sqrt(
                        Gamma**2 - 2*w0**2 + Gamma*sfactor))
    return pi*alpha*Gamma/denom


def mapped_constants(w0, alpha_ph, Gamma):
    gamma = Gamma / (2. * np.pi * w0)  # coupling between RC and residual bath
    kappa= np.sqrt(np.pi * alpha_ph * w0 / 2.)  # coupling strength between the TLS and RC
    shift = 0.5*pi*alpha_ph/2
    return w0, gamma, kappa, shift

def mapped_operators_and_constants(H_sub, sigma, T_ph, Gamma, Omega, alpha_ph, N, 
                                    w_laser=0., shift=True, shift_op=None):
    wRC, gamma, kappa = mapped_constants(Omega, alpha_ph, Gamma)
    H, A_em, A_nrwa, A_ph = Ham_RC_gen(H_sub, sigma, wRC, kappa, N,
                                        shift_op=shift_op, shift=shift, 
                                        w_laser=w_laser)
    return H, A_em, A_ph, wRC, gamma, kappa

                                    
def RC_mapping_general(H_sub, sigma, T_ph, Gamma, Omega, alpha_ph, N, silent=False,
                                            residual_off=False, rotating=False,
                                            shift_op = None, shift=True, new=False, w_laser=0.):
    
    # we define all of the RC parameters by the underdamped spectral density
    wRC, gamma, kappa, energy_shift = mapped_constants(Omega, alpha_ph, Gamma)

    if not silent:
        print("w_RC={} | RC-res. coupling={:0.4f} | TLS-RC coupling={:0.2f} | Gamma_RC={:0.2f} | alpha_ph={:0.2f} | N={} |".format(wRC, gamma,  kappa, Gamma, alpha_ph, N))
    if shift_op is None:
        shift_op = sigma.dag()*sigma
    if shift:
        shift = energy_shift
    else:
        shift = 0.
    H, A_em, A_nrwa, A_ph = Ham_RC_gen(H_sub, sigma, wRC, kappa, N,
                                        rotating=rotating,
                                        shift_op=shift_op, shift=shift, 
                                        w_laser=w_laser)
    if not new:
        L_RC, Z =  liouvillian_build(H, A_ph+A_ph.dag(), gamma, wRC, T_ph)
    else:
        L_RC, Z =  liouvillian_build_new(H, A_ph+A_ph.dag(), gamma, wRC, T_ph)
    return L_RC, H, A_em, A_ph, Z, wRC, kappa, gamma


I_sys = identity(2)
def displace(offset, a):
    return (offset*(a.dag()) - offset.conjugate()*a).expm()

def undisplaced_initial(init_sys, w0, T, N):
    n = Occupation(w0, T)
    return tensor(init_sys, thermal_dm(N, n))

def position_ops(N):
    a = destroy(N)
    
    return tensor(I_sys, (a + a.dag())*0.5) # Should have a 0.5 in this

def displaced_initial(init_sys, alpha, w0, T, N, silent=False, return_error=False):
    # Works for 
    offset = 0.5*sqrt(pi*alpha/(2*w0))
    a = destroy(N)
    x = position_ops(N)
    
    r0 = undisplaced_initial(init_sys, w0, T, N)
    disp = copy.deepcopy(r0)
    
    d = tensor(I_sys, displace(offset, a))
    disp =  d * disp * d.dag()
    
    error = 100*(abs((disp*x).tr()- offset))/offset
    if not silent:
        print ("Error in displacement: {:0.8f}%".format(error))
        print("Ratio of kBT to Omega: {:0.4f}".format(0.695*T/w0))
    if return_error:   
        return disp, error
    else:
        return disp

def get_converged_N(alpha, w0, T, err_threshold=1e-2, min_N=4, max_N=28, silent=True):
    err = 0
    if alpha ==0:
        return 3
    for N in range(min_N,max_N+1):           
        disp, err = displaced_initial(basis(2,0)*basis(2,0).dag(), alpha, w0, T, N, silent=True, return_error=True)
        if err<err_threshold:
            return N
    print("Error could only converge to {}".format(err))
    return N

def N_estimate_eV(alpha, w0, T, err_threshold=0.1, min_N=4, max_N=28, silent=True):
    # params come in in eV
    alpha, w0 = alpha*ev_to_inv_cm, w0*ev_to_inv_cm
    N_base = get_converged_N(alpha, w0, T, err_threshold=err_threshold, min_N=min_N, max_N=max_N, silent=silent)
    m = alpha/(0.5*1e-1*ev_to_inv_cm)
    return min((6+int(m*N_base)), N_base)

def N_estimate(alpha, w0, T, err_threshold=0.1, min_N=4, max_N=28, silent=True):
    N_base = get_converged_N(alpha, w0, T, err_threshold=err_threshold, min_N=min_N, max_N=max_N, silent=silent)
    m = alpha/(0.5*1e-1*ev_to_inv_cm)
    return min((6+int(m*N_base)), N_base)

