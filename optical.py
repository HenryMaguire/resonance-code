"""
The four electromagnetic liouvillians I am studying:
- no rotating wave approximation
- no secular approximation
- a secular approximation
- an approximation which says that the enlarged system eigenstates are the same as the uncoupled system eigenstates

"""

import numpy as np
import scipy as sp
from scipy import integrate
import qutip as qt
from qutip import destroy, tensor, qeye, spost, spre, sprepost
import time
from utils import J_minimal, beta_f, J_minimal_hard, J_underdamped, J_multipolar
import sympy
from numpy import pi

def Occupation(omega, T, time_units='cm'):
    conversion = 0.695
    if time_units == 'ev':
        conversion == 8.617E-5
    if time_units == 'ps':
        conversion == 0.131
    else:
        pass
    n =0.
    beta = 0.
    if T ==0.: # First calculate beta
        n = 0.
        beta = np.infty
    else:
        # no occupation yet, make sure it converges
        beta = 1. / (conversion*T)
        if sp.exp(omega*beta)-1 ==0.:
            n = 0.
        else:
            n = float(1./(sp.exp(omega*beta)-1))
    return n


def rate_up_super(epsilon, N, alpha):
    return 0.5*np.pi*alpha*N*(epsilon**3)

def rate_down_super(epsilon, N, alpha):
    return 0.5*np.pi*alpha*(N+1)*(epsilon**3)

def rate_up(epsilon, N, alpha):
    return 0.5*np.pi*alpha*N

def rate_down(epsilon, N, alpha):
    return 0.5*np.pi*alpha*(N+1)

"""
def J_ohmic(omega, alpha, wc=10000):
    return alpha*omega*np.exp(-omega/wc)


def J_multipolar(omega, Gamma, omega_0):
    if omega==omega_0:
        print "HELLO"
        return Gamma/(2*np.pi)
    else:
        return Gamma*(omega**3)/(2*np.pi*(omega_0**3))

def J_minimal(omega, Gamma, omega_0):
    if omega==omega_0:
        return Gamma/(2*np.pi)
    else:
        return Gamma*omega/(2*np.pi*omega_0)
def J_flat(omega, Gamma, omega_0):
    return Gamma"""

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

def int_conv(f, a, inc, omega, tol=1E-5):
    x = inc
    I = 0.
    while abs(f(x))>tol:
        #print ince x, f(x), a, omega
        I += integrate.quad(f, a, x, weight='cauchy', wvar=omega)[0]
        a+=inc
        x+=inc
        #time.sleep(0.1)
    #print(("Integral converged to {} with step size of {}".format(I, inc)))
    return I # Converged integral

def integral_converge(f, a, omega, tol=1e-5):
    for inc in [200., 100., 50., 25., 10, 5., 1, 0.5]:
        try:
            return int_conv(f, a, inc, omega, tol=tol)
        except:
            if inc == 0.5:
                raise ValueError("Integrals couldn't converge")
            else:
                pass
                
    

def DecayRate(omega, beta, J, Gamma, w0, imag_part=True, tol=1e-5, alpha=0.):
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
        if J == J_underdamped:
            #print("USING UNDERDAMPED SD")
            G = (pi*alpha*Gamma)/(beta*(w0**2))
        elif J == J_multipolar:
            G=0.
        else:
            print("Assuming J_minimal")
            G = (np.pi/2)*(2*Gamma/beta)
            # G = Gamma/(2*beta*w0)
        # The limit as omega tends to zero is zero for superohmic case?
        if imag_part:
            G += -(1j)*integral_converge(F_0, -1e-12,0., tol=tol)
        #print (integrate.quad(F_0, -1e-12, 20, weight='cauchy', wvar=0)[0])
    elif omega<0.:
        G = (np.pi/2)*(coth(beta*abs(omega)/2.)+1)*J(abs(omega),Gamma, w0, alpha=alpha)
        if imag_part:
            G += (1j/2.)*integral_converge(F_m, 0,-abs(omega), tol=tol)
            G -= (1j/2.)*integral_converge(F_p, 0,abs(omega), tol=tol)
        #print integrate.quad(F_m, 0, n, weight='cauchy', wvar=-abs(omega)), integrate.quad(F_p, 0, n, weight='cauchy', wvar=abs(omega))
    return G

def L_non_rwa(H_vib, A, w_0, Gamma, T_EM, J, principal=False, 
                                silent=False, alpha=0., tol=1e-5):
    ti = time.time()
    beta = beta_f(T_EM)
    eVals, eVecs = H_vib.eigenstates()
    #J=J_minimal # J_minimal(omega, Gamma, omega_0)
    d_dim = len(eVals)
    G = 0
    for i in range(d_dim):
        for j in range(d_dim):
            eta = eVals[i]-eVals[j]
            s = eVecs[i]*(eVecs[j].dag())
            #print A.matrix_element(eVecs[i].dag(), eVecs[j])
            overlap = A.matrix_element(eVecs[i].dag(), eVecs[j])
            s*= A.matrix_element(eVecs[i].dag(), eVecs[j])
            s*= DecayRate(eta, beta, J, Gamma, w_0, imag_part=principal, alpha=alpha, tol=tol)
            G+=s
    G_dag = G.dag()
    # Initialise liouvilliian
    L =  qt.spre(A*G) - qt.sprepost(G, A)
    L += qt.spost(G_dag*A) - qt.sprepost(A, G_dag)
    if not silent:
        print(("Calculating non-RWA Liouvilliian took {} seconds.".format(time.time()-ti)))
    return -L

def RWA_system_ops(H_vib, S):
    S_plus = 0 #np.zeros(shape=H_vib.shape)
    S_minus = 0 #np.zeros(shape=H_vib.shape)
    S_0 = 0 #np.zeros(shape=H_vib.shape)
    eVals, eVecs = H_vib.eigenstates()

    for j, phi_j in enumerate(eVecs):
        for k, phi_k in enumerate(eVecs):
            S_jk = S.matrix_element(phi_j.dag(), phi_k)
            S_contrib = S_jk*phi_j*phi_k.dag()
            if eVals[j]>eVals[k]:
                S_plus += S_contrib
            elif eVals[j]<eVals[k]:
                S_minus += S_contrib
            else:
                S_0 += S_contrib
    assert S_plus == S_minus.dag()
    #if S_0 !=0:
    #    print "Some non-zero S_0 contribution!"
    return S_plus, S_minus, S_0

"""
def L_nonrwa(H_vib, sig_x, omega_0, Gamma, T, J, time_units='cm'):
    ti = time.time()
    d = H_vib.shape[0]
    evals, evecs = H_vib.eigenstates()
    Z = 0 # initalise rate operator
    Z_dag = 0
    for i in range(int(d)):
        for j in range(int(d)):
            eps_ij = (evals[i]-evals[j])
            sig_ij = sig_x.matrix_element(evecs[i].dag(), evecs[j])
            sig_ji = (sig_x.dag()).matrix_element(evecs[j].dag(), evecs[i])
            IJ = evecs[i]*evecs[j].dag()
            JI = evecs[j]*evecs[i].dag()
            if abs(sig_ij) >0:
                Z+= decay_rate(J, eps_ij, Gamma, omega_0, T, time_units)*sig_ij*IJ
                Z_dag += decay_rate(J, eps_ij, Gamma, omega_0, T, time_units).conjugate()*sig_ji*JI
    L = spre(sig_x*Z) - sprepost(Z, sig_x) + spost(Z_dag*sig_x) - sprepost(sig_x, Z_dag)
    print "It took ", time.time()-ti, " seconds to build the non-RWA Liouvillian"
    return -L
"""

def L_nonsecular(H_vib, A, eps, Gamma, T, J, 
                time_units='cm', silent=False, w_laser=0.):
    #Construct non-secular liouvillian
    ti = time.time()
    d = H_vib.shape[0]
    evals, evecs = H_vib.eigenstates()
    X1, X2, X3, X4 = 0, 0, 0, 0
    for i in range(int(d)):
        for j in range(int(d)):
            eps_ij = abs(evals[i]-evals[j]) + w_laser
            A_ij = A.matrix_element(evecs[i].dag(), evecs[j])
            A_ji = (A.dag()).matrix_element(evecs[j].dag(), evecs[i])
            Occ = Occupation(eps_ij, T, time_units)
            IJ = evecs[i]*evecs[j].dag()
            JI = evecs[j]*evecs[i].dag()
            # 0.5*np.pi*alpha*(N+1)
            if abs(A_ij)>0 or abs(A_ji)>0:
                r_up = 2*np.pi*J(eps_ij, Gamma, eps)*Occ
                r_down = 2*np.pi*J(eps_ij, Gamma, eps)*(Occ+1)
                X3+= r_down*A_ij*IJ
                X4+= r_up*A_ij*IJ
                X1+= r_up*A_ji*JI
                X2+= r_down*A_ji*JI
    L = spre(A*X1) -sprepost(X1,A)+spost(X2*A)-sprepost(A,X2)
    L+= spre(A.dag()*X3)-sprepost(X3, A.dag())+spost(X4*A.dag())-sprepost(A.dag(), X4)
    if not silent:
        print(("It took ", time.time()-ti, " seconds to build the Non-secular RWA Liouvillian"))
    return -0.5*L


def L_nonsecular_old(H_vib, A, eps, Gamma, T, J, time_units='cm', silent=False):
    #Construct non-secular liouvillian
    ti = time.time()
    d = H_vib.shape[0]
    evals, evecs = H_vib.eigenstates()
    X1, X2, X3, X4 = 0, 0, 0, 0
    for i in range(int(d)):
        for j in range(int(d)):
            eps_ij = abs(evals[i]-evals[j])
            A_ij = A.matrix_element(evecs[i].dag(), evecs[j])
            A_ji = (A.dag()).matrix_element(evecs[j].dag(), evecs[i])
            Occ = Occupation(eps_ij, T, time_units)
            IJ = evecs[i]*evecs[j].dag()
            JI = evecs[j]*evecs[i].dag()
            # 0.5*np.pi*alpha*(N+1)
            if abs(A_ij)>0 or abs(A_ji)>0:
                r_up = 2*np.pi*J(eps_ij, Gamma, eps)*Occ
                r_down = 2*np.pi*J(eps_ij, Gamma, eps)*(Occ+1)
                X3+= r_down*A_ij*IJ
                X4+= r_up*A_ij*IJ
                X1+= r_up*A_ji*JI
                X2+= r_down*A_ji*JI

    L = spre(A*X1) -sprepost(X1,A)+spost(X2*A)-sprepost(A,X2)
    L+= spre(A.dag()*X3)-sprepost(X3, A.dag())+spost(X4*A.dag())-sprepost(A.dag(), X4)
    if not silent:
        print(("It took ", time.time()-ti, " seconds to build the Non-secular RWA Liouvillian"))
    return -0.5*L

def L_full_secular(H_vib, A, eps, Gamma, T, J, time_units='cm', silent=False):
    '''
    Initially assuming that the vibronic eigenstructure has no
    degeneracy and the secular approximation has been made
    '''
    ti = time.time()
    d = H_vib.shape[0]
    L = 0
    eVals, eVecs = H_vib.eigenstates()
    A_dag = A.dag()
    terms = 0
    for l in range(int(d)):
        for m in range(int(d)):
            for p in range(int(d)):
                for q in range(int(d)):
                    secular_freq = (eVals[l]-eVals[m]) - (eVals[p]-eVals[q])
                    if abs(secular_freq) <1E-10:
                        terms+=1
                        A_lm = A.matrix_element(eVecs[l].dag(), eVecs[m])
                        A_lm_star = A_dag.matrix_element(eVecs[m].dag(), eVecs[l])
                        A_pq = A.matrix_element(eVecs[p].dag(), eVecs[q])
                        A_pq_star = A_dag.matrix_element(eVecs[q].dag(), eVecs[p])
                        coeff_1 = A_lm*A_pq_star
                        coeff_2 = A_lm_star*A_pq
                        eps_pq = abs(eVals[p]-eVals[q])
                        Occ = Occupation(eps_pq, T, time_units)
                        r_up = np.pi*J(eps_pq, Gamma, eps)*Occ
                        r_down = np.pi*J(eps_pq, Gamma, eps)*(Occ+1)
                        LM = eVecs[l]*eVecs[m].dag()
                        ML = LM.dag()
                        PQ = eVecs[p]*eVecs[q].dag()
                        QP = PQ.dag()
                        """
                        if abs(secular_freq) !=0:
                            print (abs(secular_freq), r_up, A_lm, A_lm_star,
                                   A_pq, A_pq_star, r_down, l,m,p,q, m==q, l==p)
                        """
                        if abs(r_up*coeff_1)>0:
                            L+= r_up*coeff_1*(spre(LM*QP)-sprepost(QP,LM))
                        if abs(r_up*coeff_2)>0:
                            L+= r_up*coeff_2*(spost(PQ*ML)- sprepost(ML,PQ))
                        if abs(r_down*coeff_1)>0:
                            L+= r_down*coeff_1*(spre(ML*PQ)-sprepost(PQ, ML))
                        if abs(r_down*coeff_2)>0:
                            L+= r_down*coeff_2*(spost(QP*LM)-sprepost(LM, QP))
    if not silent:
        print(("It took ", time.time()-ti, " seconds to build the secular Liouvillian"))
        print(("Secular approximation kept {:0.2f}% of total ME terms. \n".format(100*float(terms)/(d*d*d*d))))
    return -L

def L_vib_lindblad(H_vib, A, eps, Gamma, T, J, time_units='cm', silent=False):
    '''
    Initially assuming that the vibronic eigenstructure has no
    degeneracy and the secular approximation has been made
    '''
    ti = time.time()
    d = H_vib.shape[0]
    ti = time.time()
    L = 0
    eig = H_vib.eigenstates()
    eVals = eig[0]
    eVecs = eig[1] # come out like kets
    l = 0
    occs=[]
    for i in range(int(d)):
        l = 0
        for j in range(int(d)):
            t_0 = time.time() # initial time reference for tracking slow calculations
            lam_ij = A.matrix_element(eVecs[i].dag(), eVecs[j])
            #lam_mn = (A.dag()).matrix_element(eVecs[n].dag(), eVecs[m])
            lam_ij_sq = lam_ij*lam_ij.conjugate() # real by construction
            eps_ij = abs(eVals[i]-eVals[j])
            if abs(lam_ij_sq)>0:
                IJ = eVecs[i]*eVecs[j].dag()
                JI = eVecs[j]*eVecs[i].dag()
                JJ = eVecs[j]*eVecs[j].dag()
                II = eVecs[i]*eVecs[i].dag()

                Occ = Occupation(eps_ij, T, time_units)
                r_up = 2*np.pi*J(eps_ij, Gamma, eps)*Occ
                r_down = 2*np.pi*J(eps_ij, Gamma, eps)*(Occ+1)

                T1 = r_up*spre(II)+r_down*spre(JJ)
                T2 = r_up.conjugate()*spost(II)+r_down.conjugate()*spost(JJ)
                T3 = (r_up*sprepost(JI, IJ)+r_down*sprepost(IJ,JI))
                L += lam_ij_sq*(0.5*(T1 + T2) - T3)
                l+=1
    if not silent:
        print(("It took ", time.time()-ti, " seconds to build the vibronic Lindblad Liouvillian"))
    return -L

def L_EM_lindblad(splitting, col_em, Gamma, T, J, time_units='cm', silent=False):
    # col_em is collapse operator
    ti = time.time()
    L = 0
    EMnb = Occupation(splitting, T, time_units)
    L+= 2*np.pi*J(splitting, Gamma, splitting)*(EMnb+1)*(sprepost(col_em, col_em.dag())-0.5*(spre(col_em.dag()*col_em) +spost(col_em.dag()*col_em)))
    L+= 2*np.pi*J(splitting, Gamma, splitting)*EMnb*(sprepost(col_em.dag(), col_em)-0.5*(spre(col_em*col_em.dag())+ spost(col_em*col_em.dag())))
    if not silent:
        print(("It took ", time.time()-ti, " seconds to build the electronic-Lindblad Liouvillian"))
    return L
