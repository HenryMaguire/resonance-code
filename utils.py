import numpy as np
from numpy import pi
import scipy as sp
from qutip import spre, spost, sprepost
import qutip as qt
import pickle
import sympy

print( "utils imported")

ev_to_inv_cm = 8065.5
inv_ps_to_inv_cm = 5.309

def load_obj(name ):
    with open(name + '.pickle', 'rb') as f:
        return pickle.load(f)

def save_obj(obj, name ):
    with open(name + '.pickle', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

#def Coth(x):
#    return (np.exp(2*x)+1)/(np.exp(2*x)-1)

def Coth(x):
    return float(sympy.coth(x))

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


def beta_f(T):
    conversion = 0.695
    beta = 0
    if T ==0.: # First calculate beta
        beta = np.infty
    else:
        # no occupation yet, make sure it converges
        beta = 1. / (conversion*T)
    return beta

def J_poly(omega, Gamma, omega_0, ohmicity=1, alpha=0.):
    # this won't work here
    return Gamma*(omega**ohmicity)/(2*np.pi*(omega_0**ohmicity))

def J_multipolar(omega, Gamma, omega_0, alpha=0.):
    if omega==omega_0:
        return Gamma/(2*np.pi)
    else:
        return Gamma*(omega**3)/(2*np.pi*(omega_0**3))

def J_minimal(omega, Gamma, omega_0, alpha=0.):
    if omega==omega_0:
        return Gamma/(2*np.pi)
    else:
        return Gamma*omega/(2*np.pi*omega_0)

def J_minimal_hard(omega, Gamma, omega_0, cutoff):
    if omega <cutoff:
        return Gamma*omega/(omega_0) #2*np.pi*
    else:
        return 0.

def J_RC(omega, gamma, omega_0):
    return gamma*omega

def J_flat(omega, Gamma, omega_0):
    return Gamma

#def J_overdamped(omega, alpha, wc):
#    return alpha*wc*omega/(omega**2+wc**2)


"""def J_underdamped(omega, Gamma, omega_0, alpha=0.):
    return alpha*Gamma*pow(omega_0,2)*omega/(pow(pow(omega_0,2)-pow(omega,2),2)+(Gamma**2 *omega**2))
"""

def J_underdamped(omega, Gamma, omega_0, alpha=0.):
    return alpha*Gamma*(omega_0**2)*omega/(((omega_0**2)-(omega**2))**2+(Gamma*omega)**2)


def rate_up(w, n, gamma, J, w_0):
    rate = 0.5 * pi * n * J(w, gamma, w_0)
    return rate

def rate_down(w, n, gamma, J, w_0):
    rate = 0.5 * pi * (n + 1. ) * J(w, gamma, w_0)
    return rate

def lin_construct(O):
    Od = O.dag()
    L = 2. * spre(O) * spost(Od) - spre(Od * O) - spost(Od * O)
    return L

def ground_and_excited_states(states):
    # For a TLS, gives separate ground and excited state manifolds
    ground_list = []
    excited_list = []
    concat_list = [ground_list, excited_list]
    for i in range(len(states)): #
        is_ground = sum(states[i])[0][0].real == 1.
        if is_ground:
            ground_list.append(i)
        else:
            excited_list.append(i)
    return ground_list, excited_list


def initialise_TLS(init_sys, init_RC, states, w0, T_ph, H_RC=np.ones((2,2))):
    # allows specific state TLS-RC states to be constructed easily
    G = qt.ket([0])
    E = qt.ket([1])
    ground_list, excited_list = ground_and_excited_states(states)
    concat_list = [ground_list, excited_list]
    N = states[1].shape[0]/2
    if init_sys == 'coherence':
        rho_left = (states[concat_list[0][init_RC]]+states[concat_list[1][init_RC]])/np.sqrt(2)
        rho_right = rho_left.dag()
        init_rho = rho_left*rho_right
    elif type(init_sys) == tuple:
        #coherence state
        if type(init_RC) == tuple:
            # take in a 2-tuple to initialise in coherence state.
            print(( init_sys, init_RC))
            rho_left = states[concat_list[init_sys[0]][init_RC[0]]]
            rho_right = states[concat_list[init_sys[1]][init_RC[1]]].dag()
            init_rho = rho_left*rho_right
        else:
            raise ValueError
    elif init_sys == 0:
        # population state
        init_rho = states[ground_list[init_RC]]*states[ground_list[init_RC]].dag()
    elif init_sys==1:
        init_rho = states[excited_list[init_RC]]*states[excited_list[init_RC]].dag()
    elif init_sys==2:
        Therm = qt.thermal_dm( N, Occupation(w0, T_ph))
        init_rho = qt.tensor(E*E.dag(), Therm)
        # if in neither ground or excited
        # for the minute, do nothing. This'll be fixed below.
    else:
        # finally, if not in either G or E, initialise as thermal
        num = (-H_RC*beta_f(T_ph)).expm()
        init_rho =  num/num.tr()
    return init_rho

def exciton_states(detuning, Omega, shift=0.):
    detuning+=shift
    eps = detuning
    eta = np.sqrt(eps**2 + (Omega**2))
    lam_m = (-detuning-eta)*0.5
    lam_p = (-detuning+eta)*0.5
    v_p = qt.Qobj(np.array([np.sqrt(eta+eps), np.sqrt(eta-eps)]))/np.sqrt(2*eta)
    v_m = qt.Qobj(np.array([np.sqrt(eta-eps), -np.sqrt(eta+eps)]))/np.sqrt(2*eta)

    return [lam_m, lam_p], [v_m, v_p]

def fourier(timelist, signal, absval=False):
    spec = sp.fftpack.fft(signal)
    dt = timelist[1]-timelist[0]
    freq = 2 * pi * np.array(sp.fftpack.fftfreq(spec.size, dt))
    spec = 2 * dt* np.real(spec)
    if absval:
        spec = 2 * dt* np.abs(spec)
    return freq, spec

def plot_fourier(tlist, signal, vline=None, absval=False, x_lim = None):
    plt.figure()
    freq, spec = fourier(tlist, signal-signal[-1], absval=absval)
    freq, spec = zip(*sorted(zip(freq, np.array(spec).real)))
    plt.plot(freq, spec)
    if vline is not None:
        plt.axvline(vline, ls='dotted')
    if x_lim is not None:
        plt.xlim(-x_lim, x_lim)
    else:
        plt.xlim(freq[0], freq[-1])
    plt.show()
