################################################
# Quantisation functions
################################################

import numpy as np
import scipy.stats as scs
import scipy.special as sp
import cmath

### Decomposition
def decompose_rectangular(seq):
    """
    Decompose a complex sequence into real and imaginary part.
    Input:
        seq: complex sequence
    Outputs:
        real_seq: sequence of real part
        imag_seq: sequence of imaginary part
    """
    seq      = np.array(seq)
    real_seq = np.real(seq)
    imag_seq = np.imag(seq)
    
    return real_seq, imag_seq

def decompose_polar(seq):
    """
    Decompose a complex sequence into absolute value and angle.
    Input:
        seq: complex sequence
    Outputs:
        abs_seq: sequence of absolute values
        ang_seq: sequence of angle values
    """
    seq     = np.array(seq)
    abs_seq = np.abs(seq)
    ang_seq = np.angle(seq)
    
    return abs_seq, ang_seq

### Uniform compander quantisation
def quantise_uniform(x, m):
    """
    Apply quantisation with uniform compander.
    Inputs:
        x: original sequence in [0,1]
        m: alphabet size
    """

    assert np.max(x)<=1 and np.min(x)>=0, "x should be in [0,1]. min(x)={}, max(x)={}".format(np.min(x), np.max(x)) 
    
    x = np.array(x).clip(0,1)
    
    Delta = 1/m                    # Spacing
    idx   = np.round(m * x - 0.5)  # Quantise
    idx   = idx.clip(0,m-1)
    x_hat = idx * Delta + Delta/2  # Representation points
    
    idx = np.array(idx, dtype=int)
    assert np.min(idx)>=0 and np.max(idx)<m, "uniform quantisation m={}, min={}, max={}".format(m,np.min(idx),np.max(idx))
    assert np.min(x_hat)>=0 and np.max(x_hat)<=1, "x_hat should be in [0,1], but is in [{},{}]".format(np.min(x_hat),np.max(x_hat))

    return idx, x_hat

### Mu-law compander quantisation
def mu_compression(x, mu):
    x   = np.array(x)
    aux = 1+mu*x
    return np.log(aux, where=0<aux, out=[0]*aux)/np.log(1+mu)

def mu_expansion(x, mu):
    x = np.array(x)
    return (1/mu) * ((1+mu)**x - 1)

def find_mu(x, M, mu0=1e-4, Nmax=200, step=0.1, mu_min=1e-5, mu_max=1e10):
    """
    Find mu value that minimises KL-divergence.
    Inputs:
        x:      original sequences in [0,1]
        M:      alphabet size
        mu0:    mu initial value
        Nmax:   maximum number of iterations
        step:   step size for Newton's method
        mu_min: minimum value for mu
        mu_max: maximum value for mu
    Output:
        mu: optimised mu value
    """

    x = np.array(x)
    
    # Initialisation
    mu = mu0
    rho = np.inf
    i = 0
        
    # Find a compander that uniformises population
    while(np.abs(rho)>0.01 and i<Nmax):
        J1 = 1/mu - (1+mu)**(-1)/np.log(1+mu) - np.mean(x/(1+mu*x))
        J2 = -1/mu**2 + (1+1/np.log(1+mu)) * (1+mu)**(-2)/np.log(1+mu) + np.mean((x/(1+mu*x))**2)
        rho = J1/J2
        mu -= step*rho
        if mu<mu_min:
            mu = mu_min
            break
        if mu>mu_max:
            mu = mu_max
            break
        i += 1
    
    # Adjust such that first and last interval have same average error
    if M>0:
        x_compressed = mu_compression(x,mu)
        N1 = np.sum(x_compressed < 1/M)
        NM = np.sum(x_compressed > (M-1)/M)
        Delta1 = mu_expansion(1/M, mu)
        DeltaM = 1 - mu_expansion((M-1)/M, mu)

        mu_next = mu

        while (N1*Delta1**2 < NM*DeltaM**2 and mu > 1e-8):
            mu      = mu_next
            mu_next = mu/2

            x_compressed = mu_compression(x,mu_next)
            N1 = np.sum(x_compressed < 1/M)
            NM = np.sum(x_compressed > (M-1)/M)
            Delta1 = mu_expansion(1/M, mu_next)
            DeltaM = 1 - mu_expansion((M-1)/M, mu_next)
    
    mu = max(mu, mu_min)
    
    return mu

def quantise_mu(x, M, mu0=1e-4, Nmax=200, step=0.1, mu_min=1e-5, mu_max=1e10):
    """
    Apply quantisation with mu-law compander to a sequence.
    Inputs:
        x: original sequence in [0,1]
        M: alphabet size
        mu0, Nmax, step, mu_min, mu_max: parameters for find_mu
    Outputs:
        idx:   quantisation indices
        x_hat: quantisation points
    """
    
    x = np.array(x)
    mu = find_mu(x, M, mu0, Nmax, step, mu_min, mu_max)
    
    # Quantisation indices
    x_mu     = mu_compression(x,mu).clip(0,1)
    idx, aux = quantise_uniform(x_mu, M)
    
    # Representation points
    ## Simple way
    x_hat    = mu_expansion(aux, mu)
    assert np.min(idx)>=0 and np.max(idx)<M, "mu quantisation M={}, min={}, max={}".format(M,np.min(idx),np.max(idx))
    
    ## Minimise MSE
    #aux     = np.linspace(0,1,M)
    #mu_dict = (1/mu) * ( (1+mu)**aux * ( (1+mu)**(1/M) - 1 ) * M/np.log(1+mu) - 1 )
    #x_hat = mu_dict[idx]
    
    return idx, x_hat

### Beta-law compander quantisation
def beta_compression(x, a, b):
    x = np.array(x)
    return scs.beta.cdf(x, a, b)

def beta_expansion(x, a, b):
    x = np.array(x)
    return scs.beta.ppf(x, a, b)

def beta_ext_distortion(x, M, a, b):
    """
    Find external distortion.
    Inputs:
        x:    sequence of symbols
        M:    alphabet size
        a, b: beta-law parameters
    """
    edges = beta_expansion(np.linspace(0,M,M+1)/M,a,b)
    x0    = 1 / (1+(b-1)/(a-1))
    
    # Distortion at the left and right extremes
    N1 = np.sum(x < edges[1])
    NM = np.sum(x > edges[M-1])
    Delta1 = (edges[1])**2
    DeltaM = (edges[M]-edges[M-1])**2
    
    L = N1 * Delta1
    R = NM * DeltaM

    if max(a,b)<1 or min(a,b)>1:
        M0 = int(np.floor(beta_expansion(x0,a,b)*M))
        # Distortion in the middle
        NC     = np.sum((x>edges[M0]) * (x<=edges[M0+1]))
        DeltaC = (edges[M0+1]-edges[M0])**2
        C      = NC * DeltaC
        
        if min(a,b)<1:
            A = C                # Largest interval in the middle
            B = R if a<b else L  # Smallest interval on the right or left according to (a,b)
        else:
            B = C                # Smallest interval in the middle
            A = R if a<b else L  # Largest interval on the right or left according to (a,b)
    else:
        [A,B] = [R,L] if a<b else [L,R]

    return A, B

def find_beta(x, M, theta0=[1, 1], Nmax=200, step=0.1, theta_min=1e-5, theta_max=1e10):
    """
    Find theta=[a,b] values that minimises KL-divergence.
    Inputs:
        x:         original sequences in [0,1]
        M:         alphabet size
        theta0:    parameters initial value
        Nmax:      maximum number of iterations
        step:      step size for Newton's method
        theta_min: minimum value for parameters
        theta_max: maximum value for parameters
    Output:
        theta=[a,b]: optimised parameters
    """
    
    x = np.array(x)
    # To avoid problems
    epsilon = 1e-10
    x = np.clip(x, epsilon, 1-epsilon)
    
    # Initialisation
    theta = theta0
    rho = np.inf
    i = 0
    v = np.array([np.mean(np.log(x)), np.mean(np.log(1-x))])
    
    # Find a compander that uniformises population
    while(np.linalg.norm(rho)>0.01 and i<Nmax):
        [a, b] = theta
        J1 = v - np.array([sp.polygamma(0,a), sp.polygamma(0,b)]) + sp.polygamma(0,a+b)
        J2 = -np.diag([sp.polygamma(1,a), sp.polygamma(1,b)]) + sp.polygamma(1,a+b)
        rho = np.linalg.solve(J2,J1)
        if np.min(theta-step*rho) >= theta_min and np.max(theta-step*rho) <= theta_max:
            theta = theta-step*rho
        else:
            if np.min(theta-step*rho) < theta_min:
                theta = np.array([theta_min, theta_min])
            else:
                theta = np.array([theta_max, theta_max])
        
        i += 1

    # Adjust beta to reduce distortion
    if M>0:
        A, B = beta_ext_distortion(x, M, theta[0], theta[1])
        l = 0.9
        i = 0
        theta_next = theta
        while (A>B and i<Nmax):
            theta = theta_next
            # Go towards [1,1] 
            theta_next = l*theta + 1-l
            A, B = beta_ext_distortion(x, M, theta_next[0], theta_next[1])
            i += 1
	
    return theta

def quantise_beta(x, M, theta0=[1, 1], Nmax=200, step=0.1, theta_min=1e-5, theta_max=1e10, show=False):
    """
    Apply quantisation with beta-law compander to a sequence.
    Inputs:
        x: original sequence in [0,1]
        M: alphabet size
        theta0, Nmax, step, theta_min, theta_max: parameters for find_beta
    Outputs:
        idx:   quantisation indices
        x_hat: quantisation points
    """
    
    x = np.array(x)
    [a,b] = find_beta(x, M, theta0, Nmax, step, theta_min, theta_max)
    if show==True:
        print([a,b])

    # Quantisation indices
    x_beta   = beta_compression(x,a,b).clip(0,1)
    idx, aux = quantise_uniform(x_beta, M)

    # Representation points
    ## Simple way
    #x_hat = beta_expansion(aux,a,b)
    ## Minimise MSE
    beta_bounds = beta_expansion(np.linspace(0,1,M+1),a,b)
    beta_dict   = np.diff(beta_compression(beta_bounds,a+1,b)) * M*a/(a+b)       
    x_hat       = beta_dict[idx]
    assert np.min(idx)>=0 and np.max(idx)<M, "beta quantisation M={}, min={}, max={}".format(M,np.min(idx),np.max(idx))
        
    return idx, x_hat

### Cube-split compander
def CS_compander(x):
    x   = np.array(x)
    x2  = np.abs(x)**2
    idx = np.where(x2>0)

    w      = np.zeros(len(x), dtype=complex)
    w[idx] = np.sqrt( 2*(np.log(1+x2[idx])-np.log(1-x2[idx])) ) * (x[idx]/np.sqrt(x2[idx]))
    y      = scs.norm.cdf(np.real(w)) + 1j* scs.norm.cdf(np.imag(w))
    
    return y

def CS_decompander(y, m):
    y = np.array(y)
    w = scs.norm.ppf((np.real(y)+0.5)/m) + 1j * scs.norm.ppf((np.imag(y)+0.5)/m)
    w2 = np.abs(w)**2
    x = np.sqrt((1-np.exp(-w2/2))/(1+np.exp(-w2/2)))*np.exp(1j*np.angle(w));
    return x

def quantise_CS(h, m):
    """
    Apply cube-split quantisation.
    Inputs:
        h: array of complex-valued coefficients
        m: alphabet size for real and imaginary parts
    Outputs:
        idx_CS_re: quantisation indices for real part
        idx_CS_im: quantisation indices for imaginary part
        hhat_CS:   complex representation points
    """
    
    h = np.array(h, dtype=complex)
    
    h_CS_comp    = CS_compander(h)

    idx_CS_re, _ = quantise_uniform(np.real(h_CS_comp), m)
    idx_CS_im, _ = quantise_uniform(np.imag(h_CS_comp), m)

    idx_CS_re    = np.array(idx_CS_re)
    idx_CS_im    = np.array(idx_CS_im)

    hhat_CS      = CS_decompander(idx_CS_re + 1j*idx_CS_im, m)
    
    return idx_CS_re, idx_CS_im, hhat_CS

### Auxiliary functions
def computeMSE(Hnorm, Hhat):
    """
    Compute mean-square error between Hnorm and Hhat.
    """
    _,_,n = Hnorm.shape
    MSE    = np.linalg.norm(Hhat - Hnorm)**2
    MSE_dB = 10*np.log10(MSE)
    return MSE, MSE_dB

def computeNMSE(Hnorm, Hhat):
    """
    Compute normalised MSE between Hnorm and Hhat.
    """
    _,_,n   = Hnorm.shape
    NMSE    = np.linalg.norm(Hhat - Hnorm)**2/np.linalg.norm(Hnorm)**2
    NMSE_dB = 10*np.log10(NMSE)
    return NMSE, NMSE_dB

def computeMSCD(Hnorm, Hhat):
    """
    Compute mean squared chordal distance between Hnorm and Hhat.
    """
    Nr, Nt, n = Hnorm.shape
    user_mean_chordal_distance = np.zeros(Nr)
    
    for user_id in range(Nr):
        user_chordal_distances = []
        
        for t in range(n):
            h1 = Hnorm[user_id,:,t]
            h2 = Hhat[user_id,:,t]
            rho = 1 - ((np.abs(np.vdot(h1,h2))**2) / ((np.linalg.norm(h1)**2)*(np.linalg.norm(h2)**2)))
            user_chordal_distances.append(rho)
            
        user_mean_chordal_distance[user_id] = np.mean(user_chordal_distances)
        
    MSCD    = np.mean(user_mean_chordal_distance)
    MSCD_dB = 10*np.log10(MSCD)

    return MSCD, MSCD_dB

### Auxiliary functions
def partialIndicesToMatrix(qts_idx, idx_other, n, Nt):
    qts_idx_lin = np.zeros(Nt*n)
    qts_idx_lin[idx_other] = qts_idx
    return np.reshape(qts_idx_lin, (n,Nt)).T

def partialHhatToMatrix(hhat, idx_other, n, Nt):
    hhat_lin = np.ones(Nt*n, dtype=complex)
    hhat_lin[idx_other] = hhat
    return np.reshape(hhat_lin, (n, Nt)).T