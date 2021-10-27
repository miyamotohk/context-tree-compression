################################################
# Quantisation functions
################################################

import numpy as np

from quantisation_aux import *

def applyQuantisationMultilevel(H, m_all_H, m_all_L, qts_flags=[1,1,1,1]):
    """
    Apply multi-level quantisation for a CSI matrix.
    Inputs:
        H: original CSI matrix
        m_all_H = [m_abs_H, m_ang_H, m_CS_H]: high-resolution alphabet sizes
        m_all_L = [m_abs_L, m_ang_L, m_CS_L]: low-resolution alphabet sizes
        qts_flags = [unif_flag, mu_flag, beta_flag, CS_flag]: flags indicating if quantiser should be run
    """

    Nr, Nt, n = H.shape

    [m_abs_H, m_ang_H, m_CS_H] = m_all_H
    [m_abs_L, m_ang_L, m_CS_L] = m_all_L

    [unif_flag, mu_flag, beta_flag, CS_flag] = qts_flags

    n_comp = 3 # Number of companders (unif, mu, beta)
    n_res  = 3 # Number of resolutions (H, L, L2H)

    # Optimisation parameters
    mu0       = 1e-4
    Nmax      = 200
    step      = 0.2
    mu_min    = 1e-5
    theta0    = [1,1]
    theta_min = 1e-5

    # Normalised vectors
    Hnorm = np.zeros(H.shape, dtype=complex)

    # Quantised vectors
    Hhat_pol_unif_H    = np.zeros(H.shape, dtype=complex)
    Hhat_pol_mu_H      = np.zeros(H.shape, dtype=complex)
    Hhat_pol_beta_H    = np.zeros(H.shape, dtype=complex)

    Hhat_rect_unif_H   = np.zeros(H.shape, dtype=complex)
    Hhat_rect_mu_H     = np.zeros(H.shape, dtype=complex)
    Hhat_rect_beta_H   = np.zeros(H.shape, dtype=complex)

    Hhat_pol_unif_L    = np.zeros(H.shape, dtype=complex)
    Hhat_pol_mu_L      = np.zeros(H.shape, dtype=complex)
    Hhat_pol_beta_L    = np.zeros(H.shape, dtype=complex)

    Hhat_rect_unif_L   = np.zeros(H.shape, dtype=complex)
    Hhat_rect_mu_L     = np.zeros(H.shape, dtype=complex)
    Hhat_rect_beta_L   = np.zeros(H.shape, dtype=complex)

    Hhat_pol_unif_L2H  = np.zeros(H.shape, dtype=complex)
    Hhat_pol_mu_L2H    = np.zeros(H.shape, dtype=complex)
    Hhat_pol_beta_L2H  = np.zeros(H.shape, dtype=complex)

    Hhat_rect_unif_L2H = np.zeros(H.shape, dtype=complex)
    Hhat_rect_mu_L2H   = np.zeros(H.shape, dtype=complex)
    Hhat_rect_beta_L2H = np.zeros(H.shape, dtype=complex)

    Hhat_CS_H          = np.zeros(H.shape, dtype=complex)
    Hhat_CS_L          = np.zeros(H.shape, dtype=complex)
    Hhat_CS_L2H        = np.zeros(H.shape, dtype=complex)

    # Quantisation indices
    Idx_abs_unif_H   = np.zeros(H.shape, dtype=int)
    Idx_ang_unif_H   = np.zeros(H.shape, dtype=int)
    Idx_abs_unif_L   = np.zeros(H.shape, dtype=int)
    Idx_ang_unif_L   = np.zeros(H.shape, dtype=int)
    Idx_abs_unif_L2H = np.zeros(H.shape, dtype=int)
    Idx_ang_unif_L2H = np.zeros(H.shape, dtype=int)

    Idx_abs_mu_H     = np.zeros(H.shape, dtype=int)
    Idx_ang_mu_H     = np.zeros(H.shape, dtype=int)
    Idx_abs_mu_L     = np.zeros(H.shape, dtype=int)
    Idx_ang_mu_L     = np.zeros(H.shape, dtype=int)
    Idx_abs_mu_L2H   = np.zeros(H.shape, dtype=int)
    Idx_ang_mu_L2H   = np.zeros(H.shape, dtype=int)

    Idx_abs_beta_H   = np.zeros(H.shape, dtype=int)
    Idx_ang_beta_H   = np.zeros(H.shape, dtype=int)
    Idx_abs_beta_L   = np.zeros(H.shape, dtype=int)
    Idx_ang_beta_L   = np.zeros(H.shape, dtype=int)
    Idx_abs_beta_L2H = np.zeros(H.shape, dtype=int)
    Idx_ang_beta_L2H = np.zeros(H.shape, dtype=int)

    Idx_CS_re_H      = np.zeros(H.shape, dtype=int)
    Idx_CS_im_H      = np.zeros(H.shape, dtype=int)
    Idx_CS_re_L      = np.zeros(H.shape, dtype=int)
    Idx_CS_im_L      = np.zeros(H.shape, dtype=int)
    Idx_CS_re_L2H    = np.zeros(H.shape, dtype=int)
    Idx_CS_im_L2H    = np.zeros(H.shape, dtype=int)

    for user_id in range(Nr):

        ## Normalise channel vectors
        H_i = H[user_id] # (Nt x n)

        # Normalise coefficients by strongest one and isolate other entries
        Nt, n       = H_i.shape 
        Hnorm1_i    = np.zeros((Nt,n),   dtype=complex)
        Hnorm_other = np.zeros((Nt-1,n), dtype=complex)
        amax        = np.argmax(np.abs(H_i), axis=0)
        # Linearise indices
        amax_pos    = np.arange(len(amax))
        amax_lin    = amax_pos * Nt + amax
        idx_other   = np.setdiff1d(np.arange(Nt*n), amax_lin)

        for t in range(n):
            Hnorm1_i[:,t]    = H_i[:,t]/H_i[amax[t],t]           # Normalise by strongest coefficient
            Hnorm_other[:,t] = np.delete(Hnorm1_i[:,t], amax[t]) # Remove strongest coefficient

        Hnorm[user_id,:,:] = Hnorm1_i
        Hlin = np.reshape(Hnorm_other.T, -1)

        ## Decompose coefficients
        h_abs, h_ang = decompose_polar(Hlin)

        h_abs_n = 1 - h_abs
        h_ang_n = np.abs(h_ang/np.pi)

        h_abs_n = np.clip(h_abs_n, 0, 1)
        h_ang_n = np.clip(h_ang_n, 0, 1)

        n_lin = len(Hlin)

        idx_abs = np.zeros((n_comp, n_res, n_lin), dtype=int)
        idx_ang = np.zeros((n_comp, n_res, n_lin), dtype=int)
        hhat    = np.zeros((n_comp, n_res, n_lin), dtype=complex)

        ## Quantisation
        # Uniform quantisation
        if unif_flag:
            # Quantisation indices
            idx_abs_unif_H, hhat_abs_unif_H     = quantise_uniform(h_abs_n, m_abs_H)
            idx_ang_unif_H, hhat_ang_unif_H     = quantise_uniform(h_ang_n, m_ang_H)
            idx_abs_unif_L, hhat_abs_unif_L     = quantise_uniform(h_abs_n, m_abs_L)
            idx_ang_unif_L, hhat_ang_unif_L     = quantise_uniform(h_ang_n, m_ang_L)
            idx_abs_unif_L2H, hhat_abs_unif_L2H = quantise_uniform(hhat_abs_unif_L, m_abs_H)
            idx_ang_unif_L2H, hhat_ang_unif_L2H = quantise_uniform(hhat_ang_unif_L, m_ang_H)

            Idx_abs_unif_H[user_id,:,:]   = partialIndicesToMatrix(idx_abs_unif_H+1, idx_other, n, Nt)
            Idx_ang_unif_H[user_id,:,:]   = partialIndicesToMatrix(idx_ang_unif_H+1, idx_other, n, Nt)
            Idx_abs_unif_L[user_id,:,:]   = partialIndicesToMatrix(idx_abs_unif_L+1, idx_other, n, Nt)
            Idx_ang_unif_L[user_id,:,:]   = partialIndicesToMatrix(idx_ang_unif_L+1, idx_other, n, Nt)
            Idx_abs_unif_L2H[user_id,:,:] = partialIndicesToMatrix(idx_abs_unif_L2H+1, idx_other, n, Nt)
            Idx_ang_unif_L2H[user_id,:,:] = partialIndicesToMatrix(idx_ang_unif_L2H+1, idx_other, n, Nt)

            # Representation points
            hhat_pol_unif_H   = (1-hhat_abs_unif_H)  *np.exp(1j*hhat_ang_unif_H*np.pi*np.sign(h_ang))
            hhat_pol_unif_L   = (1-hhat_abs_unif_L)  *np.exp(1j*hhat_ang_unif_L*np.pi*np.sign(h_ang))
            hhat_pol_unif_L2H = (1-hhat_abs_unif_L2H)*np.exp(1j*hhat_ang_unif_L2H*np.pi*np.sign(h_ang))

            Hhat_pol_unif_H[user_id,:,:]   = partialHhatToMatrix(hhat_pol_unif_H, idx_other, n, Nt)
            Hhat_pol_unif_L[user_id,:,:]   = partialHhatToMatrix(hhat_pol_unif_L, idx_other, n, Nt)
            Hhat_pol_unif_L2H[user_id,:,:] = partialHhatToMatrix(hhat_pol_unif_L2H, idx_other, n, Nt)

        # Mu-law quantisation
        if mu_flag:
            # Quantisation indices
            idx_abs_mu_H, hhat_abs_mu_H     = quantise_mu(h_abs_n, m_abs_H, mu0, Nmax, step, mu_min)
            idx_ang_mu_H, hhat_ang_mu_H     = quantise_mu(h_ang_n, m_ang_H, mu0, Nmax, step, mu_min)
            idx_abs_mu_L, hhat_abs_mu_L     = quantise_mu(h_abs_n, m_abs_L, mu0, Nmax, step, mu_min)
            idx_ang_mu_L, hhat_ang_mu_L     = quantise_mu(h_ang_n, m_ang_L, mu0, Nmax, step, mu_min)
            idx_abs_mu_L2H, hhat_abs_mu_L2H = quantise_mu(hhat_abs_mu_L, m_abs_H, mu0, Nmax, step, mu_min)
            idx_ang_mu_L2H, hhat_ang_mu_L2H = quantise_mu(hhat_ang_mu_L, m_ang_H, mu0, Nmax, step, mu_min)

            Idx_abs_mu_H[user_id,:,:]   = partialIndicesToMatrix(idx_abs_mu_H+1, idx_other, n, Nt)
            Idx_ang_mu_H[user_id,:,:]   = partialIndicesToMatrix(idx_ang_mu_H+1, idx_other, n, Nt)
            Idx_abs_mu_L[user_id,:,:]   = partialIndicesToMatrix(idx_abs_mu_L+1, idx_other, n, Nt)
            Idx_ang_mu_L[user_id,:,:]   = partialIndicesToMatrix(idx_ang_mu_L+1, idx_other, n, Nt)
            Idx_abs_mu_L2H[user_id,:,:] = partialIndicesToMatrix(idx_abs_mu_L2H+1, idx_other, n, Nt)
            Idx_ang_mu_L2H[user_id,:,:] = partialIndicesToMatrix(idx_ang_mu_L2H+1, idx_other, n, Nt)

            # Representation points
            hhat_pol_mu_H   = (1-hhat_abs_mu_H)  *np.exp(1j*hhat_ang_mu_H*np.pi*np.sign(h_ang))
            hhat_pol_mu_L   = (1-hhat_abs_mu_L)  *np.exp(1j*hhat_ang_mu_L*np.pi*np.sign(h_ang))
            hhat_pol_mu_L2H = (1-hhat_abs_mu_L2H)*np.exp(1j*hhat_ang_mu_L2H*np.pi*np.sign(h_ang))

            Hhat_pol_mu_H[user_id,:,:]   = partialHhatToMatrix(hhat_pol_mu_H, idx_other, n, Nt)
            Hhat_pol_mu_L[user_id,:,:]   = partialHhatToMatrix(hhat_pol_mu_L, idx_other, n, Nt)
            Hhat_pol_mu_L2H[user_id,:,:] = partialHhatToMatrix(hhat_pol_mu_L2H, idx_other, n, Nt)

        # Beta-law quantisation
        if beta_flag:
            # Quantisation indices
            idx_abs_beta_H, hhat_abs_beta_H     = quantise_beta(h_abs_n, m_abs_H, theta0, Nmax, step, theta_min)
            idx_ang_beta_H, hhat_ang_beta_H     = quantise_beta(h_ang_n, m_ang_H, theta0, Nmax, step, theta_min)
            idx_abs_beta_L, hhat_abs_beta_L     = quantise_beta(h_abs_n, m_abs_L, theta0, Nmax, step, theta_min)
            idx_ang_beta_L, hhat_ang_beta_L     = quantise_beta(h_ang_n, m_ang_L, theta0, Nmax, step, theta_min)
            idx_abs_beta_L2H, hhat_abs_beta_L2H = quantise_beta(hhat_abs_beta_L, m_abs_H, theta0, Nmax, step, theta_min)
            idx_ang_beta_L2H, hhat_ang_beta_L2H = quantise_beta(hhat_ang_beta_L, m_ang_H, theta0, Nmax, step, theta_min)

            Idx_abs_beta_H[user_id,:,:]   = partialIndicesToMatrix(idx_abs_beta_H+1, idx_other, n, Nt)
            Idx_ang_beta_H[user_id,:,:]   = partialIndicesToMatrix(idx_ang_beta_H+1, idx_other, n, Nt)
            Idx_abs_beta_L[user_id,:,:]   = partialIndicesToMatrix(idx_abs_beta_L+1, idx_other, n, Nt)
            Idx_ang_beta_L[user_id,:,:]   = partialIndicesToMatrix(idx_ang_beta_L+1, idx_other, n, Nt)
            Idx_abs_beta_L2H[user_id,:,:] = partialIndicesToMatrix(idx_abs_beta_L2H+1, idx_other, n, Nt)
            Idx_ang_beta_L2H[user_id,:,:] = partialIndicesToMatrix(idx_ang_beta_L2H+1, idx_other, n, Nt)

            # Representation points
            hhat_pol_beta_H   = (1-hhat_abs_beta_H)  *np.exp(1j*hhat_ang_beta_H*np.pi*np.sign(h_ang))
            hhat_pol_beta_L   = (1-hhat_abs_beta_L)  *np.exp(1j*hhat_ang_beta_L*np.pi*np.sign(h_ang))
            hhat_pol_beta_L2H = (1-hhat_abs_beta_L2H)*np.exp(1j*hhat_ang_beta_L2H*np.pi*np.sign(h_ang))

            Hhat_pol_beta_H[user_id,:,:]   = partialHhatToMatrix(hhat_pol_beta_H, idx_other, n, Nt)
            Hhat_pol_beta_L[user_id,:,:]   = partialHhatToMatrix(hhat_pol_beta_L, idx_other, n, Nt)
            Hhat_pol_beta_L2H[user_id,:,:] = partialHhatToMatrix(hhat_pol_beta_L2H, idx_other, n, Nt)

        # Cube-split
        if CS_flag:
            idx_CS_re_H,   idx_CS_im_H,   hhat_CS_H   = quantise_CS(Hlin, m_CS_H)
            idx_CS_re_L,   idx_CS_im_L,   hhat_CS_L   = quantise_CS(Hlin, m_CS_L)
            idx_CS_re_L2H, idx_CS_im_L2H, hhat_CS_L2H = quantise_CS(hhat_CS_L, m_CS_H)

            Idx_CS_re_H[user_id,:,:]   = partialIndicesToMatrix(idx_CS_re_H+1, idx_other, n, Nt)
            Idx_CS_im_H[user_id,:,:]   = partialIndicesToMatrix(idx_CS_im_H+1, idx_other, n, Nt)
            Hhat_CS_H[user_id,:,:]     = partialHhatToMatrix(hhat_CS_H, idx_other, n, Nt)

            Idx_CS_re_L[user_id,:,:]   = partialIndicesToMatrix(idx_CS_re_L+1, idx_other, n, Nt)
            Idx_CS_im_L[user_id,:,:]   = partialIndicesToMatrix(idx_CS_im_L+1, idx_other, n, Nt)
            Hhat_CS_L[user_id,:,:]     = partialHhatToMatrix(hhat_CS_L, idx_other, n, Nt)

            Idx_CS_re_L2H[user_id,:,:] = partialIndicesToMatrix(idx_CS_re_L2H+1, idx_other, n, Nt)
            Idx_CS_im_L2H[user_id,:,:] = partialIndicesToMatrix(idx_CS_im_L2H+1, idx_other, n, Nt)
            Hhat_CS_L2H[user_id,:,:]   = partialHhatToMatrix(hhat_CS_L2H, idx_other, n, Nt)

    # Alphabet sizes (symbol 0 indicates strongest coefficient)
    [M_abs_H, M_abs_L] = [m_abs_H+1, m_abs_L+1]
    [M_ang_H, M_ang_L] = [m_ang_H+1, m_ang_L+1]

    M_pol_H = 2*m_abs_H*m_ang_H + 1
    M_pol_L = 2*m_abs_L*m_ang_L + 1

    [M_CS_H, M_CS_L] = [m_CS_H+1, m_CS_L+1]
    M_CS_tot_H = m_CS_H**2 + 1
    M_CS_tot_L = m_CS_L**2 + 1

    M_all = [M_abs_H, M_abs_L, M_ang_H, M_ang_L, M_CS_H, M_CS_L,\
             M_pol_H, M_pol_L, M_CS_tot_H, M_CS_tot_L]

    # Represent everything in arrays
    # compander x resolution x user x antenna x time
    Hhat_pol_all = [[Hhat_pol_unif_H, Hhat_pol_unif_L, Hhat_pol_unif_L2H],\
                    [Hhat_pol_mu_H, Hhat_pol_mu_L, Hhat_pol_mu_L2H],\
                    [Hhat_pol_beta_H, Hhat_pol_beta_L, Hhat_pol_beta_L2H]
                    ]

    Idx_abs_all = [[Idx_abs_unif_H, Idx_abs_unif_L, Idx_abs_unif_L2H],\
                   [Idx_abs_mu_H,   Idx_abs_mu_L,   Idx_abs_mu_L2H],\
                   [Idx_abs_beta_H, Idx_abs_beta_L, Idx_abs_beta_L2H]
                  ]

    Idx_ang_all = [[Idx_ang_unif_H, Idx_ang_unif_L, Idx_ang_unif_L2H],\
                   [Idx_ang_mu_H,   Idx_ang_mu_L,   Idx_ang_mu_L2H],\
                   [Idx_ang_beta_H, Idx_ang_beta_L, Idx_ang_beta_L2H]
                  ]

    # resolution x user x antenna x time
    Hhat_CS_all   = [Hhat_CS_H, Hhat_CS_L, Hhat_CS_L2H]
    Idx_CS_re_all = [Idx_CS_re_H, Idx_CS_re_L, Idx_CS_re_L2H]
    Idx_CS_im_all = [Idx_CS_im_H, Idx_CS_im_L, Idx_CS_im_L2H]

    # To array
    Hhat_pol_all  = np.array(Hhat_pol_all)
    Hhat_CS_all   = np.array(Hhat_CS_all)
    Idx_abs_all   = np.array(Idx_abs_all)
    Idx_ang_all   = np.array(Idx_ang_all)
    Idx_CS_re_all = np.array(Idx_CS_re_all)
    Idx_CS_im_all = np.array(Idx_CS_im_all)
    
    return Hnorm, Hhat_pol_all, Hhat_CS_all, Idx_abs_all, Idx_ang_all, Idx_CS_re_all, Idx_CS_im_all, M_all