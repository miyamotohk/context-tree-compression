################################################
# Communication rate bounds
#
# From G. Caire et al. "Multiuser MIMO Achievable Rates With Downlink Training and Channel State Feedback,"
# IEEE Trans. Inf. Theory, vol. 56, no. 6, pp. 2845-2866, June 2010
################################################

import numpy as np
import scipy as sp
import scipy.special as sc

# Find communication rates
def communication_rate(Hquant, Hobs_testing, H_testing, SNR, Nt, Nr, b2=1e10):
    """
    Compute communication rates.
    Inputs:
        Hquant:       quantised CSI matrix
        Hobs_testing: observed (noisy) CSI matrix
        H_testing:    original CSI matrix
        SNR:          signal-to-noise ratio (dB)
        Nt:           number of transmitters
        Nr:           number of receivers
        b2:           number of orthogonal training symbols transmitted by BS in dedicated training
    Outputs:
        qtz_CSI_L: quantised feedback sum rate
        anl_CSI:   analog feedback sum rate
        perf_CSI:  zero-forcing sum rate
    """
    [qtz_CSI_L, anl_CSI, perf_CSI, _, _, _] = rateBounds(Hquant, Hobs_testing, H_testing, SNR, Nt, Nr, b2)
    return [qtz_CSI_L, anl_CSI, perf_CSI]

def computeV(H):
    """
    Compute linear beamforming matrix.
    Inputs:
        H: channel matrix
    Outputs:
        V: linear beamforming matrix
    """
    # Pseudo-inverse of H
    #V = np.linalg.inv(H)   # Inverse
    V = np.linalg.pinv(H)  # Pseudo-inverse
    
    # Normalise columns
    [m,n] = V.shape
    for i in range(n):
        V[:,i] = V[:,i]/np.linalg.norm(V[:,i])
    
    return V

def Rzf(M, SNR):
    """
    Compute ZF rate.
    Inputs:
        M:   number of BS antennas
        SNR: signal-to-noise ratio
    Output:
        ZF rate
    """
    R = np.exp(M/SNR) * sp.special.exp1(M/SNR)
    R = R/np.log(2)
    return M*R

def rateBounds(Hquant, Hanl, H, SNR, Nt, Nr, b2=1):
    """
    Compute rate bounds: upper bound for quantised and unquantised channel vectors and zero-forcing.
    Inputs:
        Hquant: channel matrix corresponding to quantised CSI
        Hanl:   analog (observed, noisy) channel matrix
        H:      original channel matrix
        SNR:    signal-to-noise ratio
        Nt:     number of transmitters
        Nr:     number of receivers
        b2:     number of orthogonal training symbols transmitted by BS in dedicated training
    Outputs:
        sum_ub:        upper bound for sum rate (quantised feedback)
        sum_ubobs:     upper bound for sum rate (analog feedback)
        sum_zf:        zero-forcing sum rate
        UB_sum_vec:    vector of upper bound sum rates (quantised feedback)
        UBobs_sum_vec: vector of upper bound sum rates (analog feedback)
        ZF_sum_vec:    vector of zero-forcing sum rates
    """

    # Initialisations
    UB    = [[] for i in range(Nr)]
    UBobs = [[] for i in range(Nr)]
    ZF    = [[] for i in range(Nr)]
    
    # For each frame
    for i in range(np.shape(Hquant)[2]):
        # Quantised channel matrix
        Hhat  = np.asmatrix(np.transpose(Hquant[0:Nr,0:Nt,i]))
        HhatH = Hhat.getH()        
        Vhat  = computeV(HhatH)
 
        # Observed (noisy) channel matrix
        Hobs  = np.asmatrix(np.transpose(Hanl[0:Nr,0:Nt,i]))
        HobsH = Hobs.getH()        
        Vobs  = computeV(HobsH)
        
        # Original channel matrix
        Htrue  = np.asmatrix(np.transpose(H[0:Nr,0:Nt,i]))
        HtrueH = Htrue.getH()
        Vtrue  = computeV(HtrueH)
        
        a     = HtrueH @ Vhat
        aobs  = HtrueH @ Vobs
        atrue = HtrueH @ Vtrue

        for k in range(Nr):
            sum1    = 0
            sum1obs = 0

            for j in range(Nr):
                if j != k: 
                    sum1    += np.abs(a[k,j])**2
                    sum1obs += np.abs(aobs[k,j])**2
                    
            UB[k].append( np.log2( 1 + ( (SNR/Nr * np.abs(a[k,k])**2)/(1 + sum1*SNR/Nr) ) ) )
            UBobs[k].append( np.log2( 1 + ( (SNR/Nr * np.abs(aobs[k,k])**2)/(1 + sum1obs*SNR/Nr) ) ) )
            ZF[k].append( np.log2( 1 + ( (SNR/Nr * np.abs(atrue[k,k]**2)) ) ) )
        
    UB_sum_vec    = np.sum(UB, axis=0)
    UBobs_sum_vec = np.sum(UBobs, axis=0)
    ZF_sum_vec    = np.sum(ZF, axis=0)
 
    sum_zf    = np.mean(ZF_sum_vec)
    sum_ub    = np.mean(UB_sum_vec)
    sum_ubobs = np.mean(UBobs_sum_vec)
    
    return [sum_ub, sum_ubobs, sum_zf, UB_sum_vec, UBobs_sum_vec, ZF_sum_vec]