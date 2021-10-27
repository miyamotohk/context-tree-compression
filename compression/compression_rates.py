################################################
# Compression rate functions
################################################

import numpy as np
import scipy as sp
import scipy.special as sc
import cmath
import sys

from scipy import io

sys.path.append('./../ct')

from ct_main        import *
from compression    import *

def import_channel_matrix(LTE_mobility, LTE_MIMOcorr, SNR_dB=30, Nr=4, n=10000):
    """
    Import channel matrix from MATLAB files and add Gaussian noise.
    Inputs:
        LTE_mobility: 'EPA5', 'EVA30', 'EVA70'
        LTE_MIMOcorr: 'Low', 'Medium', 'High'
        SNR_dB:       signal-to-noise ratio in dB
        Nr:           number of receivers
        n:            number of samples
    Outputs:
        H:    original CSI matrix
        Hobs: observed (noisy) CSI matrix
    """
    
    SNR = 10**(SNR_dB/10)

    H    = []
    Hobs = []
    
    for user_id in range(1, Nr+1):
        
        # Import from MATLAB file
        filename = '{}_MIMOcorr{}_user{}.mat'.format(LTE_mobility, LTE_MIMOcorr, user_id)
        mat = io.loadmat('./../LTE-channel/'+filename)
        H_i = mat['H'].T
        H.append(H_i)
        
        # Add Gaussian noise
        r,c = H_i.shape
        S   = np.linalg.norm(H_i)**2/(r*c)
        NoiseVar = S / SNR
        Hobs_i   = H_i + (np.random.randn(r,c) + 1j*np.random.randn(r,c))*np.sqrt(NoiseVar/2)
        Hobs.append(Hobs_i) 
        
    # H and Hobs are arrays with dimension (Nr X Nt X n)
    H = np.array(H)[:,:,:n]
    Hobs = np.array(Hobs)[:,:,:n]
    
    return H, Hobs

def compressionRate(sequence_H, sequence_L, sequence_L2H, m_H, m_L, D, q, train_size=0.5, beta=0.5,\
                    alpha=2, nupdate=100, adaptiveFlag=False, ntimes=10, trainingFlag=True,\
                    drawTree=False, filename='tree', show_probs=False):
    """
    Compute compression rates of different methods.
    Inputs:
        sequence_H:      high-resolution sequence
        sequence_L:      low-resolution sequence
        sequence_L2H:    projected low-to-high resolutio sequence
        m_H:             high-resolution alphabet size
        m_L:             low-resolution alphabet size
        D:               maximum tree depth
        q:               number of bits (minus one) used in levels 0 and 1
        train_size:      ratio of the total sequence used for training (between 0 and 1)
        beta:            weighting coefficient
        alpha:           Dirichlet coefficient
        nupdate:         interval between tree updates
        adaptiveFlag:    indicates if adaptive model should be used
        ntimes:          defines the number of symbols used for adapting the model as nupdate*ntimes
        drawTree:        flag indicating to plot the final tree model
        filename:        filename for tree model plot (pdf)
        show_probs:      flag indicating to write probabilities in the model plot
    Outputs:
        R_ctw:           compression rate of CTW
        R_ctm_sbs:       compression rate of CTM symbol by symbol
        R_prop:          compression rate of proposed method
        Levels_training: levels used in training sequence compression
        Levels_testing:  levels used in testing sequence compression
    """
    
    # Split sequences into training and testing
    len_all      = len(sequence_H)
    
    if trainingFlag:
        len_training = int(np.round(train_size * len_all))
        len_testing  = len_all - len_training

        training_H   = sequence_H[0:len_training]
        training_L   = sequence_L[0:len_training]
        training_L2H = sequence_L2H[0:len_training]

        testing_H    = sequence_H[len_training:len_all]
        testing_L    = sequence_L[len_training:len_all]
        testing_L2H  = sequence_L2H[len_training:len_all]
    else:
        training_H = []
        training_L = []
        training_L2H = []

        testing_H   = sequence_H
        testing_L   = sequence_L
        testing_L2H = sequence_L2H

    m_H2L = m_L
    
    # CTW compression
    assert np.min(testing_H)>=0 and np.max(testing_H)<m_H, "testing_H should be in [0,{}], but is in [{},{}]".format(m_H-1,np.min(testing_H),np.max(testing_H))
    [L_ctw, R_ctw] = ctw(testing_H, m_H, D, beta=beta, alpha=alpha)

    # CTM symbol by symbol
    [L_sbs, R_sbs] = ctm_sbs(training_H, testing_H, m_H, D, beta=beta, alpha=alpha)

    # Proposed method
    L_prop, R_prop, Levels_training, Levels_testing = compressMultiRes(training_L, training_H, training_L2H, \
                                                                       testing_L, testing_H, testing_L2H, \
                                                                       m_L, m_H, D, q, beta, alpha, nupdate=100,\
                                                                       adaptiveFlag=adaptiveFlag, ntimes=ntimes,\
                                                                       trainingFlag=trainingFlag, drawFlag=drawTree,\
                                                                       filename=filename, show_probs=show_probs)

    return R_ctw, R_sbs, R_prop, Levels_training, Levels_testing

def computeMatrixMixed(Flag1, Flag2, Hquant_H, Hquant_L, seq_length):
    """
    Compute mixed-resolution matrix.
    Inputs:
        Flag1:      matrix of first component (abs) level flags
        Flag2:      matrix of second component (ang) level flags
        Hquant_H:   high-resolution quantised matrix
        Hquant_L:   low-resolution quantised matrix
        seq_length: sequence length
    Output:
        Hquant_mixed: mixed-resolution quantisation matrix
    """    
    Bad  = ((Flag1==2) + (Flag2==2))*1
    Good = 1-Bad    
    Hquant_mixed = Hquant_H*Good + Hquant_L*Bad
    
    return Hquant_mixed

def computeJointRate(Flag1_test, Flag2_test, Flag1_train, Flag2_train, m_L1, m_L2, q1, q2, Nt=4, Nr=4, D=2,\
                     adaptiveFlag=False, ntimes=10, trainingFlag=True):
    """
    Compute joint rate for varying CSI indices.
    Inputs:
        Flag1_test:   level flag for testing sequence, absolute value
        Flag2_test:   level flag for testing sequence, angle
        Flag1_train : level flag for training sequence, absolute value
        Flag2_train:  level flag for training sequence, angle
        m_L1:         low-resolution alphabet size for absolute value
        m_L2:         low-resolution alphabet size for angle
        q1:           number of bits (minus one) for 0 and 1 levels, absolute value
        q2:           number of bits (minus one) for 0 and 1 levels, angle
        Nt:           number of transmitters
        Nr:           number of receivers
        D:            tree maximum depth
        adaptiveFlag: indicates if adaptive model should be used
        ntimes:       defines the number of symbols used for adapting the model as nupdate*ntimes

    Outputs:
        R_joint_prp:  joint rate using proposed (multi-level) compression
        R_joint_sbs:  joint rate using CTM symbol-by-symbol
        R_joint_smp:  joint rate using simple scheme
        Flags_all:    leve flags for proposed compression
    """

    assert q1[0]==0 and q2[0]==0, "We should have q1[0]==0 and q2[0]==0"
    # Alphabet size for state indicator
    m_state = int(2**Nt)

    if trainingFlag==False:
        Flag1_train = []
        Flag2_train = []

    # Number of levels
    nl1 = 1+np.sum(np.array(q1)>=0)
    nl2 = 1+np.sum(np.array(q2)>=0)

    # Identify the indices that are varying
    C_test   = np.logical_or(Flag1_test, Flag2_test)*1
    C_train  = np.logical_or(Flag1_train, Flag2_train)*1

    # The rate of the joint indicator part
    D_all     = []
    R_prp     = []
    R_sbs     = []
    R_smp     = []
    Flags_all = []

    for user_id in range(Nr):
        # Convert state from binary to decimal
        Cusr_test  = C_test[user_id,:,:].T
        n_test, _  = Cusr_test.shape
        Dusr_test  = np.zeros(n_test)
        for i in range(Nt):
            Dusr_test  += (2**i)*Cusr_test[:,i]
        Dusr_test  = Dusr_test.astype(int).tolist()
        D_all.append(Dusr_test)

        if trainingFlag:
            Cusr_train = C_train[user_id,:,:].T
            n_train, _ = Cusr_train.shape
            Dusr_train = np.zeros(n_train)        
            for i in range(Nt):
                Dusr_train += (2**i)*Cusr_train[:,i]
            Dusr_train = (Dusr_train.astype(int)).tolist()
        else:
            Dusr_train = []

        ## Compress sequence
        # Simple scheme
        R_smp_st = (Dusr_test==0)*1 + (Dusr_test!=0)*(1+Nt)
        R_smp.append(R_smp_st)

        # Proposed scheme
        [L_prp_st, R_prp_st, Flag_st] = compressSingleRes(Dusr_train, Dusr_test, m_state, D, \
                                                          adaptiveFlag=adaptiveFlag, ntimes=ntimes,\
                                                          trainingFlag=trainingFlag)
        R_prp.append(R_prp_st)
        Flags_all.append(Flag_st)

        # CTM symbol by symbol
        [L_sbs_st, R_sbs_st] = ctm_sbs(Dusr_train, Dusr_test, m_state, D)
        R_sbs.append(R_sbs_st)

    D_all = np.array(D_all) # Nt x n
    R_prp = np.array(R_prp)
    R_sbs = np.array(R_sbs)
    R_smp = np.array(R_smp)

    R_ind_prp = R_prp.mean()/Nt
    R_ind_sbs = R_sbs.mean()/Nt
    R_ind_smp = R_smp.mean()/Nt

    # The rate of the change description part

    # The number of bits needed to describe the varying CSI
    E1 = np.ceil((Flag1_test==0)*(1+q1[1]) + (Flag1_test==1)*(1+q1[1]) + (Flag1_test==2)*(1+np.log2(m_L1)))
    E2 = np.ceil((Flag2_test==0)*(1+q2[1]) + (Flag2_test==1)*(1+q2[1]) + (Flag2_test==2)*(1+np.log2(m_L2)))
    
    E     = E1+E2
    R_chg = (C_test*E).mean()

    # Overall rate
    R_joint_prp = R_ind_prp + R_chg
    R_joint_sbs = R_ind_sbs + R_chg
    R_joint_smp = R_ind_smp + R_chg
        
    return R_joint_prp, R_joint_sbs, R_joint_smp, Flags_all

def computeJointRate2(Flag1_test, Flag2_test, Flag1_train, Flag2_train, m_L1, m_L2, q1, q2, Nt=4, Nr=4, D=2,\
                     adaptiveFlag=False, ntimes=10, trainingFlag=True):
    """
    Compute joint rate for varying CSI indices.
    Inputs:
        Flag1_test:   level flag for testing sequence, absolute value
        Flag2_test:   level flag for testing sequence, angle
        Flag1_train : level flag for training sequence, absolute value
        Flag2_train:  level flag for training sequence, angle
        m_L1:         low-resolution alphabet size for absolute value
        m_L2:         low-resolution alphabet size for angle
        q1:           number of bits (minus one) for 0 and 1 levels, absolute value
        q2:           number of bits (minus one) for 0 and 1 levels, angle
        Nt:           number of transmitters
        Nr:           number of receivers
        D:            tree maximum depth
        adaptiveFlag: indicates if adaptive model should be used
        ntimes:       defines the number of symbols used for adapting the model as nupdate*ntimes

    Outputs:
        R_joint_prp:  joint rate using proposed (multi-level) compression
        R_joint_sbs:  joint rate using CTM symbol-by-symbol
        R_joint_smp:  joint rate using simple scheme
        Flags_all:    leve flags for proposed compression
    """

    assert q1[0]==0 and q2[0]==0, "We should have q1[0]==0 and q2[0]==0"
    m_state = int(2**(2*Nt))

    # Number of levels
    nl1 = 1+np.sum(np.array(q1)>=0)
    nl2 = 1+np.sum(np.array(q2)>=0)

    # Identify the indices that are varying
    C_test   = np.concatenate((Flag1_test!=0,  Flag2_test!=0),  axis=1)*1
    C_train  = np.concatenate((Flag1_train!=0, Flag2_train!=0), axis=1)*1

    # The rate of the joint indicator part
    D_all     = []
    R_prp     = []
    R_sbs     = []
    R_smp     = []
    Flags_all = []

    for user_id in range(Nr):
        # Convert state from binary to decimal
        Cusr_test  = C_test[user_id,:,:].T
        n_test, _  = Cusr_test.shape
        Dusr_test  = np.zeros(n_test)
        for i in range(2*Nt):
            Dusr_test  += (2**i)*Cusr_test[:,i]
        Dusr_test  = Dusr_test.astype(int).tolist()
        D_all.append(Dusr_test)

        if trainingFlag:
            Cusr_train = C_train[user_id,:,:].T
            n_train, _ = Cusr_train.shape
            Dusr_train = np.zeros(n_train)
            for i in range(2*Nt):
                Dusr_train += (2**i)*Cusr_train[:,i]
            Dusr_train = (Dusr_train.astype(int)).tolist()
        else:
            Dusr_train = []

        ## Compress sequence
        # Simple scheme
        R_smp_st = (Dusr_test==0)*1 + (Dusr_test!=0)*(1+2*Nt)
        R_smp.append(R_smp_st)

        # Proposed scheme
        #[L_prp_st, R_prp_st, Flag_st] = compressSingleRes(Dusr_train, Dusr_test, m_state, D, \
        #                                                  adaptiveFlag=adaptiveFlag, ntimes=ntimes,\
        #                                                  trainingFlag=trainingFlag)
        R_prp_st = 0
        Flag_st  = []
        R_prp.append(R_prp_st)
        Flags_all.append(Flag_st)

        # CTM symbol by symbol
        #[L_sbs_st, R_sbs_st] = ctm_sbs(Dusr_train, Dusr_test, m_state, D)
        R_sbs_st = 0
        R_sbs.append(R_sbs_st)
        
    D_all = np.array(D_all) # Nt x n
    R_prp = np.array(R_prp)
    R_sbs = np.array(R_sbs)
    R_smp = np.array(R_smp)

    R_ind_prp = R_prp.mean()/Nt
    R_ind_sbs = R_sbs.mean()/Nt
    R_ind_smp = R_smp.mean()/Nt

    # The rate of the change description part

    # The number of bits needed to describe the varying CSI
    E1 = np.ceil((Flag1_test==0)*(0) + (Flag1_test==1)*(1+q1[1]) + (Flag1_test==2)*(1+np.log2(m_L1)))
    E2 = np.ceil((Flag2_test==0)*(0) + (Flag2_test==1)*(1+q2[1]) + (Flag2_test==2)*(1+np.log2(m_L2)))
    
    R_chg = (E1+E2).mean()

    R_joint_prp = R_ind_prp + R_chg
    R_joint_sbs = R_ind_sbs + R_chg
    R_joint_smp = R_ind_smp + R_chg
        
    return R_joint_prp, R_joint_sbs, R_joint_smp, Flags_all