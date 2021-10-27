################################################
# Main test function
################################################

import numpy as np
import pickle
import cmath
import sys

from scipy import io
from scipy.special import polygamma
from tqdm import tqdm

sys.path.append('./../quantisation')
sys.path.append('./../compression')
sys.path.append('./../communication')

from quantisation_main import *
from compression_rates import *
from communication_rates import communication_rate

def evaluate(LTE_mobility, LTE_MIMOcorr, SNR_dB, mm_abs_L, mm_ang_L, mm_abs_H, mm_ang_H,\
             n=10000, Nt=4, Nr=4, D=2, train_size=0.2, beta=0.5, alpha=2, nupdate=100,\
             qts_flags=[1,1,1,1], save_flag=False, draw_flag_ext=False, ntimes=10, training=True):
    """
    Evaluate performance in terms of compression rante, communication rate and distortion rate.
    Inputs:
        LTE_mobility:  EPA5', 'EVA30', EVA70'
        LTE_MIMOcorr:  'Low', 'Medium', 'High'
        SNR_dB:        AWGN channel SNR in dB
        mm_abs_L:      number of quantisation levels for absolute value in low-resolution codebook
        mm_ang_L:      number of quantisation levels for angle in low-resolution codebook
        mm_abs_H:      number of quantisation levels for absolute value in high-resolution codebook
        mm_ang_H:      number of quantisation levels for angle in high-resolution codebook
        n:             sequence length
        Nt:            number of transmitter antennas
        Nr:            number of receivers
        D:             maximum tree depth
        train_size:    ratio training sequence length/total sequence length
        beta:          weighting coefficient
        alpha:         Dirichlet coefficient
        nupdate:       interval between tree updates
        qts_flags = [unif_flag, mu_flag, beta_flag, CS_flag]:
                       flags indicating if quantiser should be run
        save_flag:     flag indicating to save quantisation data
        draw_flag:     flag indicating to plot context trees
        ntimes:        defines the number of symbols used for adapting the model as nupdate*ntimes
        training:      indicates if training sequence should be used to initialise models (if False, train_size is ignored)

    Outputs:
        (...)
    """

    [unif_flag, mu_flag, beta_flag, CS_flag] = qts_flags   
    SNR = 10**(SNR_dB/10)
    adaptiveFlag = False

    compressor_name = ['unif', 'mu', 'beta']   
    tree_filename   = './trees/Tree_{}_corr{}_mH_{}_{}_mL_{}_{}_SNR{}dB'.format(LTE_mobility, LTE_MIMOcorr, mm_abs_H, mm_ang_H, mm_abs_L, mm_ang_L, SNR_dB)

    path = './trees'
    if draw_flag_ext==True and os.path.isdir(path)==False:
        os.mkdir(path)

    # Outputs
    # Uncompressed rates (quantisation rate)
    R_uncomp_all        = []
    # Compression rates
    R_ctw_all           = []
    R_sbs_all           = []
    R_prp_indiv_all     = []
    R_prp_joint_all     = []
    R_prp_joint_sbs_all = []
    R_prp_joint_smp_all = []
    
    # Communication rates
    comm_rate_H_all  = []
    comm_rate_M_all  = []
    
    # Distortion rates
    NMSE_dB_H_all  = []
    MSCD_dB_H_all  = []
    NMSE_dB_M_all  = []
    MSCD_dB_M_all  = []
    
    # Training and testing sizes
    seq_length     = n
    len_all        = seq_length
    if training:
        len_training = int(np.round(train_size * len_all))
        len_testing  = len_all - len_training
    else:
        len_training = 0
        len_testing  = len_all
        len_testing  = len_all - int(np.round(train_size * len_all))

    ###################################################
    ## Import channel matrices
    ###################################################
    H, Hobs = import_channel_matrix(LTE_mobility, LTE_MIMOcorr, n=n, Nr=Nr, SNR_dB=SNR_dB)

    if not training:
        H    = H[:,:,int(np.round(train_size * len_all)):]
        Hobs = Hobs[:,:,int(np.round(train_size * len_all)):]

    ###################################################
    # Quantisation
    ###################################################

    # Equivalent alphabet size for cube-split
    mm_CS_H = int(np.round(np.sqrt(mm_abs_H*mm_ang_H*2)))
    mm_CS_L = int(np.round(np.sqrt(mm_abs_L*mm_ang_L*2)))

    m_all_H = [mm_abs_H, mm_ang_H, mm_CS_H]
    m_all_L = [mm_abs_L, mm_ang_L, mm_CS_L]

    [Hnorm, Hhat_pol_all, Hhat_CS_all, Idx_abs_all, Idx_ang_all, Idx_CS_re_all, Idx_CS_im_all, M_all] \
        = applyQuantisationMultilevel(Hobs, m_all_H, m_all_L, qts_flags=qts_flags)
    [M_abs_H, M_abs_L, M_ang_H, M_ang_L, M_CS_H, M_CS_L, M_pol_H, M_pol_L, M_CS_tot_H, M_CS_tot_L] = M_all

    # Uncompressed rate (quantisation rate)
    [R_pol_uncomp, R_CS_uncomp] = np.ceil(np.log2([M_pol_H, M_CS_tot_H]))
    R_uncomp_all = [R_pol_uncomp, R_pol_uncomp, R_pol_uncomp, R_CS_uncomp]

    # Number of bits for levels 0 and 1 in proposed compression
    q_abs  = [0, max(0, int(np.ceil(np.log2(mm_abs_L)))-2)]
    q_ang  = [0, max(0, int(np.ceil(np.log2(mm_ang_L)))-2)]
    q_rect = [0, max(0, int(np.ceil(np.log2(mm_CS_L)))-2)]

    # Save quantisation results
    if save_flag:
        with open('./data/Quant_{}_corr{}_mH_{}_{}_mL_{}_{}_SNR{}dB.pkl'.format(LTE_mobility, LTE_MIMOcorr, mm_abs_H, mm_ang_H, mm_abs_L, mm_ang_L, SNR_dB), 'wb') as f:
            pickle.dump([Hnorm, Hhat_pol_all, Hhat_CS_all,\
                         Idx_abs_all, Idx_ang_all, Idx_CS_re_all, Idx_CS_im_all,\
                         M_all], f)

        filename = './data/Quant_{}_corr{}_mH_{}_{}_mL_{}_{}_SNR{}dB.pkl'.format(LTE_mobility, LTE_MIMOcorr, mm_abs_H, mm_ang_H, mm_abs_L, mm_ang_L, SNR_dB)
    
    ###################################################
    # Compression
    ###################################################
    
    ## Compander quantisers
    # Matrices with results for all quantisers: unif, mu, beta, CS
    Flags_abs_test_all  = []
    Flags_ang_test_all  = []
    Flags_abs_train_all = []
    Flags_ang_train_all = []
    
    for i_comp in range(3): # unif, mu, beta
        
        i_name = tree_filename + '_' +compressor_name[i_comp]

        R_ctw_pol_matrix = np.zeros((Nr, Nt))
        R_sbs_pol_matrix = np.zeros((Nr, Nt))
        R_prp_pol_matrix = np.zeros((Nr, Nt))

        if training:
            Flag_abs_train = np.zeros((Nr, Nt, len_training-D))
            Flag_ang_train = np.zeros((Nr, Nt, len_training-D))
        else:
            Flag_abs_train = []
            Flag_ang_train = []
        Flag_abs_test  = np.zeros((Nr, Nt, len_testing-D))
        Flag_ang_test  = np.zeros((Nr, Nt, len_testing-D))

        if qts_flags[i_comp]:

            for user_id in range(Nr):
                for antenna_id in range(Nt):
                    if user_id==0 and antenna_id==0 and draw_flag_ext==True:
                        draw_flag = True
                    else:
                        draw_flag = False

                    ii_name = i_name + '_usr{}_ant{}'.format(user_id, antenna_id)

                    idx_abs_H   = Idx_abs_all[i_comp,0,user_id,antenna_id,:n].tolist()
                    idx_abs_L   = Idx_abs_all[i_comp,1,user_id,antenna_id,:n].tolist()
                    idx_abs_L2H = Idx_abs_all[i_comp,2,user_id,antenna_id,:n].tolist()

                    idx_ang_H   = Idx_ang_all[i_comp,0,user_id,antenna_id,:n].tolist()
                    idx_ang_L   = Idx_ang_all[i_comp,1,user_id,antenna_id,:n].tolist()
                    idx_ang_L2H = Idx_ang_all[i_comp,2,user_id,antenna_id,:n].tolist()

                    R_ctw_abs, R_sbs_abs, R_prp_abs, flag_abs_train, flag_abs_test\
                        = compressionRate(idx_abs_H, idx_abs_L, idx_abs_L2H, M_abs_H, M_abs_L,\
                                           D, q_abs, train_size, beta, alpha, nupdate, adaptiveFlag, ntimes, training,\
                                           draw_flag, ii_name+'_abs', show_probs=True)
                    
                    R_ctw_ang, R_sbs_ang, R_prp_ang, flag_ang_train, flag_ang_test\
                        = compressionRate(idx_ang_H, idx_ang_L, idx_ang_L2H, M_ang_H, M_ang_L,\
                                           D, q_ang, train_size, beta, alpha, nupdate, adaptiveFlag, ntimes, training,\
                                           draw_flag, ii_name+'_ang', show_probs=True)

                    if training:
                        Flag_abs_train[user_id][antenna_id] = flag_abs_train
                        Flag_ang_train[user_id][antenna_id] = flag_ang_train

                    Flag_abs_test[user_id][antenna_id]  = flag_abs_test
                    Flag_ang_test[user_id][antenna_id]  = flag_ang_test

                    R_ctw_pol_matrix[user_id][antenna_id] = R_ctw_ang + R_ctw_abs
                    R_sbs_pol_matrix[user_id][antenna_id] = R_sbs_ang + R_sbs_abs
                    R_prp_pol_matrix[user_id][antenna_id] = R_prp_ang + R_prp_abs

            # CTW    
            R_ctw_pol = R_ctw_pol_matrix.mean()
            R_ctw_all.append(R_ctw_pol)

            # CTM symbol by symbol
            R_sbs_pol = R_sbs_pol_matrix.mean()
            R_sbs_all.append(R_sbs_pol)

            # Proposed individual
            R_prp_pol_indiv = R_prp_pol_matrix.mean()
            R_prp_indiv_all.append(R_prp_pol_indiv)

            # Proposed joint
            R_prp_pol_joint, R_prp_pol_joint_sbs, R_prp_pol_joint_smp, _ = \
                computeJointRate(Flag_abs_test, Flag_ang_test, Flag_abs_train, Flag_ang_train,\
                                 mm_abs_L, mm_ang_L, q_abs, q_ang, Nt, Nr, D, adaptiveFlag, ntimes, training)

            R_prp_joint_all.append(R_prp_pol_joint)
            R_prp_joint_sbs_all.append(R_prp_pol_joint_sbs)
            R_prp_joint_smp_all.append(R_prp_pol_joint_smp)

            # Save results
            Flags_abs_train_all.append(Flag_abs_train)
            Flags_ang_train_all.append(Flag_ang_train)
            Flags_abs_test_all.append(Flag_abs_test)
            Flags_ang_test_all.append(Flag_ang_test)

        else:
            R_ctw_all.append(0)
            R_sbs_all.append(0)
            R_prp_indiv_all.append(0)
            R_prp_joint_all.append(0)
            R_prp_joint_sbs_all.append(0)
            R_prp_joint_smp_all.append(0)

            if training:
                Flags_abs_train_all.append(Flag_abs_train)
                Flags_ang_train_all.append(Flag_ang_train)
            Flags_abs_test_all.append(Flag_abs_test)
            Flags_ang_test_all.append(Flag_ang_test)


    # To array
    Flags_abs_train_all = np.array(Flags_abs_train_all)
    Flags_ang_train_all = np.array(Flags_ang_train_all)
    Flags_abs_test_all  = np.array(Flags_abs_test_all)
    Flags_ang_test_all  = np.array(Flags_ang_test_all)

    ## Cube split quantiser
    if qts_flags[3]:
        R_ctw_CS_matrix  = np.zeros((Nr, Nt))
        R_sbs_CS_matrix  = np.zeros((Nr, Nt))
        R_prp_CS_matrix  = np.zeros((Nr, Nt))

        Flag_CS_re_train = np.zeros((Nr, Nt, len_training-D))
        Flag_CS_im_train = np.zeros((Nr, Nt, len_training-D))
        Flag_CS_re_test  = np.zeros((Nr, Nt, len_testing-D))
        Flag_CS_im_test  = np.zeros((Nr, Nt, len_testing-D))

        i_name = tree_filename + '_CS'

        for user_id in range(Nr):
            for antenna_id in range(Nt):
                ii_name = i_name + '_usr{}_ant{}'.format(user_id, antenna_id)
                if user_id==0 and antenna_id==0 and draw_flag_ext==True:
                        draw_flag = True
                else:
                        draw_flag = False

                idx_CS_re_H   = Idx_CS_re_all[0,user_id,antenna_id,:n].tolist()
                idx_CS_re_L   = Idx_CS_re_all[1,user_id,antenna_id,:n].tolist()
                idx_CS_re_L2H = Idx_CS_re_all[2,user_id,antenna_id,:n].tolist()

                idx_CS_im_H   = Idx_CS_im_all[0,user_id,antenna_id,:n].tolist()
                idx_CS_im_L   = Idx_CS_im_all[1,user_id,antenna_id,:n].tolist()
                idx_CS_im_L2H = Idx_CS_im_all[2,user_id,antenna_id,:n].tolist()

                R_ctw_CS_re, R_sbs_CS_re, R_prp_CS_re, flag_CS_re_train, flag_CS_re_test\
                    = compressionRate(idx_CS_re_H, idx_CS_re_L, idx_CS_re_L2H, M_CS_H, M_CS_L,\
                                      D, q_rect, train_size, beta, alpha, nupdate, adaptiveFlag, ntimes, training)

                R_ctw_CS_im, R_sbs_CS_im, R_prp_CS_im, flag_CS_im_train, flag_CS_im_test\
                    = compressionRate(idx_CS_im_H, idx_CS_im_L, idx_CS_im_L2H, M_CS_H, M_CS_L,\
                                      D, q_rect, train_size, beta, alpha, nupdate, adaptiveFlag, ntimes, training)

                if training:
                    Flag_CS_re_train[user_id][antenna_id] = flag_CS_re_train
                    Flag_CS_im_train[user_id][antenna_id] = flag_CS_im_train
                Flag_CS_re_test[user_id][antenna_id]  = flag_CS_re_test
                Flag_CS_im_test[user_id][antenna_id]  = flag_CS_im_test

                R_ctw_CS_matrix[user_id][antenna_id]  = R_ctw_CS_re + R_ctw_CS_im
                R_sbs_CS_matrix[user_id][antenna_id]  = R_sbs_CS_re + R_sbs_CS_im
                R_prp_CS_matrix[user_id][antenna_id]  = R_prp_CS_re + R_prp_CS_im

        # CTW    
        R_ctw_CS = R_ctw_CS_matrix.mean()
        R_ctw_all.append(R_ctw_CS)

        # CTM symbol by symbol
        R_sbs_CS = R_sbs_CS_matrix.mean()
        R_sbs_all.append(R_sbs_CS)

        # Proposed individual
        R_prp_CS_indiv  = R_prp_CS_matrix.mean()
        R_prp_indiv_all.append(R_prp_CS_indiv)

        # Proposed joint
        R_prp_CS_joint, R_prp_CS_joint_sbs, R_prp_CS_joint_smp, _ = \
            computeJointRate(Flag_CS_re_test, Flag_CS_im_test, Flag_CS_re_train, Flag_CS_im_train,\
                             mm_CS_L, mm_CS_L, q_rect, q_rect, Nt, Nr, D, adaptiveFlag, ntimes, training)

        R_prp_joint_all.append(R_prp_CS_joint)
        R_prp_joint_sbs_all.append(R_prp_CS_joint_sbs)
        R_prp_joint_smp_all.append(R_prp_CS_joint_smp)

    else:
        R_ctw_all.append(0)
        R_sbs_all.append(0)
        R_prp_indiv_all.append(0)
        R_prp_joint_all.extend([0,0,0])
    
    ###################################################
    # Communication rate and distortion
    ###################################################

    H_testing    = H[:,:,len_training+D:len_all]
    Hobs_testing = Hobs[:,:,len_training+D:len_all]

    ## Compander quantisers
    for i_comp in range(3): # unif, mu, beta
        if qts_flags[i_comp]:

            # Load quantised matrices
            Hhat_pol_H   = Hhat_pol_all[i_comp,0,:,:,:n]
            Hhat_pol_L   = Hhat_pol_all[i_comp,1,:,:,:n]
            Hhat_pol_L2H = Hhat_pol_all[i_comp,2,:,:,:n]

            Hhat_pol_M    = np.zeros(Hhat_pol_H.shape,  dtype=complex)

            # Load Flag
            Flag_abs = Flags_abs_test_all[i_comp,:,:,:]
            Flag_ang = Flags_ang_test_all[i_comp,:,:,:]

            Hhat_pol_H_testing = Hhat_pol_H[:,:,len_training+D:len_all]
            Hhat_pol_L_testing = Hhat_pol_L[:,:,len_training+D:len_all]
            Hhat_pol_M_testing = computeMatrixMixed(Flag_abs, Flag_ang, Hhat_pol_H_testing, Hhat_pol_L_testing, seq_length)    

            # Real distortion
            # Polar
            _, NMSE_dB_pol_H = computeNMSE(Hnorm[:,:,len_training+D:len_all], Hhat_pol_H_testing)
            _, MSCD_dB_pol_H = computeMSCD(Hnorm[:,:,len_training+D:len_all], Hhat_pol_H_testing)
            NMSE_dB_H_all.append(NMSE_dB_pol_H)
            MSCD_dB_H_all.append(MSCD_dB_pol_H)

            _, NMSE_dB_pol_M = computeNMSE(Hnorm[:,:,len_training+D:len_all], Hhat_pol_M_testing)
            _, MSCD_dB_pol_M = computeMSCD(Hnorm[:,:,len_training+D:len_all], Hhat_pol_M_testing)
            NMSE_dB_M_all.append(NMSE_dB_pol_M)
            MSCD_dB_M_all.append(MSCD_dB_pol_M)

            # Communication rate
            # Polar
            [qtz_CSI_pol_H, anl_CSI, perf_CSI] = communication_rate(Hhat_pol_H_testing, Hobs_testing, H_testing, SNR, Nt, Nr, b2=1e10)
            [qtz_CSI_pol_M, anl_CSI, perf_CSI] = communication_rate(Hhat_pol_M_testing, Hobs_testing, H_testing, SNR, Nt, Nr, b2=1e10)
            comm_rate_H_all.append(qtz_CSI_pol_H)
            comm_rate_M_all.append(qtz_CSI_pol_M)

        else:
            NMSE_dB_H_all.append(0)
            MSCD_dB_H_all.append(0)
            NMSE_dB_M_all.append(0)
            MSCD_dB_M_all.append(0)
            comm_rate_H_all.append(0)
            comm_rate_M_all.append(0)

    ## Cube-split quantiser
    if qts_flags[3]:
        # Load quantised matrices
        Hhat_CS_H   = Hhat_CS_all[0,:,:,:n]
        Hhat_CS_L   = Hhat_CS_all[1,:,:,:n]
        Hhat_CS_L2H = Hhat_CS_all[2,:,:,:n]

        Hhat_CS_H_testing = Hhat_CS_H[:,:,len_training+D:len_all]
        Hhat_CS_L_testing = Hhat_CS_L[:,:,len_training+D:len_all]
        Hhat_CS_M_testing = computeMatrixMixed(Flag_CS_re_test, Flag_CS_im_test, Hhat_CS_H_testing, Hhat_CS_L_testing, seq_length)    
        
        # Real distortion   
        _, NMSE_dB_CS_H = computeNMSE(Hnorm[:,:,len_training+D:len_all], Hhat_CS_H_testing)
        _, MSCD_dB_CS_H = computeMSCD(Hnorm[:,:,len_training+D:len_all], Hhat_CS_H_testing)
        NMSE_dB_H_all.append(NMSE_dB_CS_H)
        MSCD_dB_H_all.append(MSCD_dB_CS_H)

        _, NMSE_dB_CS_M = computeNMSE(Hnorm[:,:,len_training+D:len_all], Hhat_CS_M_testing)
        _, MSCD_dB_CS_M = computeMSCD(Hnorm[:,:,len_training+D:len_all], Hhat_CS_M_testing)
        NMSE_dB_M_all.append(NMSE_dB_CS_M)
        MSCD_dB_M_all.append(MSCD_dB_CS_M)
        
        # Communication rate
        [qtz_CSI_CS_H, anl_CSI, perf_CSI] = communication_rate(Hhat_CS_H_testing, Hobs_testing, H_testing, SNR, Nt, Nr, b2=1e10)
        [qtz_CSI_CS_M, anl_CSI, perf_CSI] = communication_rate(Hhat_CS_M_testing, Hobs_testing, H_testing, SNR, Nt, Nr, b2=1e10)
        
        comm_rate_H_all.append(qtz_CSI_CS_H)
        comm_rate_M_all.append(qtz_CSI_CS_M)
    else:
        NMSE_dB_H_all.append(0)
        MSCD_dB_H_all.append(0)
        NMSE_dB_M_all.append(0)
        MSCD_dB_M_all.append(0)
        comm_rate_H_all.append(0)
        comm_rate_M_all.append(0)

    return R_uncomp_all, R_ctw_all, R_sbs_all, R_prp_indiv_all, R_prp_joint_all, R_prp_joint_sbs_all, R_prp_joint_smp_all,\
    		comm_rate_H_all, comm_rate_M_all, anl_CSI, perf_CSI,\
            NMSE_dB_H_all, NMSE_dB_M_all, MSCD_dB_H_all, MSCD_dB_M_all,\
            Flags_abs_test_all, Flags_ang_test_all, Flags_abs_train_all, Flags_ang_train_all