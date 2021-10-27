################################################
# Compression functions
################################################

import numpy as np
import scipy as sp
import copy
import os, sys

from math import log2, floor, ceil, sqrt
from operator import add

from ct_main import *
from ct_draw import drawTree

### Compression schemes

def ctw(data, m, D, beta=0.5, alpha=2, show=False):
    """
    CTW algorithm with arithmetic encoding. Use weighting probability for encoding.
    Inputs:
        data:  sequence to compress
        m:     alphabet size
        D:     maximum tree depth
        beta:  weighting coefficient
        alpha: Dirichlet coefficient
        show:  flag to display intermediate steps
    Outputs:
        L:     length of the binary compressed sequence
        R:     compression rate
    """
    tree   = buildTree(data, m, D, 'CTW', beta, alpha)
    log2P0 = getlog2ProbSeqPw(tree)
    L      = ceil(-log2P0) + 1
    R      = float(L)/float(len(data))
    
    if show:
        print("log2P = {}".format(log2P0))
        print("L = {}".format(L))
        print("rho = {}".format(rho))
    
    return [L, R]

def ctm(training, testing, m, D, beta=0.5, alpha=2, show=False):
    """
    Two-step CTM algorithm with arithmetic encoding.
    Build model with training sequence and use maximising probability for encoding.
    Inputs:
        training: training sequence
        testing:  testing sequence
        m:        alphabet size
        D:        maximum tree depth
        beta:     weighted coefficient
        alpha:    Dirichlet coefficient
        show:     flag to display intermediate steps
    Outputs:
        L:     length of the binary compressed sequence
        R:     compression rate
    """
    tree   = buildTree(training, m, D, 'CTM', beta, alpha)
    ####################################
    log2P0 = newGetSeqProb(tree, D, testing, alpha)
    ###################################
    L      = ceil(-log2P0) + 1
    R      = float(L)/float(len(testing)-D)
    
    if show:
        print("log2P = {}".format(log2P0))
        print("L = {}".format(L))
        print("rho = {}".format(rho))
    
    return [L, R]

def ctm_sbs(training, testing, m, D, beta=0.5, alpha=2, nupdate=100):
    """
    Symbol by symbol compression using CTM model.
    Build initial model with training sequence, update model. Compute the maximising probability pi of each symbol
    and compress it using ceil(log2(-pi)) bits.
    Inputs:
        training: training sequence
        testing:  testing sequence
        m:        alphabet size
        D:        maximum tree depth
        beta:     weighted coefficient
        alpha:    Dirichlet coefficient
        nupdate:  interval between tree updates
    Outputs:
        L:     length of the binary compressed sequence
        R:     compression rate
    """

    # Use training sequnce to build CTM model
    tree  = buildTree(training, m, D, 'CTM', beta=beta, alpha=alpha, pruneFlag=False)
    tree2 = copy.deepcopy(tree)
    tree2 = pruneTree(tree2, D, beta=beta)

    # Initialisations
    testing = ([0]*D) + testing
    context  = testing[0:D]
    mybuffer = []
    log2P = 0
    L     = 0
    
    # Encode testing sequencce
    for i in range(len(testing)-D):
        # Find symbol and context
        xnext   = testing[i+D]
        context = findContext(tree2, testing[i:i+D])
        count   = context.count
        Ms      = sum(count)

        # Compute probability
        log2Pnext = log2(count[xnext]+(1/alpha))-log2(m/alpha + Ms)
        log2P     += log2Pnext
        
        lnext = np.ceil(-log2Pnext)
        L     += lnext

        # Save in buffer
        mybuffer.append(testing[i])

        # Update tree
        if i%nupdate==0:
            # Update hard copy
            tree = updateTree(tree, mybuffer, D, 'CTM', beta=beta, alpha=alpha, pruneFlag=False)
            # Update current model
            tree2 = copy.deepcopy(tree)
            tree2 = pruneTree(tree2, D, beta)
            # Clear buffer
            mybuffer.clear()
            mybuffer = []

    R = float(L)/float(len(testing)-D)
    L = int(L)

    return [L, R]

def encodeSymbolSingleRes(root, D, idx, idx_context, alpha=2):
    """
    Encode symbol with proposed method. Three-level encoding. Single-resolution.
    Inputs:
        root:        root node of tree
        D:           maximum tree depth
        idx:         symbol to encode
        idx_context: context of the symbol idx
        alpha:       Dirichlet coefficient
    Outputs:
        out:         encoded binary sequence
        new_context: context of next symbol
        level:       level used in the encoding (0, 1 or 2)
    """

    # Find context
    xnext = idx
    try:
        context = findContext(root, idx_context)
    except IndexError:        
        print(idx_context[len(idx_context)-1])        
        sys.exit()
        
    count = context.count

    # Compute theta: probabilities of each symbol
    m  = root.m
    Ms = sum(count)
    log2theta = [0] * m

    for j in range(m):
        log2theta[j] = log2(count[j]+(1/alpha))-log2(m/alpha + Ms)
    # Add little noise to make sure there are no ties
    extra     = [x*1e-8 for x in range(m)]
    log2theta = list(map(add, log2theta, extra))

    # Search max probability
    log2thetamax = max(log2theta)
    jmax         = log2theta.index(log2thetamax)
    # Save the probability of next symbol
    pxnext       = log2theta[xnext]

    # Sort vector and get index of xnext's probability
    log2theta.sort(reverse=True)

    # Compute size of level 1 list
    q2      = max(1,int(np.ceil(np.log2(m)))-2)
    m2      = 2**q2
    m2probs = log2theta[1:1+m2]

    # Update context
    new_context = idx_context

    # If xnext is the most probable symbol
    if xnext==jmax:
        out  = [0]
        new_context.append(idx)
        level = 0
    # If it is among m2 most probable ones
    elif pxnext in m2probs:
        idx2 = m2probs.index(pxnext)
        out  = [1]+[0] + myBin(idx2, q2)
        new_context.append(idx)
        level = 1
    # Otherwise
    else:
        # Encode the index
        out = [1]+[1] + myBin(idx, np.ceil(np.log2(m)))
        # Update the context with a re-encoded high-resolution one
        new_context.append(idx)
        level = 2

    return [out, new_context, level]

def encodeSymbolMultiRes(root, D, idx_L, idx_H, idx_L2H, idx_context, m_L, q, alpha=2):
    """
    Encode symbol with proposed method. Three-level encoding. Multi-resolution.
    Inputs:
        root:        root node of tree
        D:           maximum tree depth
        idx_L:       low-resolution symbol (index)
        idx_H:       high-resolution symbol (index)
        idx_L2H:     projected low-to-high resolution symbol (index)
        idx_context: context of the symbol idx
        m_L:         low-resolution alphabet size
        q=[q1,q2]:   number of bits (minus one) used in levels 0 and 1       
        alpha:       Dirichlet coefficient
    Outputs:
        out:         encoded binary sequence
        new_context: context of next symbol
        level:       level used in the encoding (0, 1 or 2)
    """
    # Find context
    xnext = idx_H
    try:
        context = findContext(root, idx_context)
    except IndexError:        
        print(idx_context[len(idx_context)-1])        
        sys.exit()

    count = context.count

    # Compute theta: proabilities of each symbol
    m  = root.m
    Ms = sum(count)
    log2theta = [0] * m

    for j in range(m):
        log2theta[j] = log2(count[j]+(1/alpha))-log2(m/alpha + Ms)
    # Add little noise to make sure there are no ties
    extra     = [x*1e-8 for x in range(m)]
    log2theta = list(map(add, log2theta, extra))

    # Search max probability
    log2thetamax = max(log2theta)
    jmax         = log2theta.index(log2thetamax)
    # Save the probability of next symbol
    pxnext       = log2theta[xnext]

    # Sort vector and get index of xnext's probability
    log2theta.sort(reverse=True)

    [q1,q2] = q
    # Verify if q1==0
    assert q1==0, "q1 should be 0, but is q1={}".format(q1)
    m1 = 2**q1
    m1probs = log2theta[0:m1]
    
    # Compute size of level 1 list
    m2 = 2**q2
    #m2probs = log2theta[m1:m1+m2]
    m2probs = log2theta[0:m2]

    # Update context
    new_context = idx_context

    # If xnext is the most probable symbol
    if pxnext in m1probs and m1==1:
        # Encode with a single bit
        idx1 = m1probs.index(pxnext)
        out  = [0]
        new_context.append(idx_H)
        level = 0
    # If it is among the m2 most probable ones
    elif pxnext in m2probs:
        # Encode index in list
        idx2 = m2probs.index(pxnext)
        out  = [1]+[0] + myBin(idx2, np.ceil(np.log2(m2)))
        new_context.append(idx_H)
        level = 1
    # Otherwise
    else:
        # Encode the low-resolution index
        out = [1]+[1] + myBin(idx_L, np.ceil(np.log2(m_L)))
        # Update the context with a re-encoded high-resolution one
        new_context.append(idx_L2H)
        level = 2

    return [out, new_context, level]

def compressSingleRes(training, testing, m, D, beta=0.5, alpha=2, nupdate=100, adaptiveFlag=False,\
                      ntimes=10, trainingFlag=True, drawFlag=False, filename='tree', show_probs=False):
    """
    Compress a sequence, with multi-level encoding and single-resolution alphabet. Used for indicator sequence.
    Inputs:
        training:     training sequence
        testing:      testing sequence
        m:            alphabet size
        D:            maximum tree depth
        beta:         weighting coefficient
        alpha:        Dirichlet coefficient
        nupdate:      interval between tree updates
        adaptiveFlag: indicates if adaptive model should be used    
        ntimes:       defines the number of symbols used for adapting the model as nupdate*ntimes
        drawFlag:     flag indicating to plot the final tree model
        filename:     filename for tree model plot (pdf)
        show_probs:   flag indicating to write probabilities in the model plot
    Outputs:
        L:            length of binary encoded sequence
        R:            compression rate
        Levels:       levels of training compressed sequence
    """

    # Use training sequence to build CTM model
    if trainingFlag:
        tree  = buildTree(training, m, D, 'CTM', beta, alpha, pruneFlag=False)
    else:
        tree  = buildTree([], m, D, 'CTM', beta, alpha, pruneFlag=False)
    
    # Initialisations
    testing   = ([0]*D) + testing
    encoded   = []
    Levels    = []
    context   = testing[0:D]

    short_buffer = []
    long_buffer  = []
    old_symbols  = []
    
    # Encode testing sequencce
    for i in range(len(testing)-D):
        short_buffer.append(testing[i])
        [enci, context, level] = encodeSymbolSingleRes(tree, D, testing[i+D], context, alpha)
        encoded += enci
        Levels.append(level)

        # Update tree
        if i%nupdate==0:
            # Update hard copy
            tree = updateTree(tree, short_buffer, D, 'CTM', beta=beta, alpha=alpha, pruneFlag=False)

            if adaptiveFlag:
                # Update long buffer
                long_buffer.append(short_buffer)

                if len(long_buffer) == ntimes+1:
                    old_symbols = long_buffer[0]
                    long_buffer = long_buffer[-ntimes:]
                    tree = downgradeTree(tree, old_symbols, D, 'CTM', beta=beta, pruneFlag=False)
            
            short_buffer = short_buffer[-D:]

    # Prune tree
    tree2 = copy.deepcopy(tree)
    tree2 = pruneTree(tree2, D, beta)
    # Draw tree
    if drawFlag:
        drawTree(tree2, 'CTM', filename, show_probs)
        
    L   = len(encoded)
    R   = float(L)/float(len(testing)-D)
    Levels = Levels[D:]
            
    return [L, R, Levels]

def compressMultiRes(training_L, training_H, training_L2H, testing_L, testing_H, testing_L2H, m_L, m_H,\
                     D, q, beta=0.5, alpha=2, nupdate=100, adaptiveFlag=False, ntimes=10, trainingFlag=True,\
                     drawFlag=False, filename='tree', show_probs=False):
    """
    Compress a sequence, with multi-level encoding and multi-resolution alphabet. Used for CSI indices.
    Adaptive scheme: at each instant the context-tree is built with nmodel indices.
    Inputs:
        training_L:      low-resolution training sequence
        training_H:      high-resolution training sequence
        training_L2H:    projected low-to-high resolution training sequence
        testing_L:       low-resolution testing sequence
        testing_H:       high-resolution testing sequence
        testing_L2H:     projected low-to-high resolution testing sequence
        m_L:             low-resolution alphabet size
        m_H:             high-resolution alphabet size
        D:               maximum tree depth
        q:               number of bits (minus one) used in levels 0 and 1
        beta:            weighting coefficient
        alpha:           Dirichlet coefficient
        nupdate:         interval between tree updates
        adaptiveFlag:    indicates if adaptive model should be used    
        ntimes:          defines the number of symbols used for adapting the model as nupdate*ntimes
        drawFlag:        flag indicating to plot the final tree model
        filename:        filename for tree model plot (pdf)
        show_probs:      flag indicating to write probabilities in the model plot
    Outputs:
        L:               length of binary encoded sequence
        R:               compression rate
        Levels_training: levels of training compressed sequence
        Levels_testing:  levels of testing compressed sequence
    """
    m = m_H

    # Use training sequnce to build CTM model
    tree  = buildTree([], m, D, 'CTM', beta, alpha, pruneFlag=False)

    # Initialisations

    encoded         = []
    Levels_training = []
    Levels_testing  = []

    if trainingFlag:
        training_H   = ([0]*D) + training_H
        training_L   = ([0]*D) + training_L
        training_L2H = ([0]*D) + training_L2H

        context      = training_H[0:D]
        short_buffer = training_H[0:D]

        # Encode training sequence
        for i in range(len(training_H)-D):
            short_buffer.append(training_H[i+D])
            [enci, context, level] = encodeSymbolMultiRes(tree, D, training_L[i+D], training_H[i+D], training_L2H[i+D],\
                                                          context, m_L, q, alpha)
            Levels_training.append(level)
            
            # Update tree
            tree = updateTree(tree, short_buffer, D, 'CTM', beta=beta, alpha=alpha, pruneFlag=False)
            short_buffer = short_buffer[-D:]

    # Prune tree
    tree2 = copy.deepcopy(tree)
    tree2 = pruneTree(tree2, D, beta)
    # Draw tree
    if drawFlag:
        drawTree(tree2, 'CTM', filename+'_train', show_probs)
    
    testing_H   = ([0]*D) + testing_H
    testing_L   = ([0]*D) + testing_L
    testing_L2H = ([0]*D) + testing_L2H

    context      = testing_H[0:D]
    short_buffer = testing_H[0:D]
    long_buffer  = []
    old_symbols  = []
    
    # Encode testing sequencce
    for i in range(len(testing_H)-D):
        short_buffer.append(testing_H[i+D])
        [enci, context, level] = encodeSymbolMultiRes(tree, D, testing_L[i+D], testing_H[i+D], testing_L2H[i+D],\
                                                      context, m_L, q, alpha)
        encoded += enci
        Levels_testing.append(level)

        # Update tree
        if i%nupdate==0:
            # Update model
            tree = updateTree(tree, short_buffer, D, 'CTM', beta=beta, alpha=alpha, pruneFlag=False)
            
            if adaptiveFlag:
                # Update long buffer
                long_buffer.append(short_buffer)

                if len(long_buffer) == ntimes+1:
                    old_symbols = long_buffer[0]
                    long_buffer = long_buffer[-ntimes:]
                    tree = downgradeTree(tree, old_symbols, D, 'CTM', beta=beta, pruneFlag=False)
                                
            # Clear short buffer
            short_buffer = short_buffer[-D:]

    # Prune tree
    tree2 = copy.deepcopy(tree)
    tree2 = pruneTree(tree2, D, beta)
    # Draw tree
    if drawFlag:
        drawTree(tree2, 'CTM', filename+'_final', show_probs)
        
    L   = len(encoded)
    R   = float(L)/float(len(testing_H)-D)
    Levels_training = Levels_training[D:]
    Levels_testing  = Levels_testing[D:]

    return [L, R, Levels_training, Levels_testing]

### Auxiliary functions
def myBin(n,L):
    """
    Convert decimal to binary with L bits.
    """
    out = []
    # Convert n to binary string
    aux = "{0:b}".format(n)
    # Add left 0s
    while (len(aux) < L):
        aux = '0' + aux
    # Convert aux into int array
    for i in range(len(aux)):
        out.append(int(aux[i]))
    return out

def myDec(x):
    """
    Convert binary to decimal.
    """
    out = 0
    x.reverse()
    for i in range(len(x)):
        out += x[i]*(2**i)
    return out

def findContext(root, data):
    """
    Find context of a sequence.
    """
    i = len(data)
    node = root

    while (node.isLeaf() or node.isToPrune) is False:
        i -= 1
        node = node.children[data[i]]
    return node