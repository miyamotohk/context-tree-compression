import numpy as np

def upper_concave_env(X, Y, tol=0.0):
    """
    Find the set forming an upper concave envelop of the input set.
    """
    
    assert len(X)==len(Y), "X and Y should have same length" 
    n = len(X)
    
    to_remove = [0 for i in range(n)] # Indices of elements to keep
    
    # For every point in the input set
    for i in range(n):
                        
        remove_point = False    
        
        S_equal = [j for j in range(n) if X[j]==X[i] and Y[j]!=Y[i]]
        S_left  = [j for j in range(n) if j!=i and X[j]<=X[i]]
        S_right = [j for j in range(n) if j!=i and X[j]>=X[i]]
        
        # LR_pair is the set of all pairs embracing i
        LR_pair = []
        for jL in S_left:
            for jR in S_right:
                LR_pair.append([jL,jR])
        
        # Check if i is above or below line formed by any pair embracing i
        ipair = 0
        while ipair<len(LR_pair) and remove_point==False:
            [X_L, X_R] = [X[LR_pair[ipair][0]], X[LR_pair[ipair][1]]]
            [Y_L, Y_R] = [Y[LR_pair[ipair][0]], Y[LR_pair[ipair][1]]]
            ipair += 1

            # Check if i is above the line formed by the pair
            # tol is a tolerence factor that is positive
            if (((Y_R-Y_L)*(X[i]-X_L))>=((Y[i]-Y_L)*(X_R-X_L)+tol)):
                if X_L!=X_R:
                    remove_point = True
            
        # Check if i has same X with lower Y as another point k
        ieq = 0
        while ieq<len(S_equal) and remove_point==False:
            k = S_equal[ieq]
            ieq += 1
            
            # Check if i has same X and lower Y
            if Y[i] <= Y[k]:
                remove_point = True
                
        # If yes, target the point as to be removed
        if remove_point==True:
            to_remove[i] = 1
        
    # Keep only elements that are not to be removed
    X_out = []
    Y_out = []
    for i_r, remove in enumerate(to_remove):
        if remove==0:
            X_out.append(X[i_r])
            Y_out.append(Y[i_r])
    n_out = len(X_out)
    
    # Sort
    mylist = [(X_out[i],Y_out[i]) for i in range(n_out)]
    mylist.sort(key = lambda z: z[0])
    X_out, Y_out = np.array(mylist).T
    
    return X_out, Y_out