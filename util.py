import numpy as np

def calculate_total_iters_hyperband(eta,R):
    
    s_max = np.floor(np.log(R)/np.log(eta))
    B = (s_max+1)*R
    s = np.arange(s_max+1)

    total_n = np.ceil((B/R)*(eta**s)/(s+1)).sum()

    return (1+s_max)*B, total_n
