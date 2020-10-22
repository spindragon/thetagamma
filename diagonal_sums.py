"""
returns a 1D array of the sums of the diagonals of the input matrix (NxN)
assumes that input is a weight matrix from a toroidal network,
and adds sum(diag(n)) to sum(diag(n-N))
if N is even, main diagonal is returned in element N/2 (when 0 indexed)
"""

import numpy as np

def diagonal_sums(a):
    s = a.shape
    if len(s)!=2 or s[0]!=s[1]:
        print('diagonal_sums: input is not a square matrix')
        return 0
    
    N = s[0]
    ds = np.zeros(N)
    ds[0] = np.sum(np.diagonal(a))
    for idiag in range(1,N):
        ds[idiag] = np.sum(np.diagonal(a,offset=idiag)) + np.sum(np.diagonal(a,offset=idiag-N))
        
    return np.roll(ds,N//2)

