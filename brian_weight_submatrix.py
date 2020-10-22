"""
returns a dense submatrix from a brian2 synaptic weight matrix
"""

from brian2 import *
import numpy as np
import scipy.sparse as sp

def brian_weight_submatrix(synapses,input_subset,output_subset):
    wmat = sp.coo_matrix((synapses.w/mV, (synapses.j, synapses.i)), shape=(synapses.target.N, synapses.source.N)).tocsr()
    wmat = wmat[output_subset,:].tocsc()
    wmat = wmat[:,input_subset].toarray()

    return wmat

"""
returns a list of indices in the native coo weight matrix that fall within the desired subset
"""

def brian_weights_in_subset(synapses,input_subset,output_subset):
    iin = np.asarray(np.isin(synapses.i,input_subset)).nonzero()
    iout = np.asarray(np.isin(synapses.j,output_subset)).nonzero()

    return np.intersect1d(iin, iout, assume_unique=True)
