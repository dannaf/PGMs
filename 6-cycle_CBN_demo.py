#!/usr/bin/env python3
# Python 3.9.10

import numpy  as np # numpy==1.19.4
import cvxpy  as cp # cvxpy==1.1.12
import pandas as pd # pandas==1.1.4
import itertools
import math


######################
## Helper functions ##
######################

# Set up helper fcns
def neighbors(a, n):
    return [(a-1)%n, (a+1)%n]

def marginalize(P, nodes, keepdims=True): # nodes is a tuple or list of the indices of the nodes to keep, marginalized
    nodes = sorted(nodes) # ensure the nodes are sorted
    n = len(P.shape)
    return P.sum(axis=tuple(axis for axis in range(n) if axis not in nodes), keepdims=keepdims)

def conditionalize(P, nodes, given_nodes, keepdims=True):
    return marginalize(P, list(set(nodes).union(set(given_nodes))), keepdims=keepdims)/marginalize(P, given_nodes , keepdims=keepdims) # not sorting here since current setup has marginalize sort it anyways

def get_subjoint_indices(local_subjoint):
    n = len(local_subjoint.shape)
    naive_list =np.array([idx for idx, dim in enumerate(local_subjoint.shape) if dim > 1])
    for _ in range(len(naive_list)):
        if((naive_list[0] != (naive_list[1] - 1) % n) or (naive_list[2] != (naive_list[1] + 1) % n)):
            naive_list = np.hstack([naive_list[1::], naive_list[0]])
        else:
            return naive_list.tolist()
    # if did not return within n tries, raise error
    raise Exception(f"failed to sort subjoint_indices_list")

def marginalization_matrix(ls, nodes=None):
    nodes = get_subjoint_indices(ls)
    df_nodes = pd.DataFrame(np.array(list(itertools.product(*[range(m) for m in np.array(ls.shape)[sorted(nodes)]]))) == 1, columns=sorted(nodes))
    df_idxs = pd.DataFrame(idxs == 1, columns=range(n))
    return np.vstack([(df_idxs[nodes] == sr).all(axis=1).astype(int) for _, sr in df_nodes.iterrows()])


def local_marginalization_matrix(ls, local_idxs_to_marginalize_to):
    nodes = np.array(get_subjoint_indices(ls))[local_idxs_to_marginalize_to].tolist()
    df_idxs = pd.DataFrame(np.array(list(itertools.product(*[range(m) for m in np.array(ls.shape)]))), columns=range(n))
    df_nodes = pd.DataFrame(np.array(list(itertools.product(*[range(m) for m in np.array(ls.shape)[sorted(nodes)]]))) == 1, columns=sorted(nodes))
    return np.vstack([(df_idxs[nodes] == sr).all(axis=1).astype(int) for _, sr in df_nodes.iterrows()])


def marginalization_matrix_for_halfjoints(shape, nodes=None):
    '''This is a function specific for getting a halfjoint vector from the fulljoint vector (to be used inside the linear program, with the fulljoint vector s, which is not the true fulljoint but rather is a surrogate system)'''
    if nodes is None: nodes = [idx for idx in range(n) if (idx % 2) == 0]
    df_idxs = pd.DataFrame(idxs == 1, columns=range(n))
    df_nodes = pd.DataFrame(np.array(list(itertools.product(*[range(m) for m in np.array(shape)[sorted(nodes)]]))) == 1, columns=sorted(nodes))
    return np.vstack([(df_idxs[nodes] == sr).all(axis=1).astype(int) for _, sr in df_nodes.iterrows()])


################################################################
## Load sample 6-cycle system local conditional specification ##
################################################################

# This is a sample 6-cycle system that was found from solving an (improper) MRF
# using the following factors expanding on the Koller+Friedman example 4-cycle
# using the following code

def create_system_as_MRF_and_return_local_subjoints_only(return_fulljoint=False):
    # Set up MRF factors

    # # a strictly positive example
    # factors = []
    # phi12 = np.array([30,  5, 1, 10 ]).reshape((2, 2)); factors.append(phi12)
    # phi23 = np.array([100, 1, 1, 100]).reshape((2, 2)); factors.append(phi23)
    # phi34 = np.array([1, 100, 100, 1]).reshape((2, 2)); factors.append(phi34)
    # phi45 = np.array([100, 1, 1, 100]).reshape((2, 2)); factors.append(phi45)
    # phi56 = np.array([50, 75, 10, 5 ]).reshape((2, 2)); factors.append(phi56)
    # phi61 = np.array([10, 10, 25, 30]).reshape((2, 2)); factors.append(phi61)
   
    # a not strictly-positive example
    factors = []
    phi12 = np.array([30,  5, 0, 10 ]).reshape((2, 2)); factors.append(phi12)
    phi23 = np.array([100, 1, 1, 100]).reshape((2, 2)); factors.append(phi23)
    phi34 = np.array([1, 100, 100, 1]).reshape((2, 2)); factors.append(phi34)
    phi45 = np.array([100, 0, 1, 100]).reshape((2, 2)); factors.append(phi45)
    phi56 = np.array([50, 75, 10, 5 ]).reshape((2, 2)); factors.append(phi56)
    phi61 = np.array([10, 10, 25, 30]).reshape((2, 2)); factors.append(phi61)
    # Compute fulljoint
    # p is the vectorized version
    # P is a multi-dimensional numpy array, shape (2, 2, 2, 2, 2, 2)
    idxs = list(itertools.product(*[range(factor.shape[0]) for factor in factors]))
    p_unnormalized = np.array([[factor[idx_tuple[factor_idx], idx_tuple[(factor_idx+1) % len(factors)]] for factor_idx, factor in enumerate(factors)] for idx_tuple in idxs]).prod(axis=1) # NOTE: this implementation only takes square factors
    p = p_unnormalized/p_unnormalized.sum()
    P = p.reshape((2, 2, 2, 2, 2, 2))
    #
    # Verify the validity of the joint distribution
    # P.sum() == 1 # check using np.abs(P.sum() - 1)
    # (P >= 0).all() == True
    #
    if(return_fulljoint):
        return P
    
    # Return the local subjoints
    n = len(P.shape)
    # local_subjoints = [marginalize(P, [((a+1)-1)%n, (a+1)%n, ((a+1)+1)%n]) for a in range(n)] # first one is centered on 2 (1 in zero-indexed) so that first suboint is P(1,2,3) rather than P(n,1,2)
    local_subjoints = [marginalize(P, [(a-1)%n, a, (a+1)%n]) for a in range(n)] # actually starting with center-idx 0

    return local_subjoints


########################################
## Compute the aggregate conditionals ##
########################################

# Here we take the local conditional specifications  of the cyclic BN
# and multiply them into an aggregate conditional distribution of all
# odds given all evens, or all evens given all odds, per Theorem 1

# Get the local conditional specification
local_subjoints = create_system_as_MRF_and_return_local_subjoints_only()

# Set up the subsystem sets for the evens and odds and for all the variables
n = len(local_subjoints)            # system size, number of random variables
A = list(range(n))                  # all the random variables in the system
E = [a for a in A if (a % 2) == 0]  # all the even random variables only
O = [a for a in A if (a % 2) == 1]  # all the odd random variables only

# Compute the aggregate conditionals according to the formula in Theorem 1
# From each local subjoint conditionalize the central variable on the others
P_E_given_O = math.prod([conditionalize(
    ls,  # conditoinalize the local subjoint
    [get_subjoint_indices(ls)[1]], # get conditional of the central node
    np.array(get_subjoint_indices(ls))[[0, 2]].tolist()) # given the outer nodes
    for ls_idx, ls in enumerate(local_subjoints) if (ls_idx % 2) == 0])
P_O_given_E = math.prod([conditionalize(
    ls,  # conditoinalize the local subjoint
    [get_subjoint_indices(ls)[1]], # get conditional of the central node
    np.array(get_subjoint_indices(ls))[[0, 2]].tolist()) # given the outer nodes
    for ls_idx, ls in enumerate(local_subjoints) if (ls_idx % 2) == 1])

#############################
## Prepare the constraints ##
#############################

# With the aggregate conditional distributions of odds given evens or vice versa
# we now start preparing the implementation of a linear program with that spec.

# The two matrices constructed below,
# O_times_E_given_O_matrix and E_times_O_given_E_matrix,
# perform the following element-wise multiplications as matrix multiplications
# P_O*P_E_given_O, if P_O were a (1, 2, 1, 2, 1, 2)-shaped numpy array
# P_E*P_O_given_E, if P_E were a (2, 1, 2, 1, 2, 1)-shaped numpy array
# where P_E_given_O and P_O_given_E both have shape (2, 2, 2, 2, 2, 2)

idxs = np.array(list(itertools.product(*[range(i) for i in P_E_given_O.shape]))) # or P_O_given_E.shape
df_idxs = pd.DataFrame(idxs == 1, columns=range(n))

df_odds = pd.DataFrame(np.array(list(itertools.product(*[range(m) for m in (1, 2, 1, 2, 1, 2)]))), columns=range(n))
df_evens = pd.DataFrame(np.array(list(itertools.product(*[range(m) for m in (2, 1, 2, 1, 2, 1)]))), columns=range(n))

E_given_O_matrix = \
    np.array([(df_odds[[1,3,5]] == df_idxs[[1,3,5]].iloc[row_idx]).all(axis=1).astype(int).values*P_E_given_O[tuple(df_idxs.iloc[row_idx].astype(int).values.tolist())] for row_idx in range(64)])

O_given_E_matrix = \
    np.array([(df_evens[[0,2,4]] == df_idxs[[0,2,4]].iloc[row_idx]).all(axis=1).astype(int).values*P_O_given_E[tuple(df_idxs.iloc[row_idx].astype(int).values.tolist())] for row_idx in range(64)])

halfjoint_marginalization_matrix_for_evens = marginalization_matrix_for_halfjoints(P_O_given_E.shape, nodes=[idx for idx in range(n) if (idx % 2) == 0])
halfjoint_marginalization_matrix_for_odds = marginalization_matrix_for_halfjoints(P_E_given_O.shape, nodes=[idx for idx in range(n) if (idx % 2) == 1])


#################################
## Run the linear program (LP) ##
#################################

# Set up the LP with the constraints
s = cp.Variable(64)
objective = cp.Minimize(1)
constraints = []
constraints.append(s >= 0)
constraints.append(cp.sum(s) == 1)
constraints.append(
        E_given_O_matrix @
        (halfjoint_marginalization_matrix_for_odds @ s)
        ==
        O_given_E_matrix @
        (halfjoint_marginalization_matrix_for_evens @ s)
        )
problem = cp.Problem(objective, constraints)
result = problem.solve()
s = s.value

# Reconstruct the full-joint distribution according to P=P(E|O)P(O)=P(O|E)P(E)
# (This just recomputes the constraint, using the final decision variable value)
p_reconstructed_from_EgO_times_O = E_given_O_matrix @ (halfjoint_marginalization_matrix_for_odds @ s)
p_reconstructed_from_OgE_times_E = O_given_E_matrix @ (halfjoint_marginalization_matrix_for_evens @ s)

# Verify the solution: the LP decision variable, s, reconstructs the fulljoint p
# (By loading the ground-truth P fulljoint only here at the end, we make sure we did not accidentally use it during the reconstruction)
p = create_system_as_MRF_and_return_local_subjoints_only(return_fulljoint=True).flatten()
np.abs(p_reconstructed_from_EgO_times_O - p).max() # 1e-16, verified
np.abs(p_reconstructed_from_OgE_times_E - p).max() # 1e-15, verified

# Discussion
# Note that the resulting decision variables from the LP do not match, ie s != p
np.abs(s-p).max() # 0.0012899476244102037, ie not equal
# But that is because s is just some other system that matches P(O) and P(E)
# We can then use it to RECONSTRUCT p, the vectorized full-joint,
# using P=P(E|O)P(O)=P(O|E)P(E), as is done above.
