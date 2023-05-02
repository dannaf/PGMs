#!/usr/bin/env python3

'''This script provides the computational verification for the proof by example claim in Theorem 6.3 by exhaustively demonstrating that the full joint distribution in Table 6.4 satisfies all of the marginal and conditional dependences and independences of the 6-cycle network as shown in Figure 6.2.'''

# Get the joint distribution
P # from 6-cycle_CBN_demo.py

# Show that the system is a valid joint probability distribution
np.abs(P.sum() - 1).max() < 1e-10
(P >= 0).all()

# Show that the system satisfies the relations in a 6-cycle graph

# Show that all the random variables are marginally dependent, i.e., not marginally independent
n = 6
np.array([(marginalize(P, [x])*marginalize(P, [y]) - marginalize(P, [x, y])).max() < 1e-10 for x, y in itertools.product(range(n), range(n)) if x != y]).any() == False

# Show that neighboring random variables are conditionally dependent, i.e., not conditionally independent
np.array([[np.abs(conditionalize(P, [x], [neighbors(x, n)[neighbor_idx]]) - marginalize(P, [x])).max() > 1e-10 for x in range(n)] for neighbor_idx in [0, 1]]).all()

# Show that the neighboring random variables are conditionally independent given the center node between them
np.array([[np.abs(conditionalize(P, [x, y], [neighbors(x, n)[0], neighbors(x, n)[1]]) - conditionalize(P, [x], [neighbors(x, n)[0], neighbors(x, n)[1]])*conditionalize(P, [y], [neighbors(x, n)[0], neighbors(x, n)[1]])).max() < 1e-10 for y in list(set(range(n)).difference(set([x, neighbors(x, n)[0], neighbors(x, n)[1]])))] for x in range(n)]).all()


# Show that the system is not strictly positive
(P == 0).any()
