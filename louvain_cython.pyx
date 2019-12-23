import numpy as np
cimport numpy as np
import networkx as nx
DTYPE = np.double
ctypedef np.double_t DTYPE_t
ctypedef np.int_t DTYPE_int

def modularity_classic(np.ndarray[DTYPE_t, ndim=2] M, np.ndarray[DTYPE_int, ndim=1] assignments):
    """
    Calculates the modularity of an unsigned weighted network
    Parameters
    ----------
    M : array_like
        p by p adjacency matrix representing the graph
    assignments : array_like
        p length vector containing node community assignments
    Returns
    -------
    modularity : float
        Value of the modularity
    """
    cdef int p = M.shape[0]
    K = M.sum(axis=0)
    cdef float Q = 0
    cdef float C_norm = M.sum()
    for i in range(p):
        for j in range(p):
            if assignments[i] != assignments[j]:
                continue

            Q += M[i,j] - ((K[i] * K[j])/C_norm)

    return Q/(C_norm)


def modularity_signed(np.ndarray[DTYPE_t, ndim=2] M, np.ndarray[DTYPE_int, ndim=1] assignments):
    """
    Calculates the modularity of a signed weighted network
    Parameters
    ----------
    M : array_like
        p by p adjacency matrix representing the graph
    assignments : array_like
        p length vector containing node community assignments
    Returns
    -------
    modularity : float
        Value of the modularity
    """
    p = M.shape[0]
    M_pos = M.copy()
    M_pos[M_pos < 0] = 0
    M_neg = M.copy()
    M_neg[M_neg > 0] = 0
    M_neg = - M_neg

    K_pos = M_pos.sum(axis=0)
    K_neg = M_neg.sum(axis=0)

    tot_pos = M_pos.sum()
    tot_neg = M_neg.sum()

    Q_pos = 0
    if tot_pos > 0:
        for i in range(p):
            for j in range(p):
                if assignments[i] == assignments[j]:
                    Q_pos += M_pos[i, j] - (K_pos[i] * K_pos[j]/tot_pos)
    if tot_pos == 0:
        Q_pos = 0
    else:
        Q_pos = Q_pos/tot_pos

    Q_neg = 0
    if tot_neg > 0:
        for i in range(p):
            for j in range(p):
                if assignments[i] == assignments[j]:
                    Q_neg += M_neg[i, j] - (K_neg[i] * K_neg[j]/tot_neg)

    if tot_neg == 0:
        Q_neg = 0
    else:
        Q_neg = Q_neg/tot_pos

    Q = tot_pos / (tot_pos + tot_neg) * Q_pos - tot_neg / (tot_pos + tot_neg) * Q_neg

    return Q

def modularity_diff_correlation(np.ndarray[DTYPE_t, ndim=2] M, int i, np.ndarray[DTYPE_int, ndim=1] assignments, int community):
    """
    Calculates the gain in modularity of taking node i from an isolated community
    into the community specified for the null model of a correlation
    network

    Parameters
    ----------
    M : array_like
        p by p adjacency matrix representing the graph
    i : integer
        index of the node being moved
    assignments : array_like
        p length vector containing node community assignments
    community : integer
        index of the community node i is being moved into
    Returns
    -------
    modularity_diff : float
        change in modularity
    Notes
    -----
    i must not be assigned to a community in the community vector - assign it to -1 community
    """
    ind = assignments == community
    m = M.sum()
    # Avoid a NaN
    if m == 0:
        return 0
    return M[i, ind].sum()#/m

def modularity_correlation(np.ndarray[DTYPE_t, ndim=2] M, np.ndarray[DTYPE_int, ndim=1] assignments):
    """
    Calculates the modularity of a correlation network
    Parameters
    ----------
    M : array_like
        p by p adjacency matrix representing the graph
    assignments : array_like
        p length vector containing node community assignments
    Returns
    -------
    modularity : float
        Value of the modularity
    """
    cdef int p = M.shape[0]
    cdef float Q = 0
    cdef float C_norm = M.sum()
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            if assignments[i] != assignments[j]:
                continue

            Q += M[i,j] 

    return Q/(C_norm)

def modularity_market_mode(np.ndarray[DTYPE_t, ndim=2] M, np.ndarray[DTYPE_int, ndim=1] assignments):
    """
    Calculates the modularity of a correlation network with the presence of a market mode
    Parameters
    ----------
    M : array_like
        p by p adjacency matrix representing the graph
    assignments : array_like
        p length vector containing node community assignments
    Returns
    -------
    modularity : float
        Value of the modularity
    """
    cdef int p = M.shape[0]
    cdef float Q = 0
    cdef float C_norm = M.sum()

    eigs, eigv = np.linalg.eig(M)
    max_ind = eigs.argmax()
    C_mode = eigs[max_ind] * eigv[max_ind] @ eigv[max_ind].T
    for i in range(p):
        for j in range(p):
            if i == j:
                val = 1
            else:
                val = 0
            if assignments[i] != assignments[j]:
                continue

            Q += M[i,j] - C_mode[i, j] - val 

    return Q/(C_norm)


def modularity_diff(np.ndarray[DTYPE_t, ndim=2] M, int i, np.ndarray[DTYPE_int, ndim=1] assignments, int community):
    """
    Calculates the gain in modularity of taking node i from an isolated community into
    the community specified for an unsigned weighted graph

    Parameters
    ----------
    M : array_like
        p by p adjacency matrix representing the graph
    i : integer
        index of the node being moved
    assignments : array_like
        p length vector containing node community assignments
    community : integer
        index of the community node i is being moved into
    Returns
    -------
    modularity_diff : float
        change in modularity
    Notes
    -----
    i must not be assigned to a community in the community vector - assign it to -1 community
    """
    ind = assignments == community
    m = M.sum()
    # Avoid a NaN
    if m == 0:
        return 0

    sum_in = M[ind, :][:, ind].sum()/2
    sum_tot = M[ind, :].sum()
    k_i = M[i, :].sum()
    k_in = M[i, ind].sum()
    return (sum_in + 2 * k_in)/(m) - ((sum_tot + k_i)/(m))**2 - sum_in/(m) + (sum_tot/(m))**2 + (k_i/(m))**2

def modularity_diff_signed(np.ndarray[DTYPE_t, ndim=2] M_pos, np.ndarray[DTYPE_t, ndim=2] M_neg, int i, np.ndarray[DTYPE_int, ndim=1] assignments, int community):
    """
    Calculates the gain in modularity of taking node i from an isolated community into
    the community specified for a signed weighted graph

    Parameters
    ----------
    M_pos : array_like
        p by p adjacency matrix representing the positive edges in the graph
    i : integer
        index of the node being moved
    assignments : array_like
        p length vector containing node community assignments
    community : integer
        index of the community node i is being moved into
    Returns
    -------
    modularity_diff : float
        change in modularity
    Notes
    -----
    i must not be assigned to a community in the community vector - assign it to -1 community
    """
    pos_gain = modularity_diff(M_pos, i, assignments, community)
    neg_gain = modularity_diff(M_neg, i, assignments, community)

    w_pos = M_pos.sum()
    w_neg = M_neg.sum()

    return (w_pos / (w_pos + w_neg)) * pos_gain - (w_neg / (w_pos + w_neg)) * neg_gain

def run_one_level(np.ndarray[DTYPE_t, ndim=2] M, int signed=False, int correlation=False):
    """
    Runs the first phase of the Louvain community detection algorithm for a weighted graph, returns a set of assignments
    for each node of the graph. 

    Parameters
    ----------
    M : array_like
        p by p adjacency matrix representing the graph
    signed : bool (optional, default=False)
        If the graph is signed or not
    correlation : bool (optional, default=False)
        If the graph is a correlation network
    Returns
    -------
    assignments : array_like
        p length vector with what community a node has been assigned to
    """
    cdef int p = M.shape[0]
    assignments = np.arange(p)
    communities = set(range(p))
    #run = True
    # we use this to count how many nodes we've gone without
    # updating - if we cycle all the way round we probably can't
    # improve anymore
    cdef int no_not_updated = 0
    #nodes = 
    modified = True

    if signed:
        pos_ind = M > 0
        neg_ind = M < 0
        M_pos = M.copy()
        M_pos[neg_ind] = 0
        M_neg = M.copy()
        M_neg[pos_ind] = 0
        M_neg[neg_ind] = -M_neg[neg_ind]

    # This stores whether we need to recalculate modularity for the network
    # i.e. has something changed on that run through round
    while modified:
        modified = False
        nodes = np.random.choice(p, size=p)
        for i in range(p):
            ind = nodes[i]
            max_diff_i = -1
            max_diff = 0
            connected_to = M[ind, :] != 0
            communities_to_consider_set = list(set(assignments[connected_to]))
            communities_to_consider = np.random.choice(communities_to_consider_set, size=len(communities_to_consider_set))

            old_com = assignments[ind]

            # Remove the node from it's community
            assignments[ind] = -1

            if signed:
                removal_cost = -modularity_diff_signed(M_pos, M_neg, ind, assignments, old_com)
            elif correlation:
                removal_cost = -modularity_diff_correlation(M, ind, assignments, old_com)
            else:
                removal_cost = -modularity_diff(M, ind, assignments, old_com)
                #assignments[ind] = old_com
                #old_mod = modularity_signed(M, assignments)
                #assignments[ind] = -1

            run_through_modified = False
            for com in communities_to_consider:
                if signed:
                    change = modularity_diff_signed(M_pos, M_neg, ind, assignments, com)
                elif correlation:
                    change = modularity_diff_correlation(M, ind, assignments, com)
                else:
                    change = modularity_diff(M, ind, assignments, com)

                diff = change + removal_cost
                if max_diff < diff:
                    max_diff = diff
                    max_diff_i = com
            if max_diff > 0 and max_diff_i != -1:
                no_not_updated = 0
                modified = True
                assignments[ind] = max_diff_i
            else:
                # If there isn't a better community to put it in, put it back
                no_not_updated += 1
                assignments[ind] = old_com

    communites = {com:i for i,com in enumerate(set(assignments))}

    # remap communities into a range from 0-number of communities
    for i in range(p):
        assignments[i] = communites[assignments[i]]

    return assignments

def induced_graph(np.ndarray[DTYPE_t, ndim=2] M, np.ndarray[DTYPE_int, ndim=1] assignments, labels, first):
    """
    Folds all the communities into their own node - phase 2 of the Louvain algorithm

    Parameters
    ----------
    M : array_like
        p by p adjacency matrix of the graph
    assignments : array_like
        p length vector containing node community assignments
    labels : either a list if this is the first run or a dictionary if we've already
             folded a network in
        Contains what nodes the communities correspond to if we've already run the 
        algorithm once. This bit isn't pretty.
    first : bool
        If this is the first run through
    Returns
        tuple (new_M, folded)

        new_M contains the new adjacency matrix and folded contains a dictionary 
        which nodes each community belongs to
    """
    p = M.shape[0]
    communities = list(set(assignments))
    no_communities = len(communities)
    new_M = np.zeros((no_communities, no_communities))
    # Folded contains the nodes that the community contains
    folded = {}
    if first:
        labels = np.array(labels)
    for i,com in enumerate(communities):
        ind = assignments == com
        # Get all the connections within the community
        self_weight = M[ind, :][:, ind].sum()
        new_M[i, i] = self_weight
        if first:
            labels = np.array(labels)
            folded[i] = set(labels[np.where(ind)[0]])
        else:
            folded[i] = set()
            nodes_in_community = np.where(ind)[0]
            for node in nodes_in_community:
                folded[i] = folded[i] | labels[node]
        for j,com_2 in enumerate(communities):
            if com == com_2:
                continue
            ind_2 = assignments == com_2
            #M_com = M.copy()
            M_com = M[ind, :]
            M_com = M_com[:, ind_2]
            weight = M_com.sum()
            new_M[j, i] = weight
            new_M[i, j] = weight

    return new_M, folded
    
def run_louvain_nx(G, nodes=None, int max_iter=100, int signed=False, int correlation=False):
    """
    Runs the Louvain community detection algorithm on a networkx graph
    Parameters
    ----------
    G : networkx graph
        Graph to run the algorithm on
    max_iter : int (optional, default=5)
        Maximum number of iterations of the algorithm to run
    a : int (optional, default=1)
        Description of what a does
    signed : bool (optional, default=False)
        If the graph is signed or not
    correlation : bool (optinal, default=False)
        If the graph is a correlation network

    Returns
        tuple (dict, dict)
        First dict contains the best possible community assignments
        Second dict contains the entire output if you wish to resolve 
        communities of multiple scales
    """
    assignments_dct = {}
    i = 0
    new_G = G.copy()
    old_mod = -np.inf
    assignments_dct = {}

    if correlation and signed:
        raise ValueError("Both correlation and signed cannot be true")

    if nodes is None:
        node_labels = list(G.nodes())
    else:
        node_labels = nodes

    while True:
        M = nx.to_numpy_array(new_G)#, nodelist=nodes)

        assignments = run_one_level(M, signed=signed, correlation=correlation)
        # Fold them into an induced graph

        M, node_labels = induced_graph(M, assignments, node_labels, i==0)
        if signed:
            mod = modularity_signed(M, assignments)
        elif correlation:
            mod = modularity_correlation(M, assignments)
        else:
            mod = modularity_classic(M, assignments)
        new_G = nx.from_numpy_array(M)
        #new_G = nx.relabel_nodes(new_G, folded)
        i += 1
        print(i)
        print(mod)
        assignments_dct[i] = node_labels
        # Quit if we can't increase the modularity
        # or if we've run out of iterations
        # or if there is one giant community
        if mod <= old_mod: #or i > max_iter: #or len(set(node_labels))==1:
            break
        old_mod = mod

    return assignments_dct[i-1], assignments_dct


