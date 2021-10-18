import networkx as nx
import numpy as np
import pandas as pd
import random
from itertools import chain, combinations
import numba as nb

from pgmpy.estimators import PC, HillClimbSearch, BicScore, K2Score
from pgmpy.factors.discrete import DiscreteFactor
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from gsq.ci_tests import ci_test_dis, ci_test_bin

from .utils.tools import generate_vals, parents, data_to_contexts, remove_cycles, cpdag_to_dags, dag_to_cpdag, generate_dag, generate_dag1, coming_in, v_structure, data_to_contexts, vars_of_context
from .utils.tools import test_anderson_k, test_epps, test_skl_divergence, reservoir_sample
from .utils.pc import estimate_cpdag, estimate_skeleton
from .mincontexts import minimal_context_dags
from .graphoid import graphoid_axioms

def learn_cpdag_pcpgmpy(data):
    _, p = data.shape
    cpdag_pgmpy = PC(pd.DataFrame(data, columns=[i+1 for i in range(p)]))
    cpdag = nx.DiGraph()
    cpdag.add_nodes_from([i+1 for i in range(data.shape[1])])
    cpdag.add_edges_from(cpdag_pgmpy.edges)
    #cpdag = remove_cycles(cpdag)
    
    return cpdag

def learn_cpdag_pcgithub(data, binary=False):
    if binary:
        ci_test = ci_test_bin
    else:
        ci_test = ci_test_dis
    # Get CPDAG skeleton
    (g, sep_set) = estimate_skeleton(indep_test_func=ci_test,
                                     data_matrix=data,
                                     alpha=0.01)

    # Get the CPDAG
    cpdag = estimate_cpdag(skel_graph=g, sep_set=sep_set)
    cpdag = nx.relabel_nodes(cpdag, lambda x: x+1)
    
    return cpdag

def learn_cpdag_hill(data):
    data_pd = pd.DataFrame(data, columns=[str(i+1) for i in range(data.shape[1])])
    cpdag_model=HillClimbSearch(data_pd)
    dag_pgmpy = cpdag_model.estimate(BicScore(data_pd), show_progress=False)
    dag_pgmpy=nx.relabel_nodes(dag_pgmpy, lambda x:int(x))
    dag = nx.DiGraph()
    dag.add_nodes_from([i+1 for i in range(data.shape[1])])
    dag.add_edges_from(dag_pgmpy.edges)
    cpdag = dag_to_cpdag(dag)
        
    return cpdag
    
def learn_cpdag(data, dag_method=None, binary=False):
    
    dag_methods = {"pcpgmpy":learn_cpdag_pcpgmpy,
               "pcgithub":lambda data: learn_cpdag_pcgithub(data, binary),
               "hill":learn_cpdag_hill}
    
    if dag_method is None:
        dag_method = list(dag_methods.keys())[1]
    else:
        assert dag_method in dag_methods.keys()

    return dag_methods[dag_method](data)

def all_mec_dags(data, method=None, binary=False):
    return cpdag_to_dags(learn_cpdag(data, method, binary))

def learn_cstree():
    pass


def contingency_table(value_dict, data):
    n, p = data.shape
    sizes = [val for val in value_dict.values()]
    u_shape = tuple(len(sizes[i]) for i in range(p))
    table= np.zeros(u_shape)
    for i in range(n):
        sample = tuple(data[i,:].astype(int))
        table[sample] +=1
    return table

def cstree_likelihood(x, u, tree, order):
    """ Compute likelihood of the sample x given the staging of the tree and
        parameters estimated from the data under current staging

    Args:
        x (np.array): sample to get the likelihood of
        order (list): ordering of the variables in the tree
        tree (nx.DiGraph): Tree to get the staging from
        data (np.array): Dataset to generate the contingency table

    """
    p = len(order)

    # Make ordering start from 0
    order_py = [o-1 for o in order]

    # u_C and u_x_C as defined in the paper
    u_C   = lambda C: \
        np.sum(u, axis=tuple([i for i in range(p) if i+1 not in C]))
    u_x_C = lambda x,C : u_C(C)[tuple(x)]

    # In self.data, the data is ordered from 0-p
    # Here we change the order according to order_py
    x = list(x)
    x_ordered = [x[i] for i in order_py]

    # Accumulator value for the likelihood
    prob = 1

    # For each level of the tree
    for level in range(1,p):
        # x_k below is x_k-1 in paper
        x_k = tuple((order[i], x_ordered[i]) for i in range(level))
        context_of_stage_x_k = tree.nodes[x_k].get("context", x_k)
        context_vars = [var for (var,_) in context_of_stage_x_k]

        # TODO Check if we really need to sort this
        next_var = order[level]
        CjUk = sorted(context_vars + [next_var])
        Cj   = sorted(context_vars)

        # Since the sample x is ordered from 0...p and
        # the variables in CjUk,Cj are in {1,...p}
        x_CjUk = [x[i] for i in range(p) if i+1 in CjUk]
        x_Cj   = [x[i] for i in range(p) if i+1 in Cj]

        numerator   = u_x_C(x_CjUk, CjUk)
        denominator = u_x_C(x_Cj,Cj)

        prob = prob*(numerator/denominator)
    return prob


def cstree_bic(value_dict, data, tree, order):
        """ Compute the BIC of a CStree with respect to some data

        Args:
            tree (nx.DiGraph): CStree
            order (list): Ordering of the variables for the tree above
            data (np.array): Dataset

        Note: The variable stages are a dictionary with keys being colors and values being the nodes belonging to that stage. The variable color_scheme is a dictionary with key value pairs node and color, where the color is white if the node belongs to a singleton stage.
        """
        n,p = data.shape
        # 1. Compute likelihood
        u = contingency_table(value_dict, data)
        log_mle = sum(list(
                       map(lambda i: np.log(cstree_likelihood(data[i,:], u, tree, order)), range(n))))

        # 2. Get the free parameters

        # Dictionary where key is the level and the value is the contexts of
        # stages in that level
        stages_per_level = cstree_to_stages(tree, order)

        # TODO Check use of order below
        free_params = sum([len(stages_per_level[i])*(len(value_dict[order[i-1]])-1)
                           for i in range(1,p)])

        return log_mle-0.5*free_params*np.log(n)


def cstree_to_stages(tree, order):
    """ Get the non-singleton stages from a learnt CStree

    Useful for model comparison and generating the CSI relations entailed by a CStree.

    Args:
        tree (nx.DiGraph): CStree to get stages from

    """
    # TODO Cache
    p = len(order)
    stages_per_level = {}
    for level in range(1, p):
        current_level_nodes = [n for n in tree.nodes
                            if nx.shortest_path_length(tree, "Root", n) == level]

        # Here, the context is a frozen set which allows stages_per_level
        # to be a set of sets, -1 is a dummy value
        stages_per_level[level] = {tree.nodes[n].get("context", -1)
                                    for n in current_level_nodes}.difference({-1})
        #print("cstreetostages",len(stages_per_level[level]))

    return stages_per_level

def stages_to_csirels(stages, order):
    """ Generate CSI relations from stages

    CSI rels are of the form (X_k: set, X_{k-1} \ C: set, set(), X_C: set)
    The stages are in a dictionary where the keys are the level of the
    tree and the values are a list of contexts. This is because a context
    and a level characterize a stage

    Args:
        stages (dict): Dictionary where keys are levels and values are a list with contexts
        order (list): Order of the CStree where the stages are from


    """
        
    csi_rels = []
    p = len(order)
    for level in range(1, p+1):
        X_k = {order[level]}
        # TODO Test when context is empty
        for context in stages[level]:
            X_C = set([var for (var,_) in context])
            X_k_minus_1 = set(order[:level])

            # TODO Is the empty set difference bug real
            if X_C == set():
                X_k_minus_1_minus_C = X_k_minus_1.copy()
            else:
                X_k_minus_1_minus_C = X_k_minus_1.difference(X_C)

            csi_rels.append((X_k, X_k_minus_1_minus_C, set(), context))
    return csi_rels

def dag_model(value_dict, dag, order):
    """ Get a DAG as a CStree with given order

    Args:
        dag (nx.DiGraph): DAG to convert to CStree
        order (list): Ordering for the CStree

    """
    #assert len(order)           == self.p # Doesnt hold for multinet
    #assert len(list(dag.nodes)) == self.p
    # TODO Assertion if order is compatible

    # Initialize empty graph for CStree
    cstree = nx.DiGraph()

    # Initialize color scheme

    level=0
    roots = (("Root",),)

    # For each level in the tree
    for level in range(1, len(dag.nodes)):
        # We want levels to start from 1 but the indexing starts from 0
        var = order[level-1]

        # Values taken by current variable
        vals = value_dict[var]

        # Nodes in current level
        current_level_nodes = tuple([(var, val) for val in vals])

        # If we are in first level, add edges from Root to first level
        if level == 1:
            edges = [("Root", (n,)) for n in current_level_nodes]
        # Else chain each outcome for variable in current level
        # to previous roots
        else:
            edges = [[(r, r+(n,)) for r in roots] for n in current_level_nodes]
            edges = list(chain.from_iterable(edges))

        # Add edges from current level to tree
        cstree.add_edges_from(edges)

        # These are the nodes in the current level which will be roots for next level
        roots = [j for (_,j) in edges]

        if dag:
            # Given the DAG we encode the contexts per node from it
            next_var = order[level]

            # Parents_G(level+1)
            pars = parents(dag, next_var)

            # If all variables prior to next_var are its parents
            # Then all stages are singleton, i.e. colored white
            # Thus we can skip this for current level
            if len(pars) == level:
                continue
            stage_contexts = generate_vals(pars, value_dict)

            for context in stage_contexts:
                # Special case when 2nd variable has no parents
                # It implies it is independent of first variable
                if level == 1 and pars == []:
                    stage_nodes = [n for n in list(cstree.nodes)[1:]
                                if n[0][0] == order[0]]
                # TODO Check line 197 in cstree.py
                else:
                    stage_nodes = [n for n in roots
                                if set(context).issubset(n)]
                for node in stage_nodes:
                    cstree.nodes[node]["context"]=frozenset(list(context))

    return cstree



def nonsingleton_stages(tree, nodes):
    """ Get the contexts of non-singleton stages in tree from a given list of nodes

    """
    existing_context_nodes = list(filter
                                    (lambda node: True if tree.nodes[node].get("context",None) is not None else False, nodes))
    existing_contexts = set(tree.nodes[node]["context"] for node in existing_context_nodes)
    return existing_contexts


def num_stages(tree, order):
    """ Get the number of non-singleton stages in a given CStree

    """
    p = len(order)
    stages = cstree_to_stages(tree, order)
    return sum([len(stages[i]) for i in range(1,p)])


def random_cstree(value_dict, order, ps, Ms):
    """ Generate a random CStree with variables and outcomes defined by value_dict
        
        The CStree is generated by choosing 2 random nodes Ms[l-1] times, and merging them
        to the same stage with probability ps[l-1] for level l

        Args:
            order (list): Order of the random CStree being generated
            ps (list): List containing the probabilities of merging 2 randomly selected nodes per level
            Ms (list): List containing number of times to select 2 random nodes per level

    """
    # TODO Look into moving contextn1,contextn2 inside the if merge block
    assert len(order)==len(list(value_dict.keys()))
    dag = generate_dag(len(order), 1)
    dag = nx.relabel_nodes(dag, lambda i: order[i-1])
    tree = dag_model(value_dict, dag, order)

    p = len(order)
    for level in range(1, p):
        #print("level", level)
        # Generate a context randomly

        for _ in range(Ms[level-1]):
            current_level_nodes = [n for n in tree.nodes
                            if nx.shortest_path_length(tree, "Root", n) == level]

            # Choose 2 random nodes
            random_node_pair = random.sample(current_level_nodes, 2)
            r_node1, r_node2 = random_node_pair[0], random_node_pair[1]
            context_n1 = tree.nodes[r_node1].get("context", r_node1)
            context_n2 = tree.nodes[r_node2].get("context", r_node2)

            # Merge their stages with probability ps[level-1]
            merge = True if np.random.uniform() < ps[level-1] else False

            if merge:
                common_subcontext = set(context_n1).intersection(set(context_n2))

                new_nodes = [n for n in current_level_nodes
                                if common_subcontext.issubset(set(n))]

                # Existing contexts of nodes above if they exist
                existing_contexts = nonsingleton_stages(tree, new_nodes)

                if existing_contexts!=set():
                    # If such contexts exist, the common context is the total intersection
                    common_subcontext = common_subcontext.intersection(*list(existing_contexts))

                    new_nodes = [n for n in current_level_nodes
                                if common_subcontext.issubset(set(n))]

                for node in new_nodes:
                    tree.nodes[node]["context"]=frozenset(common_subcontext)

    # Generate distribution with separate function
    tree_distr = tree_distribution(value_dict, tree, order)
    return tree, tree_distr


def tree_distribution(value_dict, tree, order):
    """ Geneate a random probability distribution for each outcome of a CStree
        according to the staging of the CStree

    Args:
        tree (nx.DiGraph): CStree whose staging is used to construct distribution
        order (list): Ordering of the CStree

    """

    # Note list below excludes contexts involving the last variable
    leaves = [n for n in tree.nodes
                if nx.shortest_path_length(tree, "Root", n) == len(order)-1]
    # All outcomes must include the possibilities of the last variable
    # which are however excluded in our CStree graph
    cardinalities = [len(value_dict[i]) for i in order]
    tree_distr = DiscreteFactor(["X"+str(i) for i in order],
                            cardinalities,
                            np.ones(np.prod(cardinalities))/np.prod(cardinalities))

    # Each stage encodes a distribution for the variable that comes
    # after the variable that represents the node of that stage
    distrs = {}

    first_var_outcomes  = len(value_dict[order[0]])

    first_probabilities = np.random.dirichlet(
        [10*first_var_outcomes if np.random.rand()<0.5
            else 0.5*first_var_outcomes for _ in range(first_var_outcomes)])
    first_probabilities = (1/first_var_outcomes)*np.ones(first_var_outcomes)

    # Gathering all probabilities to see if they sum to 1
    prs = []

    # For each outcome from the tree
    # TODO This could be done by appending nodes onto
    # previous path instead of generating paths this way
    for leaf in leaves:
        path = nx.shortest_path(tree, "Root", leaf)
        for val in value_dict[order[-1]]:
            # Adding last (variable, value) pair to the path
            actual_path = path+[path[-1]+((order[-1],val),)]
            # Probability of first outcome, path[1] is actual first node
            # since first is Root, [0] to access the node itself since
            # it is a tuple of tuples, -1 for the actual value
            pr = first_probabilities[path[1][0][-1]]
            # For each next node, get probabilities according to staging
            # or create them if encountering the staging for first time
            # A stage of a node is determined uniquely by the level of the
            # node in the tree and the context which fixes the stage

            # Skipping over first node which is root
            # since that value is taken into pr, skipping last
            # since we need nodes to get non-singleton stages,
            # from which we get the probability values for the
            # level afterwards and the last node is always
            # in the singleton stage
            #for level in range(1,self.p):
            #    node = actual_path[level]
            for node in actual_path[1:-1]:
                #print("Node is", node)
                # Next variable and its outcomes
                level = len(node)
                # Below is next variable since Python has 0 indexing
                next_var = order[level]

                # Edges coming out of node in level i represent
                # outcomes of variable in level i+1
                outcomes = len(value_dict[next_var])
                #print("var and no oucome", var, outcomes)

                #level = len(node)

                context = frozenset(tree.nodes[node].get("context",node))
                if distrs.get((level, context), None) is None:
                    alpha = [10 if np.random.rand()<0.5
                                else 0.5 for _ in range(outcomes)]
                    distrs[(level, context)]=np.random.dirichlet(alpha)
                # We need the next outcome value of the path
                # First -1 gets the last node in the current path
                # Next  -1 gets the value from that node

                next_outcome = actual_path[level+1][-1][-1]
                #print(level, next_outcome, next_var)
                pr = pr*distrs[(level, context)][next_outcome]


            # Adding probabilities at this level otherwise you miss the last outcome
            actual_leaf = actual_path[-1]
            kwargs = {"X"+str(var):val for (var,val) in actual_leaf}
            tree_distr.set_value(pr, **kwargs)
            prs.append(pr)

    # Final check to make sure all outcomes sum to 1
    # print("All probabilities sum to ",sum(prs))
    return tree_distr

def nodes_with_context(context, nodes):
    """ Get the nodes from a list of nodes which contain a specified context. Here, nodes are of the form ((var_i,val_k), (val_i+1,val_l))

    """
    node_contains_context = lambda n: True if set(context).issubset(set(n)) else False
    
    return list(filter(node_contains_context, nodes))


def generate_contexts(value_dict, vars, size):
    """ Generate all possible contexts from a given list of variables with a given size


    """
    var_combos = list(combinations(vars, size))

    contexts = []
    for var_combo in var_combos:
        temp = generate_vals(list(var_combo), value_dict)
        for context in temp:
            contexts.append(context)
    return contexts

def cstree_predict(value_dict, tree, order, sample, i, data):
    """ Find P(Xi|X_[p]\ {i})

    By Bayes theorem, this is the probability of observing the sample 
    i.e. P(X1=x1,...,Xp=xp) divided by the probability of the marginal i.e.
    P(X1=x1, ..., Xi-1=xi-1, Xi+1=xi+1,...,Xp=xp) is the same as
    sum_j P(X1=x1, ..., Xi-1=xi-1,Xi=j Xi+1=xi+1,...,Xp=xp) where xk are from sample

    """
    numerator    = cstree_likelihood(sample, order, tree, data)
    order_of_var = order.index(i)
    samples      = [np.array(list(sample[:order_of_var])+[val]+list(sample[order_of_var+1:]))
                    for val in value_dict[i]]
    # sum_j P(X1=x1,...,Xi-1=xi-1,Xi+1=xi+1,...,Xn|Xi=j)P(Xi=j)
    denominator  = sum([cstree_likelihood(s, order, tree, data) for s in samples])
    return numerator/denominator


def hsbm_dags(k, order, ps):
    # Generate k random DAGs for HSBM where the first
    # variable in the order is contains the minimal contexts
    dags = []
    p = len(order)
    for i in range(k):
        dag = generate_dag1(p-1, ps[i])
        dag = nx.relabel_nodes(dag, lambda i: order[1:][i-1])
        assert order[1:] in list(nx.all_topological_sorts(dag))
        dags.append(dag)
    return dags


def generate_hsbm(value_dict, k, order, dags):
        """ Construct a Hypothesis-Specific Bayesian Multinet (HSBM)
        from the list of DAGs

        This is done by generating a DAG model for each of the outcomes of the 
        first variable, which is k

        Args:
            k (int): Number of outcomes for first variable in the HSBM
            order (list): Ordering of the variables 
            ps (list): Probabilities of including an edge for each of the k DAGs

        """
        
        # DAG CStree
        # Change outcomes for first variable to accomodate for the k outcomes
        value_dict[order[0]]=[i for i in range(k)]
        p = len(order)

        # Empty graph for CStree
        #tree = nx.DiGraph()
        tree = dag_model(value_dict, generate_dag(p, 1), order)

        # Add initial edges from Root to first k outcomes
        # tree.add_edges_from([("Root",((order[0], i),)) for i in range(k)])
        
        ss=0
        # For k^th outcome in first variable, create a CStree for a random
        # DAG and connect the root of it to leaf corresponding to k^th outcome
        for i in range(k):
            # Generate a random dag with p-1 nodes where a probability
            # of an edge occuring is ps[i]
            dag = dags[i]

            # Create the CStree for the random DAG above
            dag_cstree = dag_model(value_dict, dag, order[1:])

            for node in dag_cstree:
                # Attach the CStree for the DAG to the CStree
                # for the HSBM
                if node == "Root":
                    continue
                actual_node = ((order[0], i),)+node
                # Since singleton contexts are simply not stored
                # we need to fix the first outcome context to
                # non-singleton stages
                if dag_cstree.nodes[node].get("context", None) is not None:
                    tree.nodes[actual_node]["context"] = frozenset( ((order[0], i),)).union(dag_cstree.nodes[node]["context"])

            ss+=num_stages(dag_cstree, order)

        tree_distr = tree_distribution(value_dict, tree, order)
        return tree, tree_distr

def random_hsbm(value_dict, k, order, ps):
    dags = hsbm_dags(order, k, ps)
    tree, distr = generate_hsbm(value_dict, order, k, dags)
    return tree, distr, dags


def cstree_oracle(value_dict, distribution, var, order, context):
    vars_to_marginalize    = order[order.index(var)+1:]
    vars_to_marginalize    = ["X"+str(var)
                                for var in vars_to_marginalize]
    context_to_marginalize = [("X"+str(var),val)
                                for (var,val) in context]
    distribution.marginalize(vars_to_marginalize)
    distribution.reduce(context_to_marginalize)

    rank =  np.linalg.matrix_rank(distribution.values.reshape(len(value_dict[var]),-1))

    if rank == 1:
        indep=True
    else:
        indep=False
    return indep

def cstree_node_test(value_dict, data, var, context_n1, context_n2, test):
    data_n1 = data_to_contexts(data, list(context_n1), var)
    data_n2 = data_to_contexts(data, list(context_n2), var)

    merge = test(data_n1, data_n2, value_dict[var])
    return merge


def cstree_ci_test(value_dict, data, var, B, context):
    context_vars = vars_of_context(context)
    # For each context, run the test to see if we can merge them
    data_to_context = data_to_contexts(data, context)
    B_prime = [var for var in B if var not in context_vars]
    temp_cond_set = set()
    times_independent = []

    binary = True if all([len(vals)==2 for vals in value_dict.values()]) else False
    if binary:
        ci_test=ci_test_bin
    else:
        ci_test=ci_test_dis
        
    for b in B_prime:
        p = ci_test(data_to_context, var-1, b-1, temp_cond_set)
        if p<0.05:
            # Not independent then move on to next context
            continue
        else:
            times_independent.append(True)
            temp_cond_set = temp_cond_set.union({b-1})
    if len(times_independent)==len(B):
        indep=True
    else:
        indep=False
    return indep


def merge_nodes(common_subcontext, nodes_l, cstree):
    new_nodes = [n for n in nodes_l if common_subcontext.issubset(set(n))]

    # If there are non-singleton nodes trapped in between
    # we need to get the common context of them and the
    # new common context we just learnt about
    existing_contexts = nonsingleton_stages(cstree, new_nodes)

    # If we do have such contexts, we get the new common context
    if existing_contexts!=set():
        common_subcontext = common_subcontext.intersection(*list(existing_contexts))
        new_nodes = [n for n in nodes_l if common_subcontext.issubset(set(n))]
        for node in new_nodes:
            cstree.nodes[node]["context"]=frozenset(common_subcontext)

def cstree_learn_obs_nodebased(value_dict: dict[int, list[int]], dag: nx.DiGraph,
                               order: list[int], data: np.ndarray,
                               test=lambda s1,s2,o: test_anderson_k(s1,s2,o),
                               oracle=None, node_ratios=None):
    
    cstree = dag_model(value_dict, dag, order)
    p = len(order)

    for level in range(1, p):
        var = order[level]

        nodes_l = [n for n in cstree.nodes
                    if nx.shortest_path_length(cstree, "Root", n) == level]

        nodes_to_compare = combinations(nodes_l, 2)

        if node_ratios:
            to_sample=int(node_ratios[level-1]*len(nodes_l))
            nodes_to_compare = reservoir_sample(nodes_to_compare, to_sample)

        for n1,n2 in nodes_to_compare:
            context_n1 = cstree.nodes[n1].get("context",n1)
            context_n2 = cstree.nodes[n2].get("context",n2)

            if context_n1 == context_n2:
                # Already in the same context so skip testing
                continue
            
            else:
                # In different contexts so we test
                common_subcontext = set(context_n1).intersection(set(context_n2))
                if oracle:
                    # Using the oracle version
                    merge = cstree_oracle(value_dict, oracle.copy(),
                                  var, order, common_subcontext)
                else:
                    # Using the ci testing version
                    merge = cstree_node_test(value_dict, data, var, context_n1, context_n2, test)
                if merge:
                    cstree = merge_nodes(common_subcontext, nodes_l, cstree)
    return cstree

def cstree_learn_obs_contextbased(value_dict: dict[int, list[int]], dag:nx.DiGraph,
                                  order: list[int], data: np.ndarray,
                                  oracle=None, context_limit=None):
    
    cstree = dag_model(value_dict, dag, order)
    p = len(order)

    for level in range(1, p):
        var = order[level]
        nodes_l = [n for n in cstree.nodes
                    if nx.shortest_path_length(cstree, "Root", n) == level]
        
        if context_limit is None:
            context_size_current_level = level
        else:
            context_size_current_level = context_limit[level]+1

            for context_size in range(0, context_size_current_level):
            # For each context var size from 0 inclusive
            # TODO Check if level is good to use or level+1 in line below
                B = order[:level] # B in X_A _||_ X_B | X_C=x_c, note that A={var}
                contexts = generate_contexts(value_dict, B, context_size)

                for context in contexts:
                    if oracle:
                        # Using the oracle version
                        merge = cstree_oracle(value_dict, oracle.copy(),
                                              var, order, context)
                    else:
                    # Using the ci testing version
                        merge = cstree_ci_test(value_dict, data, var, B, context)
                    if merge:
                        cstree = merge_nodes(context, nodes_l, cstree)
    return cstree

def learn_cstree_obs_mcunknown(value_dict, data, dag, order, node_based=True, **kwargs):
    if node_based:
        cstree = cstree_learn_obs_nodebased(value_dict, dag, order, data, **kwargs)
    else:
        cstree = cstree_learn_obs_contextbased(value_dict, dag, order, data, **kwargs)
    return cstree

def learn_skeleton():
    pass

def learn_mcdags(data, minimal_contexts,
                 orders=None, max_degrees=None, oracle=None):
    # Max degree - list of maximum degrees for each mc graph
    n,p = data.shape
    m = len(minimal_contexts)
    
    if max_degrees is None:
        max_degrees = p*[p-1]

    for i, mc in enumerate(minimal_contexts):
        ci_rels = frozenset()
        C = vars_of_context(mc)
        vars = [i+1 for i in range(p) if i+1 not in C]
        dag_u = nx.complete_graph(vars)

        max_degree = max_degrees[i]

        for j in range(max_degree):

            for k,l in dag.edges:
                neighbors = list(set(list(dag.neighbors(k))).difference({k,l}))
                if len(neighbors)>=j:
                    subsets = [set(i) for i in combinations(neighbors, j)]

                    for subset in subsets:
                        indep=False
                        if oracle:
                            pass
                        else:
                            pass

                        if indep:
                            dag_u.remove_edges_from([(k,l)])
                            if ({k},{l},subset) not in ci_rels:
                                ci_rels = ci_rels.union(({k},{l},subset))
                            
                        
                            

                               

                    # TODO Read PC algorithm, partition this. Do more experiments then writee


def generate_dag_order_pairs(data, use_dag_ci_rels=True, orders=None, **kwargs):
    dag_order_pairs= []
    _,p = data.shape
    if use_dag_ci_rels and orders:
        dags = all_mec_dags(data, **kwargs)
        for order in orders:
            # For each possible true order we give, we want to make sure
            # the all DAGs in the learnt MEC is consistent with the ordering
            found_dag = False
            for dag in dags:
                if order in list(nx.all_topological_sorts(dag)):
                    dag_order_pairs.append((dag, order))
                    found_dag = True
                    break
            if not found_dag:
                # If none of the DAGs in the MEC are consistent with this
                # order, we put the full DAG
                dag = generate_dag(p, 1)
                dag = nx.relabel_nodes(dag, lambda i: order[i-1])
                dag_order_pairs.append((dag, order))
                
        assert  len(dag_order_pairs)==len(orders)
            
    elif use_dag_ci_rels and not orders:
        dags = all_mec_dags(data, **kwargs)
        for dag in dags:
            orders = nx.all_topological_sorts(dag)
            for order in orders:
                dag_order_pairs.append((dag, order))
    
    elif not use_dag_ci_rels and orders:
        for order in orders:
            # If we do not the DAG CI rels we put the full DAG
            full_dag = generate_dag(p, 1)
            full_dag = nx.relabel_nodes(dag, lambda i: order[i-1])
            dag_order_pairs.append((full_dag, order))
            
    elif not use_dag_ci_rels and not orders:
        dags = all_mec_dags(data, **kwargs)
        full_dag = generate_dag(p, 1)
        full_dag = nx.relabel_nodes(full_dag, lambda i: order[i-1])
        for dag in dags:
            orders = nx.all_topological_sorts(dag)
            for order in orders:
                dag_order_pairs.append((full_dag, order))

    return dag_order_pairs
        
    
def learn_cstree_obs(value_dict, data, criteria="bic",
                     orders=None, use_dag_ci_rels=True, **kwargs):
    dag_order_pairs = generate_dag_order_pairs(data, use_dag_ci_rels, orders, **kwargs)

    best_score   = float("-inf")
    best_cstrees = []

    for dag, order in dag_order_pairs:
        cstree = learn_cstree_obs_mcunknown(value_dict, data, dag, order, **kwargs)

        # select best trees and order based on criteria
        assert criteria in ["bic", "minstages"]
        if criteria=="bic":
            criteria_func = lambda tree, order: cstree_bic(value_dict, data,tree, order)
        elif criteria=="minstages":
            criteria_func = lambda tree, order: -1*(num_stages(tree, order))
        else:
            raise ValueError("Criteria undefined")

        criteria_score = criteria_func(cstree, order)
        if criteria_score > best_score:
            best_score = criteria_score
            best_cstrees = [(cstree, order)]
        elif criteria_score == best_score:
            best_cstrees.append((cstree, order))

    return best_cstrees
            

def visualize():
    pass
