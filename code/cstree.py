import networkx as nx
import numpy as np
import pandas as pd
import random
from itertools import chain, combinations
from pgmpy.estimators import PC, HillClimbSearch, BicScore, K2Score
from pgmpy.factors.discrete import DiscreteFactor
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from gsq.ci_tests import ci_test_dis, ci_test_bin

from .utils.tools import generate_vals, parents, data_to_contexts, remove_cycles, cpdag_to_dags, dag_to_cpdag, generate_dag, coming_in, v_structure, data_to_contexts, vars_of_context
from .utils.pc import estimate_cpdag, estimate_skeleton
from .mincontexts import minimal_context_dags
from .graphoid import graphoid_axioms

class DAG(object):
    """ Class acting as a wrapper for pure (i.e. not CStree) DAG models 

    """
    def __init__(self):
        self.learn_algs =  ["pcpgmpy", "pcgithub", "hill"]

    def learn_obs(self, data, method=None):
        if method is None:
            method = self.learn_algs[0]

        _, p = data.shape

        if method == "pcpgmpy":
            cpdag_model = PC(pd.DataFrame(data, columns=[i+1 for i in range(p)]))
            cpdag_pgmpy = cpdag_model.estimate(return_type="cpdag", show_progress=False)
            cpdag = nx.DiGraph()
            cpdag.add_nodes_from([i+1 for i in range(p)])
            cpdag.add_edges_from(list(cpdag_pgmpy.edges()))
            cpdag= remove_cycles(cpdag)
        
        if method=="pcgithub":
            # If the data is binary we do a different test in the PC algorithm
            # binary_data = True if all(list(map(lambda f: True if len(f)==2 else False, list(self.val_dict.values())))) else False
            binary_data = True if all([True if len(np.unique(data[:,i]))==2 else False for
                                     i in range(p)]) else False

        # Set the test to get CPDAG
            if binary_data:
                pc_test = ci_test_bin
                #pc_test=None
            else:
                pc_test = ci_test_dis
                #pc_test=None
            
            
        # Get CPDAG skeleton
            (g, sep_set) = estimate_skeleton(indep_test_func=pc_test,
                                             data_matrix=data,
                                             alpha=0.01)

        # Get the CPDAG
            cpdag = estimate_cpdag(skel_graph=g, sep_set=sep_set)
            cpdag = nx.relabel_nodes(cpdag, lambda x: x+1)
            
        if method == "hill":
            data_pd = pd.DataFrame(data, columns=[str(i+1) for i in range(data.shape[1])])
            cpdag_model=HillClimbSearch(data_pd)
            dag_pgmpy = cpdag_model.estimate(BicScore(data_pd), show_progress=False)
            dag_pgmpy=nx.relabel_nodes(dag_pgmpy, lambda x:int(x))
            dag = nx.DiGraph()
            dag.add_nodes_from([i+1 for i in range(data.shape[1])])
            dag.add_edges_from(dag_pgmpy.edges)
            cpdag = dag_to_cpdag(dag)
        
        return cpdag

    def all_mec_dags(self, data, method=None):
        if method is None:
            method = self.learn_algs[0]
        return cpdag_to_dags(self.learn_obs(data, method))
        
    

class CStree(object):
    """ Class for CStree model
    
    For a CStree, if the node has a "context" key then the frozenset inside it
    represents the context that determines the stage for that node

    The value_dict should have the keys representing variables from [1,...,p] 
    and the outcomes should be ordered [0,...,d_k] for each variable k

    """
    
    def __init__(self, value_dict):
        """ Initialize the CSTree experiment
        
        Args:
            value_dict (dict): Dictionary containing variables as keys and outcomes for
                               variables as values. Variables must start from 1 and outcomes
                               must be a list, with outcomes starting from 0

        """
        #self.data              = data
        #self.n, self.p         = self.data.shape
        self.p                 = len(value_dict.keys())
        self.value_dict        = value_dict
        self.contingency_table = None 

    def get_contingency_table(self, data):
        """ Function to get contingency table 

        Used as a means to cache the table if already generated
        TODO Think of whether new dataset actually changes table

        Args:
            data (np.array): Dataset
            

        """
        n, p = data.shape
        # Compute contingency table if it doesnt exist
        if self.contingency_table is None:
            sizes = list(self.value_dict.values())
            u_shape = tuple(len(sizes[i]) for i in range(p))
            self.contingency_table = np.zeros(u_shape)
            for i in range(n):
                sample = tuple(data[i,:])
                self.contingency_table[sample] +=1

        return self.contingency_table

    def likelihood(self, x, order, tree, data):
        """ Compute likelihood of the sample x given the staging of the tree and
            parameters estimated from the data under current staging

        Args:
            x (np.array): sample to get the likelihood of
            order (list): ordering of the variables in the tree
            tree (nx.DiGraph): Tree to get the staging from
            data (np.array): Dataset to generate the contingency table

        """
        _, p = data.shape

        # Make ordering start from 0
        order_py = [o-1 for o in order]

        u = self.get_contingency_table(data)

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
        for level in range(1,self.p):
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
            x_CjUk = [x[i] for i in range(self.p) if i+1 in CjUk]
            x_Cj   = [x[i] for i in range(self.p) if i+1 in Cj]

            numerator   = u_x_C(x_CjUk, CjUk)
            denominator = u_x_C(x_Cj,Cj)

            prob = prob*(numerator/denominator)
        return prob
            
            
    def cstree_to_stages(self, tree):
        """ Get the non-singleton stages from a learnt CStree

        Useful for model comparison and generating the CSI relations entailed by a CStree.

        Args:
            tree (nx.DiGraph): CStree to get stages from

        """
        # TODO Cache
        stages_per_level = {}
        for level in range(1, self.p):
            current_level_nodes = [n for n in tree.nodes
                               if nx.shortest_path_length(tree, "Root", n) == level]

            # Here, the context is a frozen set which allows stages_per_level
            # to be a set of sets, -1 is a dummy value
            stages_per_level[level] = {tree.nodes[n].get("context", -1)
                                       for n in current_level_nodes}.difference({-1})
            #print("cstreetostages",len(stages_per_level[level]))

        return stages_per_level

    def stages_to_csirels(self, stages, order):
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
        for level in range(1, self.p+1):
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


    def bic(self, tree, order, data):
        """ Compute the BIC of a CStree with respect to some data

        Args:
            tree (nx.DiGraph): CStree
            order (list): Ordering of the variables for the tree above
            data (np.array): Dataset

        Note: The variable stages are a dictionary with keys being colors and values being the nodes belonging to that stage. The variable color_scheme is a dictionary with key value pairs node and color, where the color is white if the node belongs to a singleton stage.
        """
        n = data.shape[1]

        # 1. Compute likelihood
        log_mle = sum(list(map(lambda i:
                               np.log(self.likelihood(
                                   data[i,:], order, tree, data)), range(n))))

        # Since we might use another dataset for the same object,
        # and we cached the contingency table mainly to save
        # time on computing the log_mle, we make it None again
        self.contingency_table = None 
        # 2. Get the free parameters

        # Dictionary where key is the level and the value is the contexts of
        # stages in that level
        stages_per_level = self.cstree_to_stages(tree)

        # TODO Check use of order below
        free_params = sum([len(stages_per_level[i])*(len(self.value_dict[order[i-1]])-1)
                           for i in range(1,self.p)])

        return log_mle-0.5*free_params*np.log(n)


    def dag_model(self, dag, order):
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
            vals = self.value_dict[var]

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
                stage_contexts = generate_vals(pars, self.value_dict)

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



    
    def random_cstree(self, order, ps, Ms):
        """ Generate a random CStree with variables and outcomes defined by self.value_dict
        
        The CStree is generated by choosing 2 random nodes Ms[l-1] times, and merging them
        to the same stage with probability ps[l-1] for level l

        Args:
            order (list): Order of the random CStree being generated
            ps (list): List containing the probabilities of merging 2 randomly selected nodes per level
            Ms (list): List containing number of times to select 2 random nodes per level

        """
        # TODO Look into moving contextn1,contextn2 inside the if merge block
        assert len(order)==len(list(self.value_dict.keys()))
        dag = generate_dag(len(order), 1)
        dag = nx.relabel_nodes(dag, lambda i: order[i-1])
        tree = self.dag_model(dag, order)

        for level in range(1, self.p):
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
                    existing_contexts = self.nonsingleton_stages(tree, new_nodes)

                    if existing_contexts!=set():
                        # If such contexts exist, the common context is the total intersection
                        common_subcontext = common_subcontext.intersection(*list(existing_contexts))

                        new_nodes = [n for n in current_level_nodes
                                    if common_subcontext.issubset(set(n))]

                    for node in new_nodes:
                        tree.nodes[node]["context"]=frozenset(common_subcontext)

        # Generate distribution with separate function
        tree_distr = self.tree_distribution(tree, order)
        return tree, tree_distr

    def tree_distribution(self, tree, order):
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
        cardinalities = [len(self.value_dict[i]) for i in order]
        tree_distr = DiscreteFactor(["X"+str(i) for i in order],
                                cardinalities,
                                np.ones(np.prod(cardinalities))/np.prod(cardinalities))

        # Each stage encodes a distribution for the variable that comes
        # after the variable that represents the node of that stage
        distrs = {}

        first_var_outcomes  = len(self.value_dict[order[0]])
        
        first_probabilities = np.random.dirichlet(
            [10*first_var_outcomes if np.random.rand()<0.5
             else 0.5*first_var_outcomes for _ in range(first_var_outcomes)])

        # Gathering all probabilities to see if they sum to 1
        prs = []
        
        # For each outcome from the tree
        # TODO This could be done by appending nodes onto
        # previous path instead of generating paths this way
        for leaf in leaves:
            path = nx.shortest_path(tree, "Root", leaf)
            for val in self.value_dict[order[-1]]:
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
                    outcomes = len(self.value_dict[next_var])
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
        print("All probabilities sum to ",sum(prs))
        return tree_distr
        
        
    def random_hsbm(self, k, order, ps):
        """ Construct a random Hypothesis-Specific Bayesian Multinet (HSBM).

        This is done by generating a DAG model for each of the outcomes of the 
        first variable, which is k

        Args:
            k (int): Number of outcomes for first variable in the HSBM
            order (list): Ordering of the variables 
            ps (list): Probabilities of including an edge for each of the k DAGs

        """
        
        # DAG CStree
        # Change outcomes for first variable to accomodate for the k outcomes
        self.value_dict[order[0]]=[i for i in range(k)]

        # Empty graph for CStree
        #tree = nx.DiGraph()
        tree = self.dag_model(generate_dag(self.p, 1), order)

        # Add initial edges from Root to first k outcomes
        # tree.add_edges_from([("Root",((order[0], i),)) for i in range(k)])
        
        ss=0
        # For k^th outcome in first variable, create a CStree for a random
        # DAG and connect the root of it to leaf corresponding to k^th outcome
        for i in range(k):
            dag = generate_dag(self.p-1, ps[i])
            dag = nx.relabel_nodes(dag, lambda i: order[1:][i-1])
            assert order[1:] in list(nx.all_topological_sorts(dag))
            dag_cstree = self.dag_model(dag, order[1:])

            for node in dag_cstree:
                if node == "Root":
                    continue
                actual_node = ((order[0], i),)+node
                # Since singleton contexts are simply not stored
                # we need to fix the first outcome context to
                # non-singleton stages
                if dag_cstree.nodes[node].get("context", None) is not None:
                    tree.nodes[actual_node]["context"] = frozenset( ((order[0], i),)).union(dag_cstree.nodes[node]["context"])

            ss+=self.num_stages(dag_cstree)


        tree_distr = self.tree_distribution(tree, order)
        return tree, tree_distr

    
    def nonsingleton_stages(self, tree, nodes):
        """ Get the contexts of non-singleton stages in tree from a given list of node

        """
        existing_context_nodes = list(filter
                                      (lambda node: True if tree.nodes[node].get("context",None) is not None else False, nodes))
        existing_contexts = set(tree.nodes[node]["context"] for node in existing_context_nodes)
        return existing_contexts
    

    def nodes_with_context(self, context, nodes):
        """ Get the nodes from a list of nodes which contain a specified context

        """
        return list(filter(lambda node: True if set(context).issubset(set(node)) else False, nodes))

    def generate_contexts(self, vars, size):
        """ Generate all possible contexts from a given list of variables with a given size
        

        """
        var_combos = list(combinations(vars, size))

        contexts = []
        for var_combo in var_combos:
            temp = generate_vals(list(var_combo), self.value_dict)
            for context in temp:
                contexts.append(context)
        return contexts

    
    def predict(self, tree, order, sample, i, data):
        """ Find P(Xi|X_[p]\ {i})

        By Bayes theorem, this is the probability of observing the sample 
        i.e. P(X1=x1,...,Xp=xp) divided by the probability of the marginal i.e.
        P(X1=x1, ..., Xi-1=xi-1, Xi+1=xi+1,...,Xp=xp) is the same as
        sum_j P(X1=x1, ..., Xi-1=xi-1,Xi=j Xi+1=xi+1,...,Xp=xp) where xk are from sample

        """
        numerator    = self.likelihood(sample, order, tree, data)
        order_of_var = order.index(i)
        samples      = [np.array(list(sample[:order_of_var])+[val]+list(sample[order_of_var+1:]))
                        for val in self.value_dict[i]]
        # sum_j P(X1=x1,...,Xi-1=xi-1,Xi+1=xi+1,...,Xn|Xi=j)P(Xi=j)
        denominator  = sum([self.likelihood(s, order, tree, data) for s in samples])
        return numerator/denominator


    def node_based_test(self, data, context_n1, context_n2, var, order, csi_test, oracle):
        """ Wrapper for whether to merge 2 nodes or not 

        Here we test samples based on contexts fixed by the nodes

        TODO More information on this

        """
        if oracle:
            distribution_copy = oracle.copy()
            common_subcontext = set(context_n1).intersection(set(context_n2))
            vars_to_marginalize    = order[order.index(var)+1:]
            vars_to_marginalize    = ["X"+str(var)
                                      for var in vars_to_marginalize]
            context_to_marginalize = [("X"+str(var),val)
                                      for (var,val) in common_subcontext]
            distribution_copy.marginalize(vars_to_marginalize)
            distribution_copy.reduce(context_to_marginalize)

            rank =  np.linalg.matrix_rank(distribution_copy.values.reshape(len(self.value_dict[var]),-1))

            if rank == 1:
                merge=True
            else:
                merge=False

        # If using samples and not oracle
        else:
            data_n1 = data_to_contexts(data, list(context_n1), var)
            data_n2 = data_to_contexts(data, list(context_n2), var)

            merge = csi_test(data_n1, data_n2, self.value_dict[var])
        return merge

    def context_based_test(self, data, context, var, order, oracle, B, level):
        """ Wrapper for whether to decide if var is indepenent of the variables
        preceding it except for those in the context, when conditioned on the context

        TODO More details on this

        """
        if oracle:
            distribution_copy = oracle.copy()
            vars_to_marginalize    = order[order.index(var)+1:]
            vars_to_marginalize    = ["X"+str(var)
                                      for var in vars_to_marginalize]
            context_to_marginalize = [("X"+str(var),val)
                                      for (var,val) in context]
            distribution_copy.marginalize(vars_to_marginalize)
            distribution_copy.reduce(context_to_marginalize)

            rank =  np.linalg.matrix_rank(distribution_copy.values.reshape(len(self.value_dict[var]),-1))

            if rank == 1:
                merge=True
            else:
                merge=False
        else:
            context_vars = vars_of_context(context)
            # For each context, run the test to see if we can merge them
            data_to_context = data_to_contexts(data, context)
            B_prime = [var for var in B if var not in context_vars]
            temp_cond_set = set()
            times_independent = []
            for b in B_prime:
                print("testing", level, context, var-1, b-1, temp_cond_set)
                p = ci_test_bin(data_to_context, var-1, b-1, temp_cond_set)
                if p<0.05:
                    # Not independent then move on to next context
                    continue
                else:
                    print(temp_cond_set)
                    times_independent.append(True)
                    temp_cond_set = temp_cond_set.union({b-1})
            if len(times_independent)==len(B):
                merge=True
            else:
                merge=False
        return merge

    
    def learn_obs_mcknown(self, data, minimal_contexts, orders):
        minimal_context_dags = []
        m = len(minimal_contexts)
        for mc in minimal_contexts:
            if mc == [()]:
                mc = ()
            ci_rels = []
            C       = vars_of_context(mc)
            vars    = [i+1 for i in range(self.p) if i+1 not in C] #[p]\C
            print(C,vars)
            dag     = nx.complete_graph(vars)

            # Loop below is MaxDegree in overleaf

            for j in range(len(vars)+1):
                for k,l in dag.edges:
                    neighbors = list(set(list(dag.neighbors(k))).difference({k,l}))
                    if len(neighbors)>=j:
                        subsets = [set(i) for i in combinations(neighbors, j)]
                        for subset in subsets:
                            z_subset = set(i-1 for i in subset)
                        #zero_indexed_subsets = [set([i-1 for i in s]) for s in subsets]
                        #for z_subset in zero_indexed_subsets:

                            # For the CI testing, we need 0 indexing
                            # TODO Generalize this to use the csi_test argument in this method
                            #print(data_to_contexts(data, mc))
                            p_val = ci_test_bin(data_to_contexts(data, mc), k-1, l-1, z_subset)

                            #print(mc, "Tested ",k,l,subset, p_val)
                            # TODO put alpha as part of function

                            if p_val > 0.05:

                                #print("Removng edge",k,l,p_val)
                                dag.remove_edges_from([(k,l)])
                                if ({k},{l}, {i+1 for i in z_subset}) not in ci_rels:
                                    # TODO Maybe try frozen sets here
                                    ci_rels.append(({k},{l}, {i for i in z_subset}))
                                if ({l},{k}, {i+1 for i in z_subset}) not in ci_rels:
                                    ci_rels.append(({l},{k}, {i for i in z_subset}))
                            #print(ci_rels)
            minimal_context_dags.append((mc, nx.DiGraph(dag), ci_rels))


        P_edges = set()
        # Orienting edges and getting a graph P with directed edges
        for i in range(m):
            mc, dag_u, ci_rels = minimal_context_dags[i]
            dag = nx.DiGraph(dag_u)
            for k,s,l in combinations(list(dag.nodes),3):
                # Since the DAG is undirected at the moment,
                # edge direction does not matter
                if (k,s) in dag_u.edges and (s,l) in dag_u.edges and (l,k) not in dag_u.edges:
                    for K,L,S in ci_rels:
                        if {k}==K and {l}==L and s not in S:
                            if not ((s,k) in P_edges or (s,l) in P_edges):

                                dag.remove_edges_from([(s,k),(s,l)])
                                P_edges = P_edges.union({(k,s),(l,s)})
            minimal_context_dags[i] = (mc, dag, ci_rels)





        dag_P = nx.DiGraph()
        dag_P.add_nodes_from([i+1 for i in range(self.p)])
        dag_P.add_edges_from(list(P_edges))

        if orders is None:
            # Maybe call variable below possible orders
            orders = nx.all_topological_sorts(dag_P)

        # Flag for debugging
        order_found=False
        for order in orders:
            if not order_found:
                minimal_context_dags_ordered = []
                new_v_struct = False
                for i in range(m):
                    mc, dag, ci_rels = minimal_context_dags[i]

                    # edges in the fully connected DAG with this order
                    # Below line relies on how "combinations" works, namely,
                    # e.g. given [a,b,c,d] it returns [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]
                    es = [(i,j) for i,j in combinations(order, 2) if i<j]

                    for (u,v) in es:
                        if (u,v) in dag.edges and (v,u) in dag.edges:
                            # direct the edge u<->v as u->v by deleting u<-v
                            dag.remove_edge(v,u)

                            # check if this introduces a v-structure
                            edges_in = coming_in((u,v), dag)

                            # even if we get 1 new v-structure we move on to the next ordering
                            new_v_struct = any(list(map(lambda e: v_structure((u,v),e,dag), edges_in)))
                            if new_v_struct:
                                break

                if new_v_struct:
                    break
                # if no new vstruct, we choose this ordering
                # minimal_context_dags_ordered.append((mc, dag))
                order_found=True

        if order_found:
            # Since the above code breaks after finding an ordering that
            # is consistent, it is saved in the order variable which we use below
            print("Chosen order ", order)
        else:
            print("Order not found")

        for mc, dag, _ in minimal_context_dags:
            dag_edges = list(dag.edges)
            removed=[] # to debug
            for (u,v) in dag_edges:

                if (v,u) in dag_edges and (u,v) not in removed and (v,u) not in removed:
                    order_v, order_u = order.index(v), order.index(u)
                    if order_v<order_u:
                        dag.remove_edge(u,v)
                        removed.append((u,v))
                    if order_u<order_v:
                        dag.remove_edge(v,u)
                        removed.append((v,u))

        return minimal_context_dags

    def generate_dag_order_pairs(self, data, use_dag, dag_method, orders):
            
        dag_order_pairs = []

        if orders:
            if use_dag:
                # Order known AND using DAG CI relations
                possible_orders = []
                dags = DAG().all_mec_dags(data, dag_method)
                for order in orders:
                    found_dag = False
                    for dag in dags:
                        if order in list(nx.all_topological_sorts(dag)):
                            dag_order_pairs.append((dag,order))
                            found_dag=True
                            break
                    if not found_dag:
                        dag = generate_dag(self.p, 1)
                        dag = nx.relabel_nodes(dag, lambda i: order[i-1])
                        dag_order_pairs.append((dag, order))

                assert len(dag_order_pairs) == len(orders)
            else:
                # Order known AND not using DAG CI relations
                # Using full DAG which encodes no CI relations
                for order in orders:
                    dag = generate_dag(self.p, 1)
                    dag = nx.relabel_nodes(dag, lambda i: order[i-1])
                    dag_order_pairs.append((dag, order))
        else:
            dags = DAG().all_mec_dags(data, dag_method)
            for dag in dags:
                orders = nx.all_topological_sorts(dag)
                for order in orders:
                    if use_dag:
                        # Order unknown AND using DAG CI relations
                        dag_order_pairs.append((dag, order))
                    else:
                        # Order unknown AND not using CI relations
                        dag = generate_dag(self.p, 1)
                        dag = nx.relabel_nodes(dag, lambda i: order[i-1])
                        dag_order_pairs.append((dag, order))
        return dag_order_pairs

    def learn_obs(self,
                  data,
                  csi_test,
                  criteria,
                  use_nodes=True,
                  node_ratios=None,
                  context_limit=None,
                  dag_method=None,
                  orders=None,
                  oracle=False,
                  minimal_contexts=None,
                  use_dag=False):

        # If we are given the minimal contexts, we learn the minimal context
        # DAGs directly
        if minimal_contexts:
            minimal_context_dags =  self.learn_obs_mcknown(data, minimal_contexts, orders)
            return minimal_context_dags
            
        else:
        # If we are not given minimal contexts
        # TODO Put an else here to match the if minimal_contexts:
            # Generate pairs of DAGs and orders to start CStree from
            # If we do not use DAG CI relations, a fully connected DAG
            # is sent.
            dag_order_pairs = self.generate_dag_order_pairs(data, use_dag, dag_method, orders)

            best_score = -1000000000
            best_cstrees = []

            for dag, order in dag_order_pairs:
                cstree = self.dag_model(dag, order)
                skipped=0

                # This loop can be parallelized, since
                # the independence tests do not depend
                # on the level
                for level in range(1, self.p):
                    # The variable for the next level
                    var = order[level]
                    # TODO name this current_level_nodes
                    nodes_l = [n for n in cstree.nodes
                            if nx.shortest_path_length(cstree, "Root", n) == level]

                    # We test using 2 ways
                    # 1) Pairwise node testing
                    # 1a) All combinations of nodes
                    if use_nodes:
                        if node_ratios:
                            # Select a random subset of nodes to run tests on
                            nodes_l = random.sample(nodes_l, int(node_ratios[level-1]*len(nodes_l))) # 
                        #else:
                            # Select all nodes to run tests on
                        nodes_to_compare = combinations(nodes_l, 2)
                            
                        for n1,n2 in nodes_to_compare:
                            context_n1 = cstree.nodes[n1].get("context", n1)
                            context_n2 = cstree.nodes[n2].get("context", n2)

                            if context_n1 == context_n2:
                                # If the nodes have the same stage we skip testing them
                                skipped+=1
                                continue
                            else:
                                # If the nodes have different stages we attempt to merge them
                                merge = self.node_based_test(data,
                                                             context_n1, context_n2,
                                                             var, order, csi_test, oracle)
                                if merge:
                                        # Contexts are tuples if they are singleton
                                    common_subcontext = set(context_n1).intersection(set(context_n2))
                                    new_nodes = [n for n in nodes_l
                                                if common_subcontext.issubset(set(n))]

                                    # If there are non-singleton nodes trapped in between
                                    # we need to get the common context of them and the
                                    # new common context we just learnt about
                                    existing_contexts = self.nonsingleton_stages(cstree, new_nodes)

                                    # If we do have such contexts, we get the new common context
                                    if existing_contexts!=set():
                                        common_subcontext = common_subcontext.intersection(*list(existing_contexts))
                                        new_nodes = [n for n in nodes_l
                                                    if common_subcontext.issubset(set(n))]
                                    for node in new_nodes:
                                        cstree.nodes[node]["context"]=frozenset(common_subcontext)
                    else:
                        # If we use contexts instead of nodes
                        if context_limit is None:
                            context_size_current_level = level
                        else:
                            context_size_current_level = context_limit[level]+1
                            
                        for context_size in range(0, context_size_current_level):
                        # For each context var size from 0 inclusive
        
                            # TODO Check if level is good to use or level+1 in line below
                            B = order[:level] # B in X_A _||_ X_B | X_C=x_c, note that A={var}
                            contexts = self.generate_contexts(B, context_size)
                            
                            for context in contexts:
                                merge = self.context_based_test(data, context, var, order, oracle, B, level)

                                if merge:                                    
                                    new_nodes = [n for n in nodes_l
                                                if set(context).issubset(set(n))]

                                    # If there are non-singleton nodes trapped in between
                                    # we need to get the common context of them and the
                                    # new common context we just learnt about
                                    existing_contexts = self.nonsingleton_stages(cstree, new_nodes)

                                    # If we do have such contexts, we get the new common context
                                    if existing_contexts!=set():
                                        context = set(context).intersection(*list(existing_contexts))
                                        new_nodes = [n for n in nodes_l
                                                    if context.issubset(set(n))]
                                    for node in new_nodes:
                                        cstree.nodes[node]["context"]=frozenset(context)
                                    
                                        
                # select best trees and order based on criteria
                assert criteria in ["bic", "minstages"]
                if criteria=="bic":
                    criteria_func = lambda tree, order: self.bic(tree, order, data)
                elif criteria=="minstages":
                    criteria_func = lambda tree, order: -1*(self.num_stages(tree))
                else:
                    raise ValueError("Criteria undefined")

                criteria_score = criteria_func(cstree, order)
                if criteria_score > best_score:
                    best_score = criteria_score
                    best_cstrees = [(cstree, order)]
                elif criteria_score == best_score:
                    best_cstrees.append((cstree, order))

            return best_cstrees


    def num_stages(self, tree):
        """ Get the number of non-singleton stages in a given CStree
        
        """
        stages = self.cstree_to_stages(tree)
        return sum([len(stages[i]) for i in range(1,self.p)])

    
    def visualize(self, tree, order, height_limit=10, colors=None, plot_mcdags=False, save_dir=None):
        """ 

        """

        levels = len(order)
        if levels>height_limit:
            raise ValueError("Tree has too many nodes!")
        
        stages = self.cstree_to_stages(tree)
        stage_count = sum([len(stages[i]) for i in range(1,self.p)])

        if colors:
            assert len(colors) == stage_count
        else:
            colors = ["#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
                        for _ in range(stage_count)]
        print("Color count is ", len(colors))
        # Make into generator for easy access using "next" keyword
        colors = chain(colors)

        # Keys are (level, context), value is the non-white color for that stage
        stage_color_dict = {}

        color_scheme = []
        # TODO See why creating list with #FFFFFF gives syntax? error
 

        # Get the color scheme
        for node in list(tree.nodes):
            # Assign the color white first
            color_scheme.append("#FFFFFF")
            if node == "Root":
                continue
            level = len(node)
            # For all contexts determining the stages in level of current node
            for context in stages[level]:
                # If node belongs to a stage, assign color of stage
                if context.issubset(set(node)):
                    if stage_color_dict.get((level, context), None) is None:
                        stage_color_dict[(level, context)]=next(colors)
                    color_scheme[-1] = stage_color_dict[(level, context)]
                    break

        # TODO Make this nicer
        # Circular vs Tree plot for CStree
        if levels<7:
            # TODO Rotate this 90deg counter clockwise, multiple values
            # for each key, keys are nodes values are x.y locations
            tree_pos = graphviz_layout(tree, prog="dot", args="")
        else:
            tree_pos = graphviz_layout(tree, prog="twopi", args="")

        # Plot the CStree
        if not plot_mcdags:
            fig = plt.figure(figsize=(12, 12))
            tree_ax = fig.add_subplot(111)
            nx.draw_networkx(tree, node_color=color_scheme,
                             ax=tree_ax, pos=tree_pos, with_labels=False,
                             font_color="white", linewidths=1)
            tree_ax.collections[0].set_edgecolor("#000000")
            if save_dir:
                plt.savefig("figs/"+save_dir+"_cstree.pdf")
            else:
                plt.show()

        # Plot the minimal context DAGs
        else:
            csi_rels = self.stages_to_csirels(stages, order)

            csi_rels    = graphoid_axioms(csi_rels.copy(), self.value_dict)
            all_mc_dags = minimal_context_dags(order, csi_rels.copy(),
                                               self.value_dict, closure=csi_rels.copy())
            num_mc_dags = len(all_mc_dags)

            fig = plt.figure(figsize=(14,12))
            main_ax = fig.add_subplot(111)
            tree_ax = plt.subplot(2,1,2)
            dag_ax  =  [plt.subplot(2, num_mc_dags, i+1, aspect='equal')
                        for i in range(num_mc_dags)]

            nx.draw_networkx(tree, node_color=color_scheme, ax=tree_ax, pos=tree_pos,
                        with_labels=False, font_color="white", linewidths=1)

            tree_ax.collections[0].set_edgecolor("#000000")

            for i, (mc, dag) in enumerate(all_mc_dags):
                options = {"node_color":"white", "node_size":1000}
                dag = nx.relabel_nodes(dag, lambda x: x+1 if x>=4 else x)
                if mc!=():
                    mcdag_title = "".join(["$X_{}={}$  ".format(mc[i][0],mc[i][1]) for i in range(len(mc))])
                else:
                    mcdag_title = "Empty context"
                
                dag_ax[i].set_title(mcdag_title)
                if list(dag.edges)!=[]:
                    dag_pos = nx.drawing.layout.shell_layout(dag)
                    nx.draw_networkx(dag, pos=dag_pos, ax=dag_ax[i], **options)
                else:
                    # Darn equal plot size doesnt work with shell layout
                    nx.draw_networkx(dag, ax=dag_ax[i], **options)

                dag_ax[i].collections[0].set_edgecolor("#000000")
                if save_dir:
                    plt.savefig("figs/"+save_dir+"_cstree_and_mcdags.pdf")
                else:
                    plt.show()
