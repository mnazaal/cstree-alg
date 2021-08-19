import networkx as nx
import numpy as np
import pandas as pd
import random
from itertools import chain, combinations
from pgmpy.estimators import PC, HillClimbSearch, BicScore, K2Score
from pgmpy.factors.discrete import DiscreteFactor
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

from .utils.tools import generate_vals, parents, data_to_contexts, remove_cycles, cpdag_to_dags, dag_to_cpdag, generate_dag
from .utils.pc import estimate_cpdag, estimate_skeleton
from .mincontexts import minimal_context_dags
from .graphoid import graphoid_axioms
#from gsq import ci_test_bin, ci_test_dis


class DAG(object):
    """ Class acting as a wrapper for DAG model

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
            binary_data = True if all([True if len(np.unique(data[:,i])) else False for
                                     i in range(p)]) else False

        # Set the test to get CPDAG
            if binary_data:
                #pc_test = ci_test_bin
                pc_test=None
            else:
                #pc_test = ci_test_dis
                pc_test=None
            
            
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
        """

        """
        #self.data              = data
        #self.n, self.p         = self.data.shape
        self.p                 = len(value_dict.keys())
        self.value_dict        = value_dict
        self.contingency_table = None 

    def get_contingency_table(self, data):
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
        """ Compute likelihood of the data given staging

        

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
        # CSI rels are of the form (X_k: set, X_{k-1}\C: set, set(), X_C: set)
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
        """
        Note: The variable stages are a dictionary with keys being colors and values being the nodes belonging to that stage. The variable color_scheme is a dictionary with key value pairs node and color, where the color is white if the node belongs to a singleton stage.
        """
        n = data.shape[1]

        # 1. Compute likelihood
        log_mle = sum(list(map(lambda i:
                               np.log(self.likelihood(
                                   data[i,:], order, tree, data)), range(n))))

        # 2. Get the free parameters

        # Dictionary where key is the level and the value is the contexts of
        # stages in that level
        stages_per_level = self.cstree_to_stages(tree)

        # TODO Check use of order below
        free_params = sum([len(stages_per_level[i])*(len(self.value_dict[order[i-1]])-1)
                           for i in range(1,self.p)])

        return log_mle-0.5*free_params*np.log(n)


    def dag_model(self, dag, order):
        """ Get the DAG model as a CStree with given order

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
        assert len(order)==len(list(self.value_dict.keys()))
        dag = generate_dag(len(order), 1)
        dag = nx.relabel_nodes(dag, lambda i: order[i-1])
        tree = self.dag_model(dag, order)

        for level in range(1, self.p):
            # Generate a context randomly
            current_level_nodes = [n for n in tree.nodes
                               if nx.shortest_path_length(tree, "Root", n) == level]
            for _ in range(Ms[level-1]):
                # Choose 2 random nodes
                random_node_pair = random.sample(current_level_nodes, 2)
                # Merge their stages with probability ps[level-1]
                merge = True if np.random.uniform() < ps[level-1] else False
                if merge:
                    # r for random
                    r_node1, r_node2 = random_node_pair[0], random_node_pair[1]

                    # Get contexts
                    context_n1 = tree.nodes[r_node1].get("context", r_node1)
                    context_n2 = tree.nodes[r_node2].get("context", r_node2)
                    
                    # If different, get common subcontext, assign
                    # stage to all nodes in current level with that subcontext
                    if set(context_n1) != set(context_n2):
                        common_subcontext = set(context_n1).intersection(set(context_n2))
                        new_nodes = [n for n in current_level_nodes
                                     if common_subcontext.issubset(set(n))]
                        for node in new_nodes:
                            tree.nodes[node]["context"]=frozenset(common_subcontext)

        # Generate distribution with separate function
        tree_distr = self.tree_distribution(tree, order)
        return tree, tree_distr

    def tree_distribution(self, tree, order):
        # Takes a CStree, its order, then gives a DiscreteFactor object
        # which can be used to generate samples

        # Note list below excludes values from last variable
        leaves = [n for n in tree.nodes
                  if nx.shortest_path_length(tree, "Root", n) == len(order)-1]
        # All outcomes include the possibilies of the last variable
        # which are excluded in our CStree graph
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
        for leaf in leaves:
            path = nx.shortest_path(tree, "Root", leaf)
            for val in self.value_dict[order[-1]]:
                # Adding last value to the path
                actual_path = path+[path[-1]+((order[-1],val),)]
                # Probability of first outcome
                pr = first_probabilities[path[1][0][-1]]
                # For each next node, get probabilities according to staging
                # or create them if encountering the staging for first time
                # A stage of a node is determined uniquely by the level of the
                # node in the tree and the context which fixes the stage

                # Skipping over first node which is root
                # since that value is taken into pr, skipping last
                # since we need nodes to get stages, from which we get
                # the probability values for the level afterwards
                for node in actual_path[1:-1]:
                    # Next variable and its outcomes
                    var = node[-1][0]
                    outcomes = len(self.value_dict[var])
                    
                    
                    level = len(node)
                    context = frozenset(tree.nodes[node].get("context",node))
                    if distrs.get((level, context), None) is None:
                        alpha = [10*outcomes if np.random.rand()<0.5
                                 else 0.5*outcomes for _ in range(outcomes)]
                        distrs[(level, context)]=np.random.dirichlet(alpha)

                    # We need the next outcome value of the path
                    next_outcome = actual_path[level+1][-1][-1]
                    #print(distrs[(level, context)])
                    pr = pr*distrs[(level, context)][next_outcome]

                # Adding probabilities at this level otherwise you miss the last outcome
                actual_leaf = actual_path[-1]
                kwargs = {"X"+str(var):val for (var,val) in actual_leaf}
                tree_distr.set_value(pr, **kwargs)
                prs.append(pr)
                
        #print("All probabilities sum to ",sum(prs))
        return tree_distr
        
        
        
    def random_hsbm(self, k, order, ps):
        # Hypothesis-Specific Bayesian Multinet
        # DAG CStree
        # Change outcomes for first variable
        self.value_dict[order[0]]=[i for i in range(k)]
        tree = nx.DiGraph()
        tree.add_edges_from([("Root",((order[0], i),)) for i in range(k)])
        ss=0
        for i in range(k):
            dag = generate_dag(self.p-1, ps[i])
            dag = nx.relabel_nodes(dag, lambda i: order[1:][i-1])
            assert order[1:] in list(nx.all_topological_sorts(dag))
            dag_cstree = self.dag_model(dag, order[1:])

            for node in dag_cstree.nodes:
                if node=="Root":
                    continue
                # Contexts must be in order for colors to match which is why
                # we do (first context) U (dag context) instead of
                # (dag context) U (first context) below
                dag_cstree.nodes[node]["context"]=frozenset( ((order[0], k-i-1),)).union(dag_cstree.nodes[node].get("context",frozenset()))
            ss+=self.num_stages(dag_cstree)
            #self.visualize(dag_cstree, order[1:])
            #self.tree_distribution(dag_cstree, order[1:])

            #k-i-1 instead of i to maintain left-to-right increasing order
            dag_cstree = nx.relabel_nodes(dag_cstree, lambda x: ((order[0], k-i-1),)
                                          if x=="Root" else ((order[0], k-i-1),)+x)
            #tree.add_nodes_from(dag_cstree.nodes(data=True))
            #tree.add_edges_from(dag_cstree.edges(data=True))
            tree = nx.compose(dag_cstree, tree)
            for n in tree.nodes:
                pass
                #ttt = tree.nodes[n].get("context","reeee")
                #print(n,ttt)
            
            #tree.add_edges_from(list(dag_cstree.edges))

            # We need to assign the contexts separatel apparently
            # Maybe look into nx.compose or nx.union to prevent this
            """
            for node in dag_cstree.edges:
                if node == "Root":
                    continue
                #tree.add_node( ((order[0], i),)+node )
                try:
                    tree.nodes[node]["context"]=dag_cstree.nodes[node]["context"].union( ((order[0], k-i-1)))
                    print(tree.nodes[node]["context"])
                except:
                    pass"""
            #dag_cstree.remove_node("Root")
            #tree.add_edges_from(list(dag_cstree.edges))
            #tree = nx.compose(dag_cstree, tree)
            
            
        self.visualize(tree, order)
        for node in tree.nodes:
            pass
            #ttt = tree.nodes[node].get("context","reeee")
            #print(node, ttt)
            #print(node, tree.nodes[node].get("context", "nocontext"))
        #print(ss,self.num_stages(tree))
        tree_distr = self.tree_distribution(tree, order)
        return tree, tree_distr
        
            # Stick this tree onto new cstree
            

        

        # DAG CStree for variables [pi_2,...,pi_p]
        
        
        
    

    def learn_obs(self,
                  data,
                  csi_test,
                  criteria,
                  dag_method=None,
                  orders=None,
                  minimal_contexts=None,
                  use_dag=False):
        
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


        best_score = -1000000000
        best_cstrees = []
        
        for dag, order in dag_order_pairs:
            if minimal_contexts:
                # TODO
                cstree=None
                pass
            
            else:
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
                    for n1,n2 in combinations(nodes_l, 2):
                        context_n1 = cstree.nodes[n1].get("context", n1)
                        context_n2 = cstree.nodes[n2].get("context", n2)

                        if context_n1 == context_n2:
                            skipped+=1
                            continue
                        else:
                            data_n1 = data_to_contexts(data, list(context_n1), var)
                            data_n2 = data_to_contexts(data, list(context_n2), var)

                            same_distr = csi_test(data_n1, data_n2, self.value_dict[var])

                            if same_distr:
                                # Contexts are tuples if they are singleton
                                common_subcontext = set(context_n1).intersection(set(context_n2))
                                new_nodes = [n for n in nodes_l
                                             if common_subcontext.issubset(set(n))]
                                for node in new_nodes:
                                    cstree.nodes[node]["context"]=frozenset(common_subcontext)

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
        stages = self.cstree_to_stages(tree)
        return sum([len(stages[i]) for i in range(1,self.p)])

    
    def visualize(self, tree, order, height_limit=10, colors=None, plot_mcdags=False, save_dir=None):

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
