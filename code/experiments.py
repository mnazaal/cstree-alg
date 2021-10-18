import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import causaldag as cd

from .cstree import CStree
from code.utils.tools import test_anderson_k, test_epps, test_skl_divergence

significance_lvl = 0.01
ak = lambda samples1,samples2,o: test_anderson_k(samples1,samples2,o,significance_lvl)

ep = lambda samples1,samples2,o: test_epps(samples1,samples2,o,significance_lvl)

        # for SKL, the parameters are the threshold
kl_threshold = 1e-4
skl = lambda samples1,samples2,o: test_skl_divergence(samples1,samples2,o,kl_threshold)
# Sparsity and its effect from just samples
# Generate sparsiy levels
# For each sparsity level, generate m random models
# For each model, generate n samples
def hsbm_consistency(ks, m, n, vars, use_oracle, use_order, metric="shd"):
    # For each k i.e. hypotheses
    # For each sparsity level
    # Generate m models
    
    ss = [i/(vars-1) for i in range(vars)]

    true_order = [i+1 for i in range(vars)]
    value_dict = {i+1:[0,1] for i in range(vars)}

    if use_order:
        orders=[true_order]
    else:
        orders=None

    k_dict = {}
    for k in ks:
        mcs =  [((1,i),) for i in range(k)]

        s_dict = {}
        for s in ss:

            val_s = 0
            for _ in range(m):
                tree_object = CStree(value_dict)
                _, distr, true_dags = tree_object.random_hsbm(k,
                                                              true_order,
                                                              k*[s])
                samples = np.array(distr.sample(n))
                if use_oracle:
                    oracle=distr.copy()
                else:
                    oracle=None
                learnt_dags = tree_object.learn_observational(samples, None, "bic", minimal_contexts=mcs, orders=orders, oracle=oracle)

                if metric=="equal":
                    for i in range(k):
                        #print(set(true_dags[i].edges))
                        #print(set(learnt_dags[i][1].edges))
                        if set(true_dags[i].edges)==set(learnt_dags[i][1].edges):
                            val_s+=1/(k*m)
                elif metric=="shd":
                    for i in range(k):
                        true_dag_cd = cd.DAG(arcs={*list(true_dags[i].edges)})
                        learnt_dag_cd = cd.DAG(arcs={*list(learnt_dags[i][1].edges)})
                        # total SHM for this s,k
                        val_s += learnt_dag_cd.shd(true_dag_cd)/(k*m)
             
            s_dict[s]=val_s
        k_dict[k]=s_dict
        print(s_dict)
    np.save(metric+'hsbm_consistency.npy', k_dict)
 
        
def prediction(var, tree_object, learnt_tree, distribution, predsize, order, train_samples):
    # Use R code on same dataset
    test_samples = np.array(distribution.sample(predsize))
    correct = 0
    wrong_unique=[]
    for i in range(predsize):
        x      = test_samples[i,:]
        actual = x[order.index(var)]

        # this is x with position var switched
        # with all other outcomes it can take
        # cf for counterfactual
        xs_cf = []
        x_copy = x.copy()
        for val in tree_object.value_dict[var]:
            x_copy[order.index(var)] = val
            xs_cf.append(x_copy.copy())
        probs = [tree_object.predict(learnt_tree, order, x_cf, var, train_samples) for x_cf in xs_cf]
        predicted = probs.index(max(probs))
        
        #print(predicted, actual, x)
        if predicted==actual:
            correct+=1
        else:
            print(x,predicted,actual)
            if (list(x), predicted, probs) not in wrong_unique:
                wrong_unique.append((list(x), predicted, probs))
    return correct, wrong_unique
        


def hsbm_prediction(ks, m, train_n, test_n, vars, pred_var):
    # For each k i.e. hypotheses
    # For each sparsity level
    # Generate m models
    
    ss = [i/(vars-1) for i in range(vars)]

    true_order = [i+1 for i in range(vars)]
    value_dict = {i+1:[0,1] for i in range(vars)}

    k_dict = {}
    for k in ks:
        mcs =  [((1,i),) for i in range(k)]

        s_dict = {}
        for s in ss:

            val_s_cstree = 0
            val_s_set    = 0
            for _ in range(m):
                tree_object = CStree(value_dict)
                _, distr, _ = tree_object.random_hsbm(k,
                                                              true_order,
                                                              k*[s])
                train_samples = np.array(distr.sample(train_n))
 
                learnt_dags = tree_object.learn_observational(train_samples, None, "bic", minimal_contexts=mcs, orders=[true_order])
                learnt_cstree, _ = tree_object.generate_hsbm(true_order, k, [d[1] for d in learnt_dags])
                corrects,_ = prediction(pred_var, tree_object, learnt_cstree, distr, test_n, true_order, train_samples)
                val_s_cstree+=corrects/(test_n*m)
                
                
                # R code to get correct predictions from R
                #val_s_set+=
                
            s_dict[s]=(val_s_cstree, val_s_set)
            print(k,s, s_dict)
        k_dict[k]=s_dict
        print(k, k_dict)
    print(k_dict)
    np.save('hsbm_preds.npy', k_dict)


def cstree_consistency(m,n,vars,
                       use_oracle=False, use_order=False,
                       use_dag=True, **kwargs):
    # Notes TODO
    # For oracle version, the independence structure is the same
    # i.e. same nodes get coloured but the colouring is different
    # if the parameters are different but by the axioms the relations
    # together are the same
    # Generate CStree
    vd = {i+1:[0,1] for i in range(vars)}

    order = [i+1 for i in range(vars)]
    if use_order:
        orders=[order]
    else:
        orders=None

    ss = [i/(vars-1) for i in range(vars)]
    ss = [0.1 for i in range(vars)]

    Ms = [int(2*np.sqrt(i)) for i in range(vars)]

    for _ in range(m):
        rand_experiment = CStree(vd)
        rand_tree, rand_tree_distr = rand_experiment.random_cstree(order, ss, Ms)
        samples = np.array(rand_tree_distr.sample(n))
        if use_oracle:
            oracle=rand_tree_distr.copy()

        else:
            oracle=None

        # These functions should be passed after partially applying the parameters, 
        
        # for example for the Hypothesis tests, they are the significance value


        # The scoring method (bic/minstages) does not matter as they are used to choose
        # the best order, and here we give the ordering
        # rand_tree_distr
        best_cstree, best_order = rand_experiment.learn_observational(samples, ak, "minstages", orders=orders, use_dag=use_dag, oracle=oracle, **kwargs)[0]

        rand_experiment.visualize(rand_tree, order)
        rand_experiment.visualize(best_cstree, best_order)
        actual_stages = rand_experiment.cstree_to_stages(rand_tree)
        actual_csi_rels = rand_experiment.stages_to_csirels(actual_stages,order)
        learnt_stages = rand_experiment.cstree_to_stages(best_cstree)
        learnt_csi_rels = rand_experiment.stages_to_csirels(learnt_stages,order)

        a = frozenset([(frozenset(a1),frozenset(a2),frozenset(a3),a4) for a1,a2,a3,a4 in actual_csi_rels])
        b = frozenset([(frozenset(a1),frozenset(a2),frozenset(a3),a4) for a1,a2,a3,a4 in learnt_csi_rels])
        print(a)
        print(b)
        AuB = a.union(b)
        AnB = a.intersection(b)
        print(len(AnB)/len(AuB))

        # count matches

def cstree_pred(m,n,vars,train_n, test_n,pred_var,
                use_oracle=False, use_order=False,
                use_dag=True, **kwargs):
    # For each k i.e. hypotheses
    # For each sparsity level
    # Generate m models
    
    ss = [i/(vars-1) for i in range(vars)]

    true_order = [i+1 for i in range(vars)]
    vd = {i+1:[0,1] for i in range(vars)}

    order = [i+1 for i in range(vars)]
    if use_order:
        orders=[order]
    else:
        orders=None

    ss = [i/(vars-1) for i in range(vars)]
    ss = [0.3 for i in range(vars)]

    Ms = [int(2*np.sqrt(i)) for i in range(vars)]

    s_dict = {}
    for s in ss:

        val_s_cstree = 0
        val_s_set    = 0
        for _ in range(m):
            rand_experiment = CStree(vd)
            rand_tree, rand_tree_distr = rand_experiment.random_cstree(order, ss, Ms)
            train_samples = np.array(rand_tree_distr.sample(n))
            if use_oracle:
                oracle=rand_tree_distr.copy()
            
            else:
                oracle=None
            best_cstree, best_order = rand_experiment.learn_observational(train_samples, ak, "minstages", orders=orders, use_dag=use_dag, oracle=oracle, **kwargs)[0]
            corrects,_ = prediction(pred_var, rand_experiment, best_cstree, rand_tree_distr, test_n, true_order, train_samples)
            val_s_cstree+=corrects/(test_n*m)


            # R code to get correct predictions from R
            #val_s_set+=

        s_dict[s]=(val_s_cstree, val_s_set)
        print(s, s_dict)

    np.save('cstree_preds.npy', s_dict)

def cstree_approx():
    pass


def experiment_hsbm_predictions_testbased(k, m, n, predsize, p, orders=None, binary=True):
    # y-axis: correct predictions/m*pred_size
    # x-axis: Sparsity level
    sparsity_levels = [i+1/10 for i in range(10)]

    # Define variables and outcomes
    if binary:
        value_dict = {i+1:[0,1] for i in range(p)}
    else:
        value_dict = {i+1:list(range(2,5)) for i in range(p)}
        
    order = [i+1 for i in range(p)]
    minimal_contexts = [((1,i),) for i in range(k)]

    for s in sparsity_levels:
        tree_object = CStree(value_dict)
        _, rand_tree_distr, _ = tree_object.random_hsbm(k, order, k*[s])

        # Generate a training set of samples
        samples = np.array(rand_tree_distr.sample(n))

        # Learn the model
        learnt_dags = tree_object.learn_observational(samples, None, "bic", minimal_contexts=minimal_contexts, orders=[order])


        learn_cstree_model, _ = tree_object.generate_hsbm(order, k, [d[1] for d in learnt_dags])

        
        prediction(1, tree_object, learn_cstree_model, rand_tree_distr, predsize, order, samples)

        # Learn model from R
        
        

def experiment_cstree_ratios_nodebased(m, n, test, p, M=3, randomized=False, binary=True):
    # y-axis: Ratio of correctly learnt models
    # x-axis: Sparsity level
    sparsity_levels = [i+1/10 for i in range(10)]

    # Define variables and outcomes
    if binary:
        value_dict = {i+1:[0,1] for i in range(p)}
    else:
        value_dict = {i+1:list(range(2,5)) for i in range(p)}
        
    order = [i+1 for i in range(p)]

    for s in sparsity_levels:
        ratio            = 0 # ratio of correct CStrees learnt
        matches          = 0 # number of stages in true CStree learnt from algorithm
        notin            = 0 # number of stages learnt from algorithm not in true CStree
        all_truestages   = 0 # total stages in all diff_numtrees CStrees generated
        all_learntstages = 0 # total stages in all diff_numtrees CStrees learnt
        for _ in range(m):
            tree_object = CStree(value_dict)
            # Generate random tree
            tree, distr = tree_object.random_cstree(order, p*[s], p*[M])

            actual_stages = tree_object.cstree_to_stages(tree) # get the stages of true tree
            
            all_truestages+=len(actual_stages)

            # Generate samples
            samples = np.array(distr.sample(n))

            # Learn tree
            learnt_trees = tree_object.learn_observational(samples, test,
                                                            "bic", orders=[order], use_dag=True)
            learnt_tree = learnt_trees[0][0]
            learnt_stages = tree_object.cstree_to_stages(learnt_tree) # extract the tree since the above gives list of (tree,order) pairs

            all_learntstages+=len(learnt_stages)

            # See how many stages we get
            # the stages are represented as a dictionary where the keys
            # are the level, and the values are a set of 
            # sets (technically set of frozensets) where each
            # set element is the context, which is a sequence of 
            # (variable, value) pairs. Here, we loop over all
            # levels, and for each level, loop over the contexts
            # that fix the stages, and check if that context exists
            # in the set of contexts for the learnt cstree 
            # in the same level (here we use that each stage is 
            # uniquely defined by the level and associated context)
            for level in actual_stages.keys():
                for context in actual_stages[level]:
                    if context in learnt_stages[level]:
                        matches+=1

            # Here we count the stages learnt that are not in the true CStree
            for level in learnt_stages.keys():
                for context in learnt_stages[level]:
                    if context not in actual_stages[level]:
                        notin+=1

            if actual_stages==learnt_stages:
                # Where stages match perfectly
                ratio+=1/m

            # Compute similarities

def experiment_cstree_predictions_nodebased(m, n, test, p, randomized=False, binary=True):
    # y-axis: correct predictions/m*pred_size
    # x-axis: Sparsity level
    sparsity_levels = [i+1/10 for i in range(10)]

    # Define variables and outcomes
    if binary:
        value_dict = {i+1:[0,1] for i in range(p)}
    else:
        value_dict = {i+1:list(range(2,5)) for i in range(p)}
        
    order = [i+1 for i in range(p)]

    for s in sparsity_levels:
        tree_object = CStree(value_dict)
        # Generate random tree
        tree, distr = tree_object.random_cstree(order, p*[s], p*[M])

        # Generate samples
        samples = np.array(distr.sample(n))

        # Learn tree
        learnt_trees = tree_object.learn_observational(samples, test,
                                                        "bic", orders=[order], use_dag=True)
        learnt_tree = learnt_trees[0][0]

        # Jacard metric instead of what you have right now


# Oracle based experiments
# Compare with R stagedtrees package
# Compute BIC using distribution covariance
def experiment_hsbm_ratios_oracle(k, m):
    pass


def experiment_cstree_ratios_oracle(m):
    pass

def experiment_cstree_predictions_oracle(m):
    pass

# Define the oracle tests for context based methods



# Real data comparison
# Finetuned configurations
# Compare with R stagedtrees package
