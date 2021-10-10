import numpy as np
import networkx as nx

from .cstree import CStree

# Sparsity and its effect from just samples
# Generate sparsiy levels
# For each sparsity level, generate m random models
# For each model, generate n samples
def experiment_hsbm_ratios_testbased(k, m, n, p, binary=True, use_oracle=False):
    # Plot 1
    # y-axis: Ratio of correctly learnt models
    # x-axis: Sparsity level
    sparsity_levels = [(i+1)/10 for i in range(10)]

    # Define variables and outcomes
    if binary:
        value_dict = {i+1:[0,1] for i in range(p)}
    else:
        value_dict = {i+1:list(range(2,5)) for i in range(p)}
        
    order = [i+1 for i in range(p)]
    minimal_contexts = [((1,i),) for i in range(k)]
        
    for s in sparsity_levels:
        tree_object  = CStree(value_dict)
        _, rand_tree_distr, actual_dags = tree_object.random_hsbm(k, order, k*[s])
        samples = np.array(rand_tree_distr.sample(n))

        if use_oracle:
            oracle = rand_tree_distr.copy()
        else:
            oracle=None
        learnt_dags = tree_object.learn_observational(samples, None, "bic", minimal_contexts=minimal_contexts, orders=[order], oracle=oracle)

        AnB=0
        AuB=1e-50
        for i in range(k):
            AnB += len(set(actual_dags[i].edges).intersection(set(learnt_dags[i][1].edges)))
            AuB += len(set(actual_dags[i].edges).union(set(learnt_dags[i][1].edges)))
        print(AnB/AuB)
            
        # Compute similarities and store them
        # Compute when they are both same and store them
        
def prediction(var, tree_object, learnt_tree, distribution, predsize, order, train_samples):
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
            xs_cf.append(x_copy)
        probs = [tree_object.predict(learnt_tree, order, x_cf, var, train_samples) for x_cf in xs_cf]
        predicted = probs.index(max(probs))
        if predicted==actual:
            correct+=1
        else:
            if (list(x), predicted, probs) not in wrong_unique:
                wrong_unique.append((list(x), predicted, probs))
    print(correct/predsize, len(wrong_unique))
    return correct/predsize, wrong_unique
        
        
        
        
    

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
        # Generate a test set of samples
        
        # Get the prediction accuracy
        
        

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
