# Python imports
import sys
from functools import reduce
from itertools import product, combinations
import random

# Third-party library imports
import networkx as nx
import numpy as np
from scipy.stats import epps_singleton_2samp,anderson_ksamp


vars_of_context = lambda context: [] if context==() else  [var for (var,_) in context]


def reservoir_sample(iterable, n):
    # From https://stackoverflow.com/questions/2612648/reservoir-sampling
    """
    Returns @param n random items from @param iterable.
    """
    reservoir = []
    for t, item in enumerate(iterable):
        if t < n:
            reservoir.append(item)
        else:
            m = random.randint(0,t)
            if m < n:
                reservoir[m] = item
    return reservoir

def test_skl_divergence(sample1, sample2,
                   outcomes,
                   threshold=1e-2,
                   epsilon=1e-50):
    if len(sample1)>0 and len(sample2)>0:
        distr1 =  [list(sample1).count(i)/len(sample1)
                   for i in outcomes]
        distr2 =  [list(sample2).count(i)/len(sample2)
                   for i in outcomes]

        # Handle possilbe 0 samples for any outcome
        for i in outcomes:
            if distr1[i]<epsilon:
                distr1[i]+=epsilon
            if distr2[i]<epsilon:
                distr2[i]+=epsilon

        kl = lambda d1,d2: sum([d1[i]*np.log(d1[i]/d2[i]) for i in range(len(d1))])
        symmetric_kl = kl(distr1,distr2)+kl(distr2,distr1)
        if symmetric_kl<threshold:
            return True
        else:
            return False
    # If a distribution has no sample, assign high value i.e. say
    # they dont belong to the same stage
    else:
        return False        
        
    

def test_anderson_k(sample1, sample2, outcomes, p=0.01):
    try:
        _, _, actual_p = anderson_ksamp([sample1, sample2])
        if actual_p>p:
            return True
        else:
            return False
    except:
        return False


def test_epps(sample1, sample2, outcomes, p=0.01):
    try:
        _, actual_p = epps_singleton_2samp(sample1,sample2)
        if actual_p>p or actual_p is float("NaN"):
            return True
        else:
            return False
    except:
        return False


"""def criteria_stages(tree, order):
    # x-1 because the criteria is designed to be
    # good when higher
    return -1*tree.num_stages()

def criteria_bic(tree_object, tree, order):
    # TODO This can be designe better
    return tree.bic(tree, order)"""


# State space dictionaries
# binary, ternary, random ints, one dict with 1 value
binary_dict  = lambda p: {i+1:[0,1] for i in range(p)}
ternary_dict = lambda p: {i+1:[0,1,2] for i in range(p)}
mixed_dict   = lambda p: {i+1:[0,1] if i < p//2 else [0,1,2] for i in srange(p)}
# CSI relations

def generate_dag1(nodes, p_edge):
    # Generate random DAG by first generating
    # fully connected DAG
    full_dag = nx.complete_graph(nodes, nx.DiGraph())
    to_remove=[]
    for u,v in full_dag.edges:
        r = np.random.uniform()
        if r>p_edge:
            to_remove.append((u,v))
            to_remove.append((v,u))
        if u>v:
            # to ensure ordering
            to_remove.append((u,v))
    full_dag.remove_edges_from(to_remove)
    full_dag = nx.relabel_nodes(full_dag, lambda x: x+1)
    return full_dag


def generate_dag(nodes, p_edge):
    rand_graph = nx.gnp_random_graph(nodes,p_edge,directed=True)
    dag        = nx.DiGraph()
    dag.add_edges_from([(u,v) for (u,v) in rand_graph.edges if u<v])
    dag.add_nodes_from([i for i in range(nodes)])
    dag = nx.relabel_nodes(dag, lambda x: x+1)
    return dag


# DEPRECATED
def nodes_per_tree(val_dict, ordering, include_last=False):
    assert len(val_dict)==len(ordering)

    # nodes per each level
    nodes=[]
    nodes.append(len(val_dict[ordering[0]]))
    
    levels = len(ordering)
    
    if not include_last:
        levels-=1
    
    for i in range(1,levels):
        nodes.append(nodes[i-1]*len(val_dict[ordering[i]]))
        
    return sum(nodes)


def contained(p, rels, s):
    # checking whether there is any s' in the conditioning
    # subsets already d-separating the pair p such that 
    # s' is contained in the new subset s
    # because in this case we already know they are 
    # d-separated
    
    # take the relevant ci relations
    relevant = list(filter(lambda r: r[0] == p, rels))
    
    # get the ones where an existing conditioning set is a subset of s
    r = list(map(lambda r: s.issubset(r[-1]), relevant))
    
    return not reduce(lambda a,b:a and b, r, True)



def flatten(T,a):
    if not isinstance(T,tuple): return (T,)
    else:
        for i in T:
            if isinstance(i, (list,tuple)):
                for j in flatten(i,a):
                    yield j
            else:
                yield i


# reduce for and
and_ = lambda l: reduce(lambda x,y:x and y, l, False)

# check if two edges are undirected versions of each other
edge_opposite = lambda e1,e2:  e1[0]==e2[1] and e1[1]==e2[0]               

def v_structure(e1, e2, g):
    # Return true if two edges form a v-structure in g
    # check if they are head to head, AND uncoupled
    heads_meet = e1[1]==e2[1]
    tails_dont = e1[0]!=e2[0]
    all_edges  = list(g.edges)
    no_other_tail_edge = ((e1[0], e2[0]) not in all_edges) and ((e2[0], e1[0]) not in all_edges) 
    return heads_meet and tails_dont and no_other_tail_edge


def coming_in(edge, g):
    es = list(g.edges)
    return [e for e in es if e[1]==edge[1]]

def v_structure_graph(g):
    es = list(g.edges)
    all_pairs = list(itertools.combinations(es,2))
    return and_(list(map( lambda x: v_structure(x[0],x[1], g), all_pairs)))

def undirected(es):
    es=list(es)
    n = len(es)
    u_es = []
    for i in range(n):
        for j in range(i,n):
            if es[i][0] == es[j][1] and es[i][1] == es[j][0]:
                u_es.append(es[i])
    return u_es

def undirected_both(es):
    es=list(es)
    n = len(es)
    u_es = []
    for i in range(n):
        for j in range(i,n):
            if es[i][0] == es[j][1] and es[i][1] == es[j][0]:
                u_es.append(es[i])
                u_es.append((es[i][1], es[i][0]))
    return u_es

def directed(es):
    u_es = undirected(es)
    all_u_es = u_es+[(y,x) for (x,y) in u_es]
    return [d_e for d_e in es if d_e not in all_u_es]

def remove_cycles(g, count_undirected=True):
    cycles_removed=False
    while not cycles_removed:
        try:
            cycles =list(nx.simple_cycles(g))
            # undirected edges do not count as a cycle
            if count_undirected:
                cycles = [cycle for cycle in cycles if len(cycle)>=2]
            else:
                cycles = [cycle for cycle in cycles if len(cycle)>2]
            
            for cycle in cycles:
                edges_to_remove = [(cycles[i][-1],cycles[i][0]) for i in range(len(cycles))]
                cycles.remove(cycle)
            g.remove_edges_from(edges_to_remove)
            if cycles==[]:
                cycles_removed=True
        except:
            cycles_removed=True
    return g

def dag_to_cpdag(g):
    def order_edges(g):
        sorted_nodes = list(nx.topological_sort(g))
        unordered_edges = list(g.edges)
        ordered_edges = []
        while unordered_edges != []:
            y = [node for node in sorted_nodes if any([True if u in g.in_edges(node) else False for u in unordered_edges])][0]
            x = [node for node in sorted_nodes if (node, y) in unordered_edges][-1]
            ordered_edges.append((x,y))
            unordered_edges.remove((x,y))
        return ordered_edges
            
    edge_labels = {}
    ordered_edges = order_edges(g.copy())
    for edge in ordered_edges:
        edge_labels[edge]=0
    while 0 in list(edge_labels.values()):
        goto3=False
        (x,y) = [o for o in ordered_edges if edge_labels[o]==0][0]
        compelled_edges = [o for o in ordered_edges if edge_labels[o]==1]
        for (w,x) in compelled_edges:
            if w not in parents(g,y):
                edge_labels[(x,y)]=1
                for e_1 in list(g.in_edges(y)):
                    edge_labels[e_1]=1
                goto3=True
            
            if goto3:
                continue
            else:
                edge_labels[(w,y)]=1
        if goto3:
            continue
        cond_list = [e for e in list(g.edges) if (e[1]==y) and (e[0]!=x) and (e[0] not in parents(g,x))]
        temp = [e for e in g.in_edges(y) if edge_labels[e]==0]
        if len(cond_list)>0:
            edge_labels[(x,y)]=1
            for e_2 in temp:
                edge_labels[e_2]=1
        else:
            edge_labels[(x,y)]=2
            for e_3 in temp:
                edge_labels[e_3]=2

    cpdag=nx.DiGraph()
    cpdag.add_nodes_from(list(g.nodes))
    for (ed,label) in list(edge_labels.items()):
        u,v=ed[0],ed[1]
        assert label!=0
        if label==1:
            cpdag.add_edge(u,v)
        elif label==2:
            cpdag.add_edge(v,u)
            cpdag.add_edge(u,v)
        else:
            raise ValueError("unlabelled edge")
    return cpdag

def dag_to_cpdag1(g):
    fixed_edges=set()
    # Gather all v-structures
    for node in g.nodes:
        in_edges=g.in_edges(node)
        for (e1,e2) in combinations(in_edges,2):
            u1,u2=e1[0],e2[0]
            if (u1,u2) not in g.edges and (u2,u1) not in g.edges:
                fixed_edges=fixed_edges.union({e1,e2})
                
    dag = nx.DiGraph()
    dag.add_edges_from(list(fixed_edges))
    dag.add_nodes_from(list(g.nodes))
    for (u,v) in g.edges:
        if (u,v) not in fixed_edges:
            # potential edges which ae not fixed by v structures
            to_add = (v,u)

            # edges incident on node u, we want to make sure no edges of the form (x,u)
            # make a v structure
            potential_v_struct_es = dag.in_edges(u)
            if all([not v_structure(to_add, e, dag) for e in potential_v_struct_es]):
            #if not v_structure((u,v),(v,u),dag):
                dag.add_edge(v,u)
            try:
                cycles =list(nx.simple_cycles(dag))
                if len(cycles)>0:
                    dag.remove_edge(v,u)
            except:
                pass
    return dag

def P_dag_to_dags(g):
    undirected_edges = undirected(g.edges)
    directed_edges   = directed(g.edges)
    print(undirected_edges)

    def list_powerset2(lst):
        return reduce(lambda result, x: result + [subset + [x] for subset in result],
                  lst, [[]])
    all_dags=[]
    subsets = list_powerset2(list(undirected_edges))
    for subset in subsets:
        dag = nx.DiGraph()
        dag.add_nodes_from(list(g.nodes))
        not_included = [e for e in undirected_edges if e not in subset]
        other_direction = [(v,u) for u,v in not_included]
        dag.add_edges_from(list(subset)+other_direction+directed_edges)
        all_dags.append(dag)
        try:
            cycles =list(nx.simple_cycles(dag))
            print("cycls",cycles,list(dag.edges))
            if len(cycles)!=0:
                all_dags.pop(-1)
        except:
            pass
           
    return all_dags

def cpdag_to_dags(g):
    
    undirected_edges = undirected(g.edges)
    directed_edges   = directed(g.edges)
    
    if undirected_edges == []:
        #print("ha")
        try:
            cycles =list(nx.simple_cycles(g))
            if len(cycles)==0:
                yield g
        except:
            pass
    else:
        
        u = undirected_edges.pop()
        # pick just the first undirected edge, then recursively move on

        # Test first for the orientation (u,v)
        u1 = (u[0],u[1])
        # Get all the edges that come into v in g
        
        # TODO Generator here
        connected_u1 = coming_in(u1,g)
       

        # Filter the edges which were originally directed in the CPDAG
        
        # TODO Generator here
        c_and_d1 = [i for i in connected_u1 if i in directed_edges]

        # If the above list is empty, no possible v structure can be formed,
        # so set (u,v) as a directed by (by removing (v,u)) and recurse
        if not c_and_d1:
            new_g3 = g.copy()
            new_g3.remove_edge(u[1],u[0])
            yield from cpdag_to_dags(new_g3)
        else:
            # If there are edges of the form (_,v)
            for e in c_and_d1:
                # If they do not form a v structure in g, we are good
                if not v_structure(u1,e,g):
                    new_g1 = g.copy()
                    # Set (u,v) as a directed edge (by removing (v,u)) and recurse if it doesnt ca
                    new_g1.remove_edge(u[1],u[0])
                    yield from cpdag_to_dags(new_g1)

        # Then for the orientation (v,u)
        u2 = (u[1],u[0])
        connected_u2 = coming_in(u2,g)            
        c_and_d2 = [i for i in connected_u2 if i in directed_edges]
        if not c_and_d2:
            new_g4 = g.copy()
            new_g4.remove_edge(u[0],u[1])
            yield from cpdag_to_dags(new_g4)
        else:
            for e in c_and_d2:
                if not v_structure(u2,e,g):
                    new_g2 = g.copy()
                    new_g2.remove_edge(u[0],u[1])
                    yield from cpdag_to_dags(new_g2)
                    
def mixdag_to_dags(g):
    pass
                    
                    
def parents(g,node):
    return list(g.predecessors(node))

def generate_vals(T, state_space_dict):
    # generate all possible values for the variables in subset T 
    # list of lists containing the values 
    s_vals = [state_space_dict[t] for t in T]
    all_vals_prod = list(product(*s_vals)) # all values without variables in front 
    f = lambda tup: [(T[i], tup[i]) for i in range(len(T))]
    return list(map(f,all_vals_prod))


def generate_state_space(data):
    # Works under the assumption that the data has all values for each variable occuring atleast once
    return {i+1:list(np.unique(data[:,i])) for i in range(data.shape[1])}

dag_topo_sort = lambda dag: list(nx.topological_sort(dag))


def shared_contexts(cs1,cs2):
    return [c for c in cs1 if (c in cs1 and c in cs2)]


def vars_of_context(context):
    return [var for (var,val) in context]



def context_per_stage(ls):
    # Takes nodes in a stage and returns the common context they represent
    sc = shared_contexts(ls[0],ls[1])
    for l in ls[2:]:
        sc = shared_contexts(sc,l)
    return sc

def data_to_contexts(data : np.ndarray, 
                     cs   : list[tuple[int,int]], 
                     var  = None) -> np.ndarray:
    # returns the data where the contexts are of the relevant form
    # confirm if same contexts havent been put
    # TODO consider moving to dask later on
    
    # NOTE
    # When accessing columns of data, we must reduce 1 since data matrix indexing starts with 0
    # and the var indexing starts from 1
    
    p = data.shape[1]
    #c=cs[0]


    for c in cs:
        if var:
            if var == c[0]:
                raise ValueError("The variable you want is being fixed in the context")
        data = data[np.where(data[:,c[0]-1]==c[1])]
    if var:
        return data[:,var-1] #-1 because indices are labelled from 0
    else:
        return data


def context_is_contained(c, cs, possible_mc):
    # Given a context c, and a list of contexts cs,
    # return the context x in cs if c is a subcontext of x
    # and there is no context variable in cs such that
    # it is the same as a context variable in c
    contained=None
    for context in cs:
        possible_mc_here = set([(var,val) for (var,val) in context if (var,val) not in c])
        
        if set(c).issubset(context) and possible_mc.issubset(possible_mc_here):
            contained=context
            
            break
        
    return contained


def get_size(obj, seen=None):
    """Recursively finds size of objects
    Code from stackoverflow"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size
