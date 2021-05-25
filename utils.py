import numpy as np
np.set_printoptions(suppress=True)
import networkx as nx
import pandas as pd

# This function generates data according to a DAG provided in list_vertex and list_edges with mean and variance as input
# It will apply a perturbation at each node provided in perturb.
def gen_data_nonlinear(G, base_mean = 0, base_var = 0.3, mean = 0, var = 1, SIZE = 10000, err_type = 'normal', perturb = [], sigmoid = True, expon = 1.1):
    list_edges = G.edges()
    list_vertex = G.nodes()

    order = []
    for ts in nx.algorithms.dag.topological_sort(G):
        order.append(ts)

    g = []
    for v in list_vertex:
        if v in perturb:
            g.append(np.random.normal(mean,var,SIZE))
            print("perturbing ", v, "with mean var = ", mean, var)
        else:
            if err_type == 'gumbel':
                g.append(np.random.gumbel(base_mean, base_var,SIZE))
            else:
                g.append(np.random.normal(base_mean, base_var,SIZE))
            


    for o in order:
        for edge in list_edges:
            if o == edge[1]: # if there is an edge into this node
                if sigmoid:
                    g[edge[1]] += 1/1+np.exp(-g[edge[0]])
                else:
                    g[edge[1]] += g[edge[0]] ** 2
                    #if edge[0] % 2 == 0:
                    #    g[edge[1]] += np.power(np.abs(g[edge[0]]), expon) 
                    #else:
                    #    g[edge[1]] += np.log(np.abs(g[edge[0]])) * edge[0]
                    #g[edge[1]] +=  g[edge[0]]
    g = np.swapaxes(g,0,1)

    return pd.DataFrame(g, columns = list(map(str, list_vertex)))

  
def handler(signal_received, frame):
    # Handle any cleanup here
    print('SIGINT or CTRL-C detected. Exiting gracefully')
    exit(0)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.') 
     
        
def binary_sampler(p, rows, cols):
  '''Sample binary random variables.
  
  Args:
    - p: probability of 1
    - rows: the number of rows
    - cols: the number of columns
    
  Returns:
    - binary_random_matrix: generated binary random matrix.
  '''
  unif_random_matrix = np.random.uniform(0., 1., size = [rows, cols])
  binary_random_matrix = 1*(unif_random_matrix < p)
  return binary_random_matrix