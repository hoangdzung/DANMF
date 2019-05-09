import pandas as pd
import networkx as nx
from texttable import Texttable
import random
from networkx.algorithms.community.community_generators import LFR_benchmark_graph

random.seed(0)
def LFR(mu):
    return LFR_benchmark_graph(n=1000,tau1=2, tau2=1.5, mu=mu, average_degree=15, max_degree=50,min_community=20, max_community=60,seed=0)

def girvan_graphs(zout):
    """
    Create a graph of 128 vertices, 4 communities, like in
    Community Structure in  social and biological networks.
    Girvan newman, 2002. PNAS June, vol 99 n 12
    community is node_1 modulo 4
    """

    pout = float(zout) / 96.
    pin = (16. - pout * 96.) / 31.
    graph = nx.Graph()
    graph.add_nodes_from(range(128))
    for node_1 in graph.nodes():
        for node_2 in graph.nodes():
            if node_1 < node_2:
                val = random.random()
                if node_1 % 4 == node_2 % 4:
                    # nodes belong to the same community
                    if val < pin:
                        graph.add_edge(node_1, node_2)

                else:
                    if val < pout:
                        graph.add_edge(node_1, node_2)
    return graph

def read_graph(args):
    """
    Method to read graph and create a target matrix with pooled adjacency matrix powers up to the order.
    :param args: Arguments object.
    """
    print("\nTarget matrix creation started.\n")
    graph = nx.from_edgelist(pd.read_csv(args.edge_path).values.tolist())
    return graph

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(t.draw())

def loss_printer(losses):
    """
    Printing the losses for each iteration.
    :param losses: List of losses in each iteration.
    """
    t = Texttable() 
    t.add_rows([["Iteration","Reconstrcution Loss I.","Reconstruction Loss II.","Regularization Loss"]] +  losses)
    print(t.draw())
