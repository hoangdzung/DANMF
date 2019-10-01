from danmf import DANMF
from parser import parameter_parser
from utils import read_graph, tab_printer, loss_printer, LFR, girvan_graphs
import numpy as np
import pickle
import networkx as nx

def main():
    """
    Parsing command lines, creating target matrix, fitting DANMF and saving the embedding.
    """

    args = parameter_parser()
    np.random.seed(args.seed)
    tab_printer(args)
    if args.zout:
        graph = girvan_graphs(args.zout)
    elif args.mu:
        graph = LFR(args.mu)
    elif args.edge_path:
        graph = nx.read_edgelist(args.edge_path, nodetype=int)
    elif args.adj_path:
        if args.adj_path.endswith('npy'):
            adj = np.load(args.adj_path)
        elif args.adj_path.endswith('pkl'):
            adj,_,_,_ = pickle.load(open(args.adj_path,'rb'))
        adj[adj<0]=0
        graph = nx.from_numpy_matrix(adj)
    elif args.graph_path:
        graph=pickle.load(open(args.graph_path,'rb'))
    else:
        raise NotImplementedError
    # graph = read_graph(args)
    model = DANMF(graph, args)
    model.pre_training()
    model.training()
    if args.calculate_loss: 
        loss_printer(model.loss)

if __name__ =="__main__":
    main()
