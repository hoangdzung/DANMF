from danmf import DANMF
from parser import parameter_parser
from utils import read_graph, tab_printer, loss_printer, LFR, girvan_graphs
<<<<<<< HEAD
import numpy as np
import pickle
=======
>>>>>>> 3713cd159c4c4c240d3c37a1412fb3073d4c57e1
import networkx as nx

def main():
    """
    Parsing command lines, creating target matrix, fitting DANMF and saving the embedding.
    """

    args = parameter_parser()
    np.random.seed(args.seed)
    tab_printer(args)
    if args.adj_npy is not None:
        graph = nx.from_numpy_array(np.load(args.adj_npy))
    elif args.zout:
        graph = girvan_graphs(args.zout)
        args.layers = args.layers + [4]
    elif args.mu:
<<<<<<< HEAD
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
=======
        graph = LFR(args.zout)
        node2com=dict()
        n_com = 0
        for node in self.graph.nodes():
            is_old = False
            for i in self.graph.nodes[node]['community']:
                if i in node2com:
                    is_old = True
                    break
                node2com[i]=n_com
            if not is_old:
                n_com+=1
        args.layers = args.layers + [n_com]
>>>>>>> 3713cd159c4c4c240d3c37a1412fb3073d4c57e1
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
