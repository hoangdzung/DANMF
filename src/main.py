from danmf import DANMF
from parser import parameter_parser
from utils import read_graph, tab_printer, loss_printer, LFR, girvan_graphs

def main():
    """
    Parsing command lines, creating target matrix, fitting DANMF and saving the embedding.
    """
    args = parameter_parser()
    tab_printer(args)
    if args.zout:
        graph = girvan_graphs(args.zout)
        args.layers = args.layers + [4]
    elif args.mu:
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
