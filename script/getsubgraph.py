from ogb.lsc import WikiKG90MDataset as wikiData
import time
import argparse
import os
import os.path as osp
import numpy as np
from scipy.sparse import csc_matrix, load_npz


from subgraph_extraction.graph_sampler import sample_neg_wiki, links2subgraphs_wiki


class Mem:

    def __init__(self):
        self.hop = 1
        self.enclosing_sub_graph = True
        self.max_nodes_per_hop = None
        self.db_path = "/project/tantra/jerry.kong/ogb_project/dataset/wikikg90m_kddcup2021/dbs"


path = "/project/tantra/jerry.kong/ogb_project/dataset/wikikg90m_kddcup2021"
read_path = osp.join(path, "adj_mat")
dataset = wikiData(root="/project/tantra/jerry.kong/ogb_project/dataset")

start = time.time()

adj_list = [load_npz(osp.join(read_path, "adj_rel_"+str(i)+".npz")) for i in range(dataset.num_relations)]

print("loaded in "+str(time.time()-start))
pos, neg = sample_neg_wiki(adj_list, dataset.train_hrt, max_size=150000)

graphs = {}
graphs["train"] = {'triplets': dataset.train_hrt, 'max_size': 150000, 'pos': pos, 'neg': neg}

params = Mem()
if not osp.exists(params.db_path):
    os.mkdir(params.db_path)
params.db_path = osp.join(params.db_path, "mdb")
links2subgraphs_wiki(adj_list, graphs, params, None)
