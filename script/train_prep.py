import os
from subgraph_extraction.datasets import SubgraphDatasetWiki
from ogb.lsc import WikiKG90MDataset as wikiData


class Mem:

    def __init__(self):
        self.hop = 1
        self.enclosing_sub_graph = True
        self.max_nodes_per_hop = None
        self.db_path = "/project/tantra/jerry.kong/ogb_project/dataset/wikikg90m_kddcup2021/dbs/mdb"
        self.adj_path = "/project/tantra/jerry.kong/ogb_project/dataset/wikikg90m_kddcup2021/adj_mat"
        self.num_rels = 1315
        self.rel_emb_dim = 32
        self.add_ht_emb = True
        self.num_gcn_layers = 3
        self.emb_dim = 32
        self.max_label_value = 10
        self.inp_dim = 1010
        self.attn_rel_emb_dim = 32
        self.aug_num_rels = 1315
        self.num_bases = 4
        self.num_hidden_layers = 2
        self.dropout = 0
        self.edge_dropout = 0.5
        self.has_attn = True
        self.gnn_agg_type = 'sum'

params = Mem()


dataset = wikiData(root="/project/tantra/jerry.kong/ogb_project/dataset")

train = SubgraphDatasetWiki(dataset, params.db_path, 'train_pos', 'train_neg', params.adj_path)

