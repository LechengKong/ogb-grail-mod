import os
from subgraph_extraction.datasets import SubgraphDatasetWikiDynamic, SubgraphDatasetWikiLocal, SubgraphDatasetWikiEval, SubgraphDatasetWikiLocalEval, SubgraphDatasetWikiLocalTest, SubgraphDatasetWikiOnline
from ogb.lsc import WikiKG90MDataset as wikiData
import torch
import numpy as np
from utils.graph_utils import collate_dgl, move_batch_to_device_dgl, collate_dgl_val, move_batch_to_device_dgl_val
from model.dgl.graph_classifier import GraphClassifier as dgl_model
from managers.trainer import Trainer
from managers.evaluator import Evaluator
import lmdb
import random
from tqdm import tqdm
import multiprocessing as mp
import pickle


class Mem:

    def __init__(self):
        self.hop = 1
        self.enclosing_sub_graph = False
        self.max_nodes_per_hop = 100
        self.num_neg_samples_per_link = 2
        self.root_path = "/project/tantra/jerry.kong/ogb_project/dataset/wikikg90m_kddcup2021"
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
        self.optimizer = 'Adam'
        self.lr = 0.001
        self.l2 = 5e-4
        self.batch_size = 32
        self.num_workers = 16
        self.num_epochs = 20
        self.save_every = 1
        self.exp_dir = "/project/tantra/jerry.kong/ogb_project/dataset/wikikg90m_kddcup2021/"
        self.margin = 10
        self.train_edges = 10000
        self.val_size = 1000
        self.eval_every_iter = 3
        self.early_stop = 3
        self.split = 'val'
        self.make_data = False
        self.val_batch_size = 1
        self.candidate_size = 1001
        self.prefetch_val = 1


def initializer(train_d):
    global train_
    train_ = train_d


def prepare_data(idx):
    pos_g, pos_la, pos_rel, neg_g, neg_la, neg_rel = train_[idx]
    datum = {'pos_g': pos_g, 'pos_la': pos_la, 'pos_rel': pos_rel, 'neg_g': neg_g, 'neg_la': neg_la,
             'neg_rel': neg_rel}
    tp = tuple(datum.values())
    s = pickle.dumps(tp)
    str_id = '{:08}'.format(idx).encode('ascii')
    return str_id, s
    # s = pickle.dumps((1,1,2,3))
    # str_id = '{:08}'.format(idx).encode('ascii')
    # return str_id, s


def prepare_g(idx):
    g = train_[idx]
    s = pickle.dumps(g)
    str_id = '{:08}'.format(idx).encode('ascii')
    return str_id, s


if __name__ == '__main__':
    torch.manual_seed(10)
    random.seed(10)
    np.random.seed(10)

    params = Mem()

    params.db_path = os.path.join(params.root_path,
                                  f'dbs/subgraphs_en_{params.enclosing_sub_graph}_neg_{params.num_neg_samples_per_link}_hop_{params.hop}')
    params.db_path_val = os.path.join(params.root_path,
                                  f'dbs/subgraphs_en_{params.enclosing_sub_graph}_neg_{params.num_neg_samples_per_link}_hop_{params.hop}_split_{params.split}')
    dataset = wikiData(root="/project/tantra/jerry.kong/ogb_project/dataset")
    params.num_rels = dataset.num_relations

    if torch.cuda.is_available():
        params.device = torch.device('cuda:3')
    else:
        params.device = torch.device('cpu')

    params.collate_fn = collate_dgl
    params.collate_fn_val = collate_dgl_val
    params.move_batch_to_device = move_batch_to_device_dgl
    params.move_batch_to_device_val = move_batch_to_device_dgl_val
    rel_to_id = np.arange(params.num_rels)
    if (not os.path.exists(params.db_path)) or params.make_data:
        print("Create Dataset")
        if params.split == 'val':
            print("val dataset")
            train = SubgraphDatasetWikiEval(dataset, params, params.adj_path,
                                               neg_link_per_sample=params.num_neg_samples_per_link, use_feature=False,
                                               sample_size=params.train_edges)
            size = 0
            sample_size = 10
            for i in range(sample_size):
                g = train[i]
                s = len(pickle.dumps(g))
                size += s
            ave_size = size/sample_size
            print('ave_size:', ave_size)
            map_size = ave_size * params.train_edges * 2
            env = lmdb.open(params.db_path, map_size=int(map_size), max_dbs=3)
            db_name = params.split
            split_env = env.open_db(db_name.encode())

            def extraction_helper(data, split_env):
                with env.begin(write=True, db=split_env) as txn:
                    txn.put('num_graphs'.encode(),
                            params.train_edges.to_bytes(int.bit_length(params.train_edges), byteorder='little'))

                with env.begin(write=True, db=split_env) as txn:
                    with mp.Pool(processes=None, initializer=initializer, initargs=data) as p:
                        for str_id, datum in tqdm(p.imap(prepare_g, range(params.train_edges)), total=params.train_edges):
                            txn.put(str_id, datum)


            extraction_helper([train], split_env)
        elif params.split == 'train':
            train = SubgraphDatasetWikiDynamic(dataset, params, params.adj_path,
                                               neg_link_per_sample=params.num_neg_samples_per_link, use_feature=False,
                                               sample_size=params.train_edges)
            size = 0
            sample_size = 10
            for i in range(sample_size):
                pos_g, pos_la, pos_rel, neg_g, neg_la, neg_rel = train[i]
                datum = {'pos_g': pos_g, 'pos_la': pos_la, 'pos_rel': pos_rel, 'neg_g': neg_g, 'neg_la': neg_la,
                         'neg_rel': neg_rel}
                tp = tuple(datum.values())
                s = len(pickle.dumps(tp))
                size += s
            ave_size = size / sample_size
            print('ave_size:', ave_size)
            map_size = ave_size * params.train_edges * 2
            env = lmdb.open(params.db_path, map_size=int(map_size), max_dbs=3)
            db_name = 'train'
            split_env = env.open_db(db_name.encode())


            def extraction_helper(train, split_env):
                with env.begin(write=True, db=split_env) as txn:
                    txn.put('num_graphs'.encode(),
                            params.train_edges.to_bytes(int.bit_length(params.train_edges), byteorder='little'))

                with env.begin(write=True, db=split_env) as txn:
                    with mp.Pool(processes=None, initializer=initializer, initargs=train) as p:
                        for str_id, datum in tqdm(p.imap(prepare_data, range(params.train_edges)),
                                                  total=params.train_edges):
                            txn.put(str_id, datum)


            extraction_helper([train], split_env)
        print("subgraph processed")
    torch.multiprocessing.set_sharing_strategy('file_system')
    train_size = 1
    train_end = int(params.train_edges*train_size)
    train_ind = np.arange(train_end)
    test_ind = np.arange(train_end, params.train_edges)
    # train = SubgraphDatasetWikiLocal(dataset, params, params.db_path, 'train', sample_size=len(train_ind), db_index=train_ind, neg_link_per_sample=params.num_neg_samples_per_link, use_feature=True)
    train = SubgraphDatasetWikiOnline(dataset, params, params.adj_path, sample_size=params.train_edges, neg_link_per_sample=100, use_feature=True)
    test = SubgraphDatasetWikiLocalEval(dataset, params, params.db_path_val, 'val', sample_size=params.val_size, neg_link_per_sample=params.num_neg_samples_per_link, use_feature=True)
    # test = SubgraphDatasetWikiLocalTest(dataset, params, params.db_path, 'train', sample_size=len(train_ind), db_index=train_ind, neg_link_per_sample=params.num_neg_samples_per_link, use_feature=True)
    params.inp_dim = train.n_feat_dim

    graph_classifier = dgl_model(params, rel_to_id).to(device=params.device)
    validator = Evaluator(params, graph_classifier, test)
    trainer = Trainer(params, graph_classifier, train, valid_evaluator=validator)

    trainer.train()
