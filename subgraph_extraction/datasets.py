import time

from torch.utils.data import Dataset
import timeit
import os
import logging
import lmdb
import numpy as np
import json
import pickle
import dgl
import multiprocessing
from utils.graph_utils import ssp_multigraph_to_dgl, ssp_multigraph_to_dgl_wiki, construct_graph_from_edges
from utils.data_utils import process_files, save_to_file, plot_rel_dist
from subgraph_extraction.graph_sampler import subgraph_extraction_labeling_wiki, get_neighbor_nodes
from scipy.sparse import load_npz
from .graph_sampler import *
from dgl.contrib.sampling import EdgeSampler
import pdb


def generate_subgraph_datasets(params, splits=['train', 'valid'], saved_relation2id=None, max_label_value=None):
    testing = 'test' in splits
    adj_list, triplets, entity2id, relation2id, id2entity, id2relation = process_files(params.file_paths,
                                                                                       saved_relation2id)

    # plot_rel_dist(adj_list, os.path.join(params.main_dir, f'data/{params.dataset}/rel_dist.png'))

    data_path = os.path.join(params.main_dir, f'data/{params.dataset}/relation2id.json')
    if not os.path.isdir(data_path) and not testing:
        with open(data_path, 'w') as f:
            json.dump(relation2id, f)

    graphs = {}

    for split_name in splits:
        graphs[split_name] = {'triplets': triplets[split_name], 'max_size': params.max_links}

    # Sample train and valid/test links
    for split_name, split in graphs.items():
        logging.info(f"Sampling negative links for {split_name}")
        split['pos'], split['neg'] = sample_neg(adj_list, split['triplets'], params.num_neg_samples_per_link,
                                                max_size=split['max_size'],
                                                constrained_neg_prob=params.constrained_neg_prob)

    if testing:
        directory = os.path.join(params.main_dir, 'data/{}/'.format(params.dataset))
        save_to_file(directory, f'neg_{params.test_file}_{params.constrained_neg_prob}.txt', graphs['test']['neg'],
                     id2entity, id2relation)

    links2subgraphs(adj_list, graphs, params, max_label_value)


def get_kge_embeddings(dataset, kge_model):
    path = './experiments/kge_baselines/{}_{}'.format(kge_model, dataset)
    node_features = np.load(os.path.join(path, 'entity_embedding.npy'))
    with open(os.path.join(path, 'id2entity.json')) as json_file:
        kge_id2entity = json.load(json_file)
        kge_entity2id = {v: int(k) for k, v in kge_id2entity.items()}

    return node_features, kge_entity2id


class SubgraphDataset(Dataset):
    """Extracted, labeled, subgraph dataset -- DGL Only"""

    def __init__(self, db_path, db_name_pos, db_name_neg, raw_data_paths, included_relations=None,
                 add_traspose_rels=False, num_neg_samples_per_link=1, use_kge_embeddings=False, dataset='',
                 kge_model='', file_name=''):

        self.main_env = lmdb.open(db_path, readonly=True, max_dbs=3, lock=False)
        self.db_pos = self.main_env.open_db(db_name_pos.encode())
        self.db_neg = self.main_env.open_db(db_name_neg.encode())
        self.node_features, self.kge_entity2id = get_kge_embeddings(dataset, kge_model) if use_kge_embeddings else (
            None, None)
        self.num_neg_samples_per_link = num_neg_samples_per_link
        self.file_name = file_name

        ssp_graph, __, __, __, id2entity, id2relation = process_files(raw_data_paths, included_relations)
        self.num_rels = len(ssp_graph)

        # Add transpose matrices to handle both directions of relations.
        if add_traspose_rels:
            ssp_graph_t = [adj.T for adj in ssp_graph]
            ssp_graph += ssp_graph_t

        # the effective number of relations after adding symmetric adjacency matrices and/or self connections
        self.aug_num_rels = len(ssp_graph)
        self.graph = ssp_multigraph_to_dgl(ssp_graph)
        self.ssp_graph = ssp_graph
        self.id2entity = id2entity
        self.id2relation = id2relation

        self.max_n_label = np.array([0, 0])
        with self.main_env.begin() as txn:
            self.max_n_label[0] = int.from_bytes(txn.get('max_n_label_sub'.encode()), byteorder='little')
            self.max_n_label[1] = int.from_bytes(txn.get('max_n_label_obj'.encode()), byteorder='little')

            self.avg_subgraph_size = struct.unpack('f', txn.get('avg_subgraph_size'.encode()))
            self.min_subgraph_size = struct.unpack('f', txn.get('min_subgraph_size'.encode()))
            self.max_subgraph_size = struct.unpack('f', txn.get('max_subgraph_size'.encode()))
            self.std_subgraph_size = struct.unpack('f', txn.get('std_subgraph_size'.encode()))

            self.avg_enc_ratio = struct.unpack('f', txn.get('avg_enc_ratio'.encode()))
            self.min_enc_ratio = struct.unpack('f', txn.get('min_enc_ratio'.encode()))
            self.max_enc_ratio = struct.unpack('f', txn.get('max_enc_ratio'.encode()))
            self.std_enc_ratio = struct.unpack('f', txn.get('std_enc_ratio'.encode()))

            self.avg_num_pruned_nodes = struct.unpack('f', txn.get('avg_num_pruned_nodes'.encode()))
            self.min_num_pruned_nodes = struct.unpack('f', txn.get('min_num_pruned_nodes'.encode()))
            self.max_num_pruned_nodes = struct.unpack('f', txn.get('max_num_pruned_nodes'.encode()))
            self.std_num_pruned_nodes = struct.unpack('f', txn.get('std_num_pruned_nodes'.encode()))

        logging.info(f"Max distance from sub : {self.max_n_label[0]}, Max distance from obj : {self.max_n_label[1]}")

        # logging.info('=====================')
        # logging.info(f"Subgraph size stats: \n Avg size {self.avg_subgraph_size}, \n Min size {self.min_subgraph_size}, \n Max size {self.max_subgraph_size}, \n Std {self.std_subgraph_size}")

        # logging.info('=====================')
        # logging.info(f"Enclosed nodes ratio stats: \n Avg size {self.avg_enc_ratio}, \n Min size {self.min_enc_ratio}, \n Max size {self.max_enc_ratio}, \n Std {self.std_enc_ratio}")

        # logging.info('=====================')
        # logging.info(f"# of pruned nodes stats: \n Avg size {self.avg_num_pruned_nodes}, \n Min size {self.min_num_pruned_nodes}, \n Max size {self.max_num_pruned_nodes}, \n Std {self.std_num_pruned_nodes}")

        with self.main_env.begin(db=self.db_pos) as txn:
            self.num_graphs_pos = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')
        with self.main_env.begin(db=self.db_neg) as txn:
            self.num_graphs_neg = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')

        self.__getitem__(0)

    def __getitem__(self, index):
        with self.main_env.begin(db=self.db_pos) as txn:
            str_id = '{:08}'.format(index).encode('ascii')
            nodes_pos, r_label_pos, g_label_pos, n_labels_pos = deserialize(txn.get(str_id)).values()
            subgraph_pos = self._prepare_subgraphs(nodes_pos, r_label_pos, n_labels_pos)
        subgraphs_neg = []
        r_labels_neg = []
        g_labels_neg = []
        with self.main_env.begin(db=self.db_neg) as txn:
            for i in range(self.num_neg_samples_per_link):
                str_id = '{:08}'.format(index + i * (self.num_graphs_pos)).encode('ascii')
                nodes_neg, r_label_neg, g_label_neg, n_labels_neg = deserialize(txn.get(str_id)).values()
                subgraphs_neg.append(self._prepare_subgraphs(nodes_neg, r_label_neg, n_labels_neg))
                r_labels_neg.append(r_label_neg)
                g_labels_neg.append(g_label_neg)

        return subgraph_pos, g_label_pos, r_label_pos, subgraphs_neg, g_labels_neg, r_labels_neg

    def __len__(self):
        return self.num_graphs_pos

    def _prepare_subgraphs(self, nodes, r_label, n_labels):
        subgraph = dgl.DGLGraph(self.graph.subgraph(nodes))
        subgraph.edata['type'] = self.graph.edata['type'][self.graph.subgraph(nodes).parent_eid]
        subgraph.edata['label'] = torch.tensor(r_label * np.ones(subgraph.edata['type'].shape), dtype=torch.long)

        edges_btw_roots = subgraph.edge_id(0, 1)
        rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == r_label)
        if rel_link.squeeze().nelement() == 0:
            subgraph.add_edge(0, 1)
            subgraph.edata['type'][-1] = torch.tensor(r_label).type(torch.LongTensor)
            subgraph.edata['label'][-1] = torch.tensor(r_label).type(torch.LongTensor)

        # map the id read by GraIL to the entity IDs as registered by the KGE embeddings
        kge_nodes = [self.kge_entity2id[self.id2entity[n]] for n in nodes] if self.kge_entity2id else None
        n_feats = self.node_features[kge_nodes] if self.node_features is not None else None
        subgraph = self._prepare_features_new(subgraph, n_labels, n_feats)

        return subgraph

    def _prepare_features(self, subgraph, n_labels, n_feats=None):
        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1))
        label_feats[np.arange(n_nodes), n_labels] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)
        self.n_feat_dim = n_feats.shape[1]  # Find cleaner way to do this -- i.e. set the n_feat_dim
        return subgraph

    def _prepare_features_new(self, subgraph, n_labels, n_feats=None):
        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        # label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        # label_feats[np.arange(n_nodes), 0] = 1
        # label_feats[np.arange(n_nodes), self.max_n_label[0] + 1] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)

        head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
        tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
        n_ids = np.zeros(n_nodes)
        n_ids[head_id] = 1  # head
        n_ids[tail_id] = 2  # tail
        subgraph.ndata['id'] = torch.FloatTensor(n_ids)

        self.n_feat_dim = n_feats.shape[1]  # Find cleaner way to do this -- i.e. set the n_feat_dim
        return subgraph


class SubgraphDatasetWiki(Dataset):
    """Extracted, labeled, subgraph dataset -- DGL Only"""

    def __init__(self, data, db_path, db_name_pos, db_name_neg, adj_path, included_relations=None,
                 add_traspose_rels=False, num_neg_samples_per_link=1, use_kge_embeddings=False, dataset='',
                 kge_model='', file_name=''):

        self.main_env = lmdb.open(db_path, readonly=True, max_dbs=3, lock=False)
        self.db_pos = self.main_env.open_db(db_name_pos.encode())
        self.db_neg = self.main_env.open_db(db_name_neg.encode())
        self.node_features, self.kge_entity2id = get_kge_embeddings(dataset, kge_model) if use_kge_embeddings else (
            None, None)
        self.num_neg_samples_per_link = num_neg_samples_per_link
        self.file_name = file_name
        self.wiki_data = data

        ssp_graph = [load_npz(os.path.join(adj_path, "adj_rel_" + str(i) + ".npz")) for i in
                     range(self.wiki_data.num_relations)]
        self.num_rels = len(ssp_graph)

        # Add transpose matrices to handle both directions of relations.
        if add_traspose_rels:
            ssp_graph_t = [adj.T for adj in ssp_graph]
            ssp_graph += ssp_graph_t

        # the effective number of relations after adding symmetric adjacency matrices and/or self connections
        self.aug_num_rels = len(ssp_graph)
        self.graph = ssp_multigraph_to_dgl_wiki(ssp_graph)
        self.ssp_graph = ssp_graph

        self.max_n_label = np.array([0, 0])
        with self.main_env.begin() as txn:
            self.max_n_label[0] = int.from_bytes(txn.get('max_n_label_sub'.encode()), byteorder='little')
            self.max_n_label[1] = int.from_bytes(txn.get('max_n_label_obj'.encode()), byteorder='little')

            self.avg_subgraph_size = struct.unpack('f', txn.get('avg_subgraph_size'.encode()))
            self.min_subgraph_size = struct.unpack('f', txn.get('min_subgraph_size'.encode()))
            self.max_subgraph_size = struct.unpack('f', txn.get('max_subgraph_size'.encode()))
            self.std_subgraph_size = struct.unpack('f', txn.get('std_subgraph_size'.encode()))

            self.avg_enc_ratio = struct.unpack('f', txn.get('avg_enc_ratio'.encode()))
            self.min_enc_ratio = struct.unpack('f', txn.get('min_enc_ratio'.encode()))
            self.max_enc_ratio = struct.unpack('f', txn.get('max_enc_ratio'.encode()))
            self.std_enc_ratio = struct.unpack('f', txn.get('std_enc_ratio'.encode()))

            self.avg_num_pruned_nodes = struct.unpack('f', txn.get('avg_num_pruned_nodes'.encode()))
            self.min_num_pruned_nodes = struct.unpack('f', txn.get('min_num_pruned_nodes'.encode()))
            self.max_num_pruned_nodes = struct.unpack('f', txn.get('max_num_pruned_nodes'.encode()))
            self.std_num_pruned_nodes = struct.unpack('f', txn.get('std_num_pruned_nodes'.encode()))

        logging.info(f"Max distance from sub : {self.max_n_label[0]}, Max distance from obj : {self.max_n_label[1]}")

        # logging.info('=====================')
        # logging.info(f"Subgraph size stats: \n Avg size {self.avg_subgraph_size}, \n Min size {self.min_subgraph_size}, \n Max size {self.max_subgraph_size}, \n Std {self.std_subgraph_size}")

        # logging.info('=====================')
        # logging.info(f"Enclosed nodes ratio stats: \n Avg size {self.avg_enc_ratio}, \n Min size {self.min_enc_ratio}, \n Max size {self.max_enc_ratio}, \n Std {self.std_enc_ratio}")

        # logging.info('=====================')
        # logging.info(f"# of pruned nodes stats: \n Avg size {self.avg_num_pruned_nodes}, \n Min size {self.min_num_pruned_nodes}, \n Max size {self.max_num_pruned_nodes}, \n Std {self.std_num_pruned_nodes}")

        with self.main_env.begin(db=self.db_pos) as txn:
            self.num_graphs_pos = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')
        with self.main_env.begin(db=self.db_neg) as txn:
            self.num_graphs_neg = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')

        self.__getitem__(0)

    def __getitem__(self, index):
        with self.main_env.begin(db=self.db_pos) as txn:
            str_id = '{:08}'.format(index).encode('ascii')
            nodes_pos, r_label_pos, g_label_pos, n_labels_pos = deserialize(txn.get(str_id)).values()
            subgraph_pos = self._prepare_subgraphs(nodes_pos, r_label_pos, n_labels_pos)
        subgraphs_neg = []
        r_labels_neg = []
        g_labels_neg = []
        with self.main_env.begin(db=self.db_neg) as txn:
            for i in range(self.num_neg_samples_per_link):
                str_id = '{:08}'.format(index + i * self.num_graphs_pos).encode('ascii')
                nodes_neg, r_label_neg, g_label_neg, n_labels_neg = deserialize(txn.get(str_id)).values()
                subgraphs_neg.append(self._prepare_subgraphs(nodes_neg, r_label_neg, n_labels_neg))
                r_labels_neg.append(r_label_neg)
                g_labels_neg.append(g_label_neg)

        return subgraph_pos, g_label_pos, r_label_pos, subgraphs_neg, g_labels_neg, r_labels_neg

    def __len__(self):
        return self.num_graphs_pos

    def _prepare_subgraphs(self, nodes, r_label, n_labels):
        subgraph = dgl.DGLGraph(self.graph.subgraph(nodes))
        subgraph.edata['type'] = self.graph.edata['type'][self.graph.subgraph(nodes).parent_eid]
        subgraph.edata['label'] = torch.tensor(r_label * np.ones(subgraph.edata['type'].shape), dtype=torch.long)

        edges_btw_roots = subgraph.edge_id(0, 1)
        rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == r_label)
        if rel_link.squeeze().nelement() == 0:
            subgraph.add_edge(0, 1)
            subgraph.edata['type'][-1] = torch.tensor(r_label).type(torch.LongTensor)
            subgraph.edata['label'][-1] = torch.tensor(r_label).type(torch.LongTensor)

        # map the id read by GraIL to the entity IDs as registered by the KGE embeddings
        n_feats = self.data.train_hrt[nodes] if self.node_features is not None else None
        subgraph = self._prepare_features_new(subgraph, n_labels, n_feats)

        return subgraph

    def _prepare_features(self, subgraph, n_labels, n_feats=None):
        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1))
        label_feats[np.arange(n_nodes), n_labels] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)
        self.n_feat_dim = n_feats.shape[1]  # Find cleaner way to do this -- i.e. set the n_feat_dim
        return subgraph

    def _prepare_features_new(self, subgraph, n_labels, n_feats=None):
        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        # label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        # label_feats[np.arange(n_nodes), 0] = 1
        # label_feats[np.arange(n_nodes), self.max_n_label[0] + 1] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)

        head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
        tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
        n_ids = np.zeros(n_nodes)
        n_ids[head_id] = 1  # head
        n_ids[tail_id] = 2  # tail
        subgraph.ndata['id'] = torch.FloatTensor(n_ids)

        self.n_feat_dim = n_feats.shape[1]  # Find cleaner way to do this -- i.e. set the n_feat_dim
        return subgraph


class SubgraphDatasetWikiDynamic(Dataset):
    """Extracted, labeled, subgraph dataset -- DGL Only"""

    def __init__(self, data, params, adj_path, sample_size=1000, neg_link_per_sample=1, use_feature=False,
                 shuffle=True):
        self.wiki_data = data
        self.num_rels = data.num_relations
        self.edges = data.train_hrt
        self.num_edges = len(self.edges)
        self.num_nodes = data.num_entities
        self.use_feature = use_feature
        self.params = params
        self.graph = construct_graph_from_edges(self.edges.T, self.num_nodes)
        self.ssp_graph = [load_npz(os.path.join(adj_path, "adj_rel_" + str(i) + ".npz")) for i in
                          range(self.wiki_data.num_relations)]
        self.adj_mat = incidence_matrix_coo(self.ssp_graph)
        self.adj_mat += self.adj_mat.T
        self.max_n_label = [10, 10]
        self.neg_sample = neg_link_per_sample

        self.sample_size = sample_size

        if shuffle:
            self.perm = np.random.permutation(self.num_edges)

        # the effective number of relations after adding symmetric adjacency matrices and/or self connections
        self.aug_num_rels = data.num_relations
        self.__getitem__(0)

    def __len__(self):
        return self.sample_size

    def __getitem__(self, index):
        pos_link = self.edges[self.perm[index]]
        neg_links = []
        for i in range(self.neg_sample):
            neg_links.append(sample_neg_one(self.ssp_graph, pos_link, self.num_nodes))
        pos_nodes, pos_label, _, _, _ = subgraph_extraction_labeling_wiki([pos_link[0], pos_link[2]], pos_link[1],
                                                                          self.adj_mat, max_nodes_per_hop=100)
        pos_subgraph = self.graph.subgraph(pos_nodes)
        pos_subgraph.edata['type'] = self.graph.edata['type'][pos_subgraph.edata[dgl.EID]]
        pos_subgraph.edata['label'] = torch.tensor(pos_link[1] * np.ones(pos_subgraph.edata['type'].shape),
                                                   dtype=torch.long)
        # map the id read by GraIL to the entity IDs as registered by the KGE embeddings
        pos_subgraph = self._prepare_features_new(pos_subgraph, pos_label,
                                                  self.wiki_data.entity_feat[pos_nodes] if self.use_feature else None)
        neg_subgraphs = []
        for i in range(self.neg_sample):
            neg_nodes, neg_label, _, _, _ = subgraph_extraction_labeling_wiki([neg_links[i][0], neg_links[i][2]],
                                                                              neg_links[i][1],
                                                                              self.adj_mat, max_nodes_per_hop=100)

            neg_subgraph = self.graph.subgraph(neg_nodes)
            neg_subgraph.edata['label'] = torch.tensor(neg_links[i][1] * np.ones(neg_subgraph.edata['type'].shape),
                                                       dtype=torch.long)
            neg_subgraph.add_edges([0], [1])
            neg_subgraph.edata['type'][-1] = torch.tensor(neg_links[i][1], dtype=torch.int32)
            neg_subgraph.edata['label'][-1] = torch.tensor(neg_links[i][1], dtype=torch.int32)
            neg_subgraphs.append(self._prepare_features_new(neg_subgraph, neg_label,
                                                            self.wiki_data.entity_feat[
                                                                neg_nodes] if self.use_feature else None))
        if index == self.sample_size:
            self.perm = np.random.permutation(self.num_edges)
        return pos_subgraph, 1, pos_link[1], neg_subgraphs, [0] * len(neg_subgraphs), [neg_links[i][1] for i in
                                                                                       range(len(neg_subgraphs))]

    def _prepare_features_new(self, subgraph, n_labels, n_feats=None):
        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        # label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        # label_feats[np.arange(n_nodes), 0] = 1
        # label_feats[np.arange(n_nodes), self.max_n_label[0] + 1] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)

        head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
        tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
        n_ids = np.zeros(n_nodes)
        n_ids[head_id] = 1  # head
        n_ids[tail_id] = 2  # tail
        subgraph.ndata['id'] = torch.FloatTensor(n_ids)

        self.n_feat_dim = n_feats.shape[1]  # Find cleaner way to do this -- i.e. set the n_feat_dim
        return subgraph


class SubgraphDatasetWikiLocal(Dataset):
    """Extracted, labeled, subgraph dataset -- DGL Only"""

    def __init__(self, data, params, db_path, sub_db_name, sample_size=1000, db_index=None, neg_link_per_sample=1,
                 use_feature=False):
        self.wiki_data = data
        self.use_feature = use_feature
        self.params = params
        self.max_n_label = [10, 10]
        self.neg_sample = neg_link_per_sample
        self.main_env = lmdb.open(db_path, readonly=True, max_dbs=3, lock=False)
        self.sample_size = sample_size
        self.sub_db = self.main_env.open_db(sub_db_name.encode())
        if db_index is None:
            self.perm = np.random.permutation(self.sample_size)
        else:
            self.perm = db_index

        # the effective number of relations after adding symmetric adjacency matrices and/or self connections
        self.aug_num_rels = data.num_relations
        pos_g, pos_la, pos_rel, neg_g, neg_la, neg_rel = self.__getitem__(0)
        self.n_feat_dim = pos_g.ndata['feat'].shape[1]

    def __len__(self):
        return self.sample_size

    def __getitem__(self, index):
        with self.main_env.begin(db=self.sub_db) as txn:
            str_id = '{:08}'.format(index).encode('ascii')
            pos_g, pos_la, pos_rel, neg_g, neg_la, neg_rel = pickle.loads(txn.get(str_id))

        if self.use_feature:
            pos_g = self.prepare_feature(pos_g)
            for i in range(len(neg_g)):
                neg_g[i] = self.prepare_feature(neg_g[i])
        return pos_g, pos_la, pos_rel, neg_g, neg_la, neg_rel

    def prepare_feature(self, subgraph):
        labels = subgraph.ndata['feat']
        n_feats = self.wiki_data.entity_feat[subgraph.ndata[dgl.NID]]
        n_feats = np.concatenate((labels, n_feats), axis=1)
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)
        return subgraph


class SubgraphDatasetWikiLocalTest(Dataset):
    """Extracted, labeled, subgraph dataset -- DGL Only"""

    def __init__(self, data, params, db_path, sub_db_name, sample_size=1000, db_index=None, neg_link_per_sample=1,
                 use_feature=False):
        self.wiki_data = data
        self.use_feature = use_feature
        self.params = params
        self.max_n_label = [10, 10]
        self.neg_sample = neg_link_per_sample
        self.main_env = lmdb.open(db_path, readonly=True, max_dbs=3, lock=False)
        self.sample_size = sample_size
        self.sub_db = self.main_env.open_db(sub_db_name.encode())
        if db_index is None:
            self.perm = np.random.permutation(self.sample_size)
        else:
            self.perm = db_index

        # the effective number of relations after adding symmetric adjacency matrices and/or self connections
        self.aug_num_rels = data.num_relations
        pos_g, pos_rel, la = self.__getitem__(0)
        self.n_feat_dim = pos_g[0].ndata['feat'].shape[1]

    def __len__(self):
        return self.sample_size

    def __getitem__(self, index):
        with self.main_env.begin(db=self.sub_db) as txn:
            str_id = '{:08}'.format(index).encode('ascii')
            pos_g, pos_la, pos_rel, neg_g, neg_la, neg_rel = pickle.loads(txn.get(str_id))

        if self.use_feature:
            pos_g = self.prepare_feature(pos_g)
            for i in range(len(neg_g)):
                neg_g[i] = self.prepare_feature(neg_g[i])
        neg_g.append(pos_g)
        neg_la.append(pos_la)
        neg_rel.append(pos_rel)
        return neg_g, neg_rel, len(neg_g)-1

    def prepare_feature(self, subgraph):
        labels = subgraph.ndata['feat']
        n_feats = self.wiki_data.entity_feat[subgraph.ndata[dgl.NID]]
        n_feats = np.concatenate((labels, n_feats), axis=1)
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)
        return subgraph


class SubgraphDatasetWikiEval(Dataset):
    """Extracted, labeled, subgraph dataset -- DGL Only"""

    def __init__(self, data, params, adj_path, sample_size=1000, neg_link_per_sample=1, use_feature=False):
        self.wiki_data = data
        self.num_rels = data.num_relations
        self.val_dict = data.valid_dict
        self.edges = data.train_hrt
        self.num_edges = len(self.val_dict['h,r->t']['hr'])
        self.num_nodes = data.num_entities
        self.use_feature = use_feature
        self.params = params
        self.graph = construct_graph_from_edges(self.edges.T, self.num_nodes)
        self.ssp_graph = [load_npz(os.path.join(adj_path, "adj_rel_" + str(i) + ".npz")) for i in
                          range(self.wiki_data.num_relations)]
        self.adj_mat = incidence_matrix_coo(self.ssp_graph)
        self.adj_mat += self.adj_mat.T
        self.max_n_label = [10, 10]
        self.neg_sample = neg_link_per_sample

        self.sample_size = sample_size

        # the effective number of relations after adding symmetric adjacency matrices and/or self connections
        self.aug_num_rels = data.num_relations
        self.__getitem__(0)

    def __len__(self):
        return self.sample_size

    def __getitem__(self, index):
        base_nodes = set(
            self.val_dict['h,r->t']['t_candidate'][index].tolist() + [self.val_dict['h,r->t']['hr'][index, 0]])
        sample_nodes = get_neighbor_nodes(base_nodes, self.adj_mat, max_nodes_per_hop=2000)
        sample_nodes = list(base_nodes) + list(sample_nodes)
        pos_subgraph = self.graph.subgraph(sample_nodes)
        pos_subgraph.edata['type'] = self.graph.edata['type'][pos_subgraph.edata[dgl.EID]]
        return pos_subgraph


class SubgraphDatasetWikiLocalEval(Dataset):
    """Extracted, labeled, subgraph dataset -- DGL Only"""

    def __init__(self, data, params, db_path, sub_db_name, sample_size=1000, neg_link_per_sample=1,
                 use_feature=False):
        self.wiki_data = data
        self.use_feature = use_feature
        self.params = params
        self.max_n_label = [10, 10]
        self.val_dict = data.valid_dict
        self.num_edges = len(self.val_dict['h,r->t']['hr'])
        self.neg_sample = neg_link_per_sample
        self.main_env = lmdb.open(db_path, readonly=True, max_dbs=3, lock=False)
        self.sample_size = sample_size
        self.sub_db = self.main_env.open_db(sub_db_name.encode())

        # the effective number of relations after adding symmetric adjacency matrices and/or self connections
        self.aug_num_rels = data.num_relations
        graphs = self.__getitem__(0)
        self.n_feat_dim = graphs[0][0].ndata['feat'].shape[1]

    def __len__(self):
        return self.sample_size

    def __getitem__(self, index):
        index = len(self)-index-1
        p_head = self.val_dict['h,r->t']['hr'][index, 0]
        rel = self.val_dict['h,r->t']['hr'][index, 1]
        with self.main_env.begin(db=self.sub_db) as txn:
            str_id = '{:08}'.format(index).encode('ascii')
            g = pickle.loads(txn.get(str_id))

        p_id = g.ndata[dgl.NID].numpy()
        adj_mat = g.adjacency_matrix() + g.adjacency_matrix(True)
        adj_mat = adj_mat.to_dense().numpy()
        node_to_id = {pid: i for i, pid in enumerate(p_id)}
        p_candidates = self.val_dict['h,r->t']['t_candidate'][index]
        candidates = [node_to_id[i] for i in p_candidates]
        head = node_to_id[p_head]
        graphs = []
        for candidate in candidates:
            pos_nodes, pos_label, _, _, _ = subgraph_extraction_labeling_wiki([head, candidate], rel,
                                                                              adj_mat, max_nodes_per_hop=100)
            pos_subgraph = g.subgraph(pos_nodes)
            pos_subgraph.edata['type'] = g.edata['type'][pos_subgraph.edata[dgl.EID]]
            pos_subgraph.edata['label'] = torch.tensor(rel * np.ones(pos_subgraph.edata['type'].shape),
                                                       dtype=torch.long)
            pos_subgraph.add_edges([0], [1])
            pos_subgraph.edata['type'][-1] = torch.tensor(rel, dtype=torch.int32)
            pos_subgraph.edata['label'][-1] = torch.tensor(rel, dtype=torch.int32)
            pos_subgraph = self._prepare_features_new(pos_subgraph, pos_label,
                                                      self.wiki_data.entity_feat[
                                                          p_id[pos_nodes]] if self.use_feature else None)
            graphs.append(pos_subgraph)
        return graphs, [rel]*len(graphs), self.val_dict['h,r->t']['t_correct_index'][index]

    def _prepare_features_new(self, subgraph, n_labels, n_feats=None):
        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        # label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        # label_feats[np.arange(n_nodes), 0] = 1
        # label_feats[np.arange(n_nodes), self.max_n_label[0] + 1] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)

        head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
        tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
        n_ids = np.zeros(n_nodes)
        n_ids[head_id] = 1  # head
        n_ids[tail_id] = 2  # tail
        subgraph.ndata['id'] = torch.FloatTensor(n_ids)

        self.n_feat_dim = n_feats.shape[1]  # Find cleaner way to do this -- i.e. set the n_feat_dim
        return subgraph


class SubgraphDatasetWikiOnline(Dataset):
    """Extracted, labeled, subgraph dataset -- DGL Only"""

    def __init__(self, data, params, adj_path, sample_size=1000, neg_link_per_sample=1, use_feature=False,
                 shuffle=True):
        self.wiki_data = data
        self.num_rels = data.num_relations
        self.edges = data.train_hrt
        self.num_edges = len(self.edges)
        self.num_nodes = data.num_entities
        self.use_feature = use_feature
        self.params = params
        self.graph = construct_graph_from_edges(self.edges.T, self.num_nodes)
        self.ssp_graph = [load_npz(os.path.join(adj_path, "adj_rel_" + str(i) + ".npz")) for i in
                          range(self.wiki_data.num_relations)]
        self.adj_mat = incidence_matrix_coo(self.ssp_graph)
        self.adj_mat += self.adj_mat.T
        self.max_n_label = [10, 10]
        self.neg_sample = neg_link_per_sample

        self.sample_size = sample_size
        self.shuffle = shuffle
        self.perm = np.random.permutation(self.num_edges)
        self.shuffle_count = 0

        # the effective number of relations after adding symmetric adjacency matrices and/or self connections
        self.aug_num_rels = data.num_relations
        pos_g, pos_la, pos_rel, neg_g, neg_la, neg_rel = self.__getitem__(0)
        self.n_feat_dim = pos_g.ndata['feat'].shape[1]

    def __len__(self):
        return self.sample_size

    def __getitem__(self, index):
        pos_link = self.edges[self.perm[index]]
        neg_links = []
        st = time.time()
        for i in range(self.neg_sample):
            neg_links.append(sample_neg_one(self.ssp_graph, pos_link, self.num_nodes))
        nodes = [link[0] for link in neg_links] + [link[2] for link in neg_links] + [pos_link[0], pos_link[2]]
        node_set = set(nodes)
        sample_nodes = get_neighbor_nodes(node_set, self.adj_mat, max_nodes_per_hop=self.neg_sample*5)
        sample_nodes = list(node_set) + list(sample_nodes)
        main_subgraph = self.graph.subgraph(sample_nodes)
        main_subgraph.edata['type'] = self.graph.edata['type'][main_subgraph.edata[dgl.EID]]
        
        p_id = main_subgraph.ndata[dgl.NID].numpy()
        local_adj_mat = main_subgraph.adjacency_matrix() + main_subgraph.adjacency_matrix(True)
        local_adj_mat = local_adj_mat.to_dense().numpy()
        node_to_id = {pid: i for i, pid in enumerate(p_id)}

        pos_nodes, pos_label, _, _, _ = subgraph_extraction_labeling_wiki([node_to_id[pos_link[0]], node_to_id[pos_link[2]]], pos_link[1], local_adj_mat, max_nodes_per_hop=self.neg_sample*5)
        pos_subgraph = main_subgraph.subgraph(pos_nodes)
        pos_subgraph.edata['type'] = main_subgraph.edata['type'][pos_subgraph.edata[dgl.EID]]
        pos_subgraph.edata['label'] = torch.tensor(pos_link[1] * np.ones(pos_subgraph.edata['type'].shape),
                                                   dtype=torch.long)
        # map the id read by GraIL to the entity IDs as registered by the KGE embeddings
        pos_subgraph = self._prepare_features_new(pos_subgraph, pos_label,
                                                  self.wiki_data.entity_feat[p_id[pos_nodes]] if self.use_feature else None)
        neg_subgraphs = []
        for i in range(self.neg_sample):
            neg_nodes, neg_label, _, _, _ = subgraph_extraction_labeling_wiki([node_to_id[neg_links[i][0]], node_to_id[neg_links[i][2]]], neg_links[i][1], local_adj_mat, max_nodes_per_hop=self.neg_sample*5)

            neg_subgraph = main_subgraph.subgraph(neg_nodes)
            neg_subgraph.edata['label'] = torch.tensor(neg_links[i][1] * np.ones(neg_subgraph.edata['type'].shape),
                                                       dtype=torch.long)
            neg_subgraph.add_edges([0], [1])
            neg_subgraph.edata['type'][-1] = torch.tensor(neg_links[i][1], dtype=torch.int32)
            neg_subgraph.edata['label'][-1] = torch.tensor(neg_links[i][1], dtype=torch.int32)
            neg_subgraphs.append(self._prepare_features_new(neg_subgraph, neg_label,
                                                            self.wiki_data.entity_feat[
                                                                p_id[neg_nodes]] if self.use_feature else None))
        if self.shuffle_count >= self.sample_size and self.shuffle:
            j = self.perm[0]
            self.perm = np.random.permutation(self.num_edges)
            if self.perm[0] == j:
                print("wrong perm")
            self.shuffle_count = 0
        return pos_subgraph, 1, pos_link[1], neg_subgraphs, [0] * len(neg_subgraphs), [neg_links[i][1] for i in
                                                                                       range(len(neg_subgraphs))]

    def _prepare_features_new(self, subgraph, n_labels, n_feats=None):
        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        # label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        # label_feats[np.arange(n_nodes), 0] = 1
        # label_feats[np.arange(n_nodes), self.max_n_label[0] + 1] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)

        head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
        tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
        n_ids = np.zeros(n_nodes)
        n_ids[head_id] = 1  # head
        n_ids[tail_id] = 2  # tail
        subgraph.ndata['id'] = torch.FloatTensor(n_ids)

        self.n_feat_dim = n_feats.shape[1]  # Find cleaner way to do this -- i.e. set the n_feat_dim
        return subgraph

