from ogb.lsc import WikiKG90MDataset as wikiData
from data_utils import process_files_wiki
import time
import os.path as osp
import numpy as np

from graph_sampler import sample_neg_wiki

path = "/project/tantra/jerry.kong/ogb_project/dataset/wikikg90m_kddcup2021"
read_path = osp.join(path, "adj_mat")
dataset = wikiData()
adj_list = []

for i in range(wikiData.num_relations):
    adj_list.append(np.load(osp.join(read_path, str(i)), mmap_mode='r'))

pos, neg = sample_neg_wiki(adj_list, dataset.train_hrt)

print(pos.shape, neg.shape)
