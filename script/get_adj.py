from ogb.lsc import WikiKG90MDataset as wikiData
from utils.data_utils import process_files_wiki
import time
import os.path as osp

path = "/project/tantra/jerry.kong/ogb_project/dataset"

dataset = wikiData(path)
start = time.time()
process_files_wiki(dataset, osp.join(path, "wikikg90m_kddcup2021"))
print(time.time()-start)

