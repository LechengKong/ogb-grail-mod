from ogb.lsc import WikiKG90MDataset as wikiData
from data_utils import process_files_wiki
import time

dataset = wikiData()
start = time.time()
process_files_wiki(dataset, "./dataset/wikikg90m_kddcup2021")
print(time.time()-start)