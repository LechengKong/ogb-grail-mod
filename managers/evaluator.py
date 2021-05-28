import os
import numpy as np
import torch
import pdb
from sklearn import metrics
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics


class Evaluator():
    def __init__(self, params, graph_classifier, data):
        self.params = params
        self.graph_classifier = graph_classifier
        self.data = data

    def eval(self, save=False):
        pos_scores = []
        pos_labels = []
        neg_scores = []
        neg_labels = []
        mrr_scores = []
        dataloader = DataLoader(self.data, batch_size=self.params.val_batch_size, shuffle=False, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn_val, prefetch_factor=self.params.prefetch_val)

        self.graph_classifier.eval()
        with torch.no_grad():
            pbar = tqdm(dataloader)
            for batch in pbar:

                data_pos, targets_pos = self.params.move_batch_to_device_val(batch, self.params.device)
                # print([self.data.id2relation[r.item()] for r in data_pos[1]])
                # pdb.set_trace()
                score_pos = self.graph_classifier(data_pos)
                scores = score_pos.view(len(targets_pos), -1)
                scores = scores.cpu().numpy()

                tp = targets_pos.cpu().numpy()

                true_labels = np.zeros(scores.shape)
                true_labels[np.arange(len(targets_pos)), tp] = 1

                mrr_scores.append(metrics.label_ranking_average_precision_score(true_labels, scores))

        # acc = metrics.accuracy_score(labels, preds)
        mrr = 0
        for v in mrr_scores:
            mrr += v
        mrr /= len(mrr_scores)

        if save:
            pos_test_triplets_path = os.path.join(self.params.main_dir, 'data/{}/{}.txt'.format(self.params.dataset, self.data.file_name))
            with open(pos_test_triplets_path) as f:
                pos_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            pos_file_path = os.path.join(self.params.main_dir, 'data/{}/grail_{}_predictions.txt'.format(self.params.dataset, self.data.file_name))
            with open(pos_file_path, "w") as f:
                for ([s, r, o], score) in zip(pos_triplets, pos_scores):
                    f.write('\t'.join([s, r, o, str(score)]) + '\n')

            neg_test_triplets_path = os.path.join(self.params.main_dir, 'data/{}/neg_{}_0.txt'.format(self.params.dataset, self.data.file_name))
            with open(neg_test_triplets_path) as f:
                neg_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            neg_file_path = os.path.join(self.params.main_dir, 'data/{}/grail_neg_{}_{}_predictions.txt'.format(self.params.dataset, self.data.file_name, self.params.constrained_neg_prob))
            with open(neg_file_path, "w") as f:
                for ([s, r, o], score) in zip(neg_triplets, neg_scores):
                    f.write('\t'.join([s, r, o, str(score)]) + '\n')
        return {'mrr': mrr}
