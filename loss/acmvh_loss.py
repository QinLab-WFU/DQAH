
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def zero2eps(x):
    x[x == 0] = 1
    return x

# Refer
def normalize(affinity):
    col_sum = zero2eps(np.sum(affinity, axis=1)[:, np.newaxis])
    row_sum = zero2eps(np.sum(affinity, axis=0))
    out_affnty = affinity/col_sum # row data sum = 1
    in_affnty = np.transpose(affinity/row_sum) # col data sum = 1 then transpose
    return in_affnty, out_affnty

# Check in 2022-1-3
def affinity_tag_multi(tag1: np.ndarray, tag2: np.ndarray):
    '''
    Use label or plabel to create the graph.
    :param tag1:
    :param tag2:
    :return:
    '''
    aff = np.matmul(tag1, tag2.T)
    affinity_matrix = np.float32(aff)
    # affinity_matrix[affinity_matrix > 1] = 1
    affinity_matrix = 1 / (1 + np.exp(-affinity_matrix))
    affinity_matrix = 2 * affinity_matrix - 1
    in_aff, out_aff = normalize(affinity_matrix)

    return in_aff, out_aff, affinity_matrix


class Acmvh_out(nn.Module):
    
    def __init__(self, param_clf, param_sim):
        super(Acmvh_out, self).__init__()
        self.loss = torch.nn.MSELoss()
        self.param_clf = param_clf
        self.param_sim = param_sim

    def forward(self, image_hash, image_cls, label):


        _, aff_norm, aff_label = affinity_tag_multi(label.cpu().numpy(), label.cpu().numpy())

        aff_label = torch.Tensor(aff_label).to(image_hash.device)

        h = F.normalize(image_hash)

        clf_loss = self.loss(torch.sigmoid(image_cls), label)

        similarity_loss = self.loss(h.mm(h.t()), aff_label)

        loss = clf_loss * self.param_clf + similarity_loss * self.param_sim

        return loss

