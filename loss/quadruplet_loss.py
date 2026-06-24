import argparse

import torch
import torch.nn.functional as F
from torch import nn

from BaseLine.utils import distance
from loss.miner import QuadrupletMarginMiner, get_all_quadruplets


class QuadrupletMarginLoss(nn.Module):
    def __init__(self, args: argparse.Namespace, **kwargs):
        super().__init__()
        self.args = args
        self.type_of_distance = args.type_of_distance
        self.margin = args.margin
        self.need_cnt = kwargs.pop("need_cnt", False)
        self.miner = (
            None
            if args.type_of_quadruplets == "no-miner"
            else QuadrupletMarginMiner(args.margin, args.type_of_distance, args.type_of_quadruplets, args.what_is_hard)
        )

    def forward(self, logits, labels):
        if self.miner is None:
            quadruplets = get_all_quadruplets(labels)
        else:
            quadruplets = self.miner(logits, labels)

        if quadruplets is None:
            print("no quadruplets")
            return (0, 0) if self.need_cnt else 0

        mat = distance(logits, self.type_of_distance)
        I, J, K, N = quadruplets[:, 0], quadruplets[:, 1], quadruplets[:, 2], quadruplets[:, 3]
        ij_dists = mat[I, J]
        ik_dists = mat[I, K]
        in_dists = mat[I, N]
        violation1 = ij_dists - in_dists + self.margin
        violation2 = ik_dists - in_dists + self.margin
        losses = F.relu(violation1) + F.relu(violation2)
        loss = torch.mean(losses)

        return (loss, quadruplets.size(0)) if self.need_cnt else loss


if __name__ == "__main__":
    pass
