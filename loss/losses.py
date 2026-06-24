import torch
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.utils import common_functions as c_f


def bit_var_loss():
    def F(x):
        return 1 / (1+torch.exp(-x))
    def loss(z):
        return torch.mean(F(z) * (1-F(z)))
    return loss

class LowerBoundLoss(torch.nn.Module):
    def __init__(self):
        super(LowerBoundLoss, self).__init__()

    def forward(self, output):
        max_loss = torch.clamp(output, min=0, max=None)
        mean_max_loss = torch.mean(max_loss)
        return mean_max_loss


