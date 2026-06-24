import torch


class HashNetLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(HashNetLoss, self).__init__()
        self.U = torch.zeros(config["num_train"], bit).half().to(config["device"])
        # self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float().to(config["device"])
        self.bit = bit

    def forward(self, u, y, ind, config):

       # self.U[ind, :] = u[1].data.to(self.U.dtype)
        #x = torch.cat([u[0], u[1]], dim=1)
        self.U[ind, :] = u.data.to(self.U.dtype)

        self.Y[ind, :] = y.float()
        similarity = (y @ self.Y.t() > 0).float()
        dot_product = config["alpha"] * u.to(self.U.dtype) @ self.U.t()

        mask_positive = similarity.data > 0
        mask_negative = similarity.data <= 0

        exp_loss = (1 + (-dot_product.abs()).exp()).log() + dot_product.clamp(min=0) - similarity * dot_product

        # weight
        S1 = mask_positive.float().sum()
        S0 = mask_negative.float().sum()
        S = S0 + S1

        exp_loss[mask_positive] = exp_loss[mask_positive] * (S / S1)
        exp_loss[mask_negative] = exp_loss[mask_negative] * (S / S0)

        loss = exp_loss.sum() / S

        return loss
