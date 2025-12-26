import torch
from torch import nn
import torch.nn.functional as F


class GeM(nn.Module):
    def __init__(self, p: int = 3, trainable: bool = False, eps: float = 1e-6):
        super(GeM, self).__init__()
        if trainable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.p = p

        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p)
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )
