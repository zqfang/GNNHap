import torch
import torch.nn.functional as F 

# this is for  imblanced dataset, inspired from computor vision
class FocalLoss(torch.nn.Module):
    """
        This is a Pytorch implementation of Focal Loss. 
        You can pass either probability or raw logits which is controlled by the logits parameter in the constructor. 
        It will not generate nans even when the probability is 0.

        paper: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss