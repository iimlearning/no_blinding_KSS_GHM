import torch
from torch import nn
import torch.nn.functional as F

class GHM_Loss(nn.Module):
    """GHM Classification Loss.
    Details of the theorem can be viewed in the paper
    "Gradient Harmonized Single-stage Detector".
    https://arxiv.org/abs/1811.05181
    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average. 【移动平均量】
        use_sigmoid (bool): Can only be true for BCE based loss now.
        loss_weight (float): The weight of the total GHM-C loss."""

    def __init__(self, bins, alpha):
        super(GHM_Loss, self).__init__()
        self._bins = bins
        self._alpha = alpha
        self._last_bin_count = None

    def _g2bin(self, g):
        # g = (g-torch.min(g))/(torch.max(g)-torch.min(g))
        return torch.floor(g * (self._bins - 0.0001)).long()

    def _custom_loss(self, x, target, weight):
        raise NotImplementedError

    def _custom_loss_grad(self, x, target):
        raise NotImplementedError

    def forward(self, x, target):
        g = torch.abs(self._custom_loss_grad(x, target)).detach() # 每个样本 计算出的 梯度g
#         print("g:",g)
        bin_idx = self._g2bin(g) # 每个样本 属于的区域
#         print("bin_idx:",bin_idx)
        # 求得 每个区域内样本的数量
        bin_count = torch.zeros((self._bins))
        for i in range(self._bins):
            bin_count[i] = (bin_idx == i).sum().item()
#         print("bin_count:",bin_count)
        N = (x.size(0) * x.size(1)) # 一共有N的样本
        
        # EMA
        if self._last_bin_count is None:
            self._last_bin_count = bin_count
        else:
            bin_count = self._alpha * self._last_bin_count + (1 - self._alpha) * bin_count
            self._last_bin_count = bin_count
        
        nonempty_bins = (bin_count > 0).sum().item() # M
#         print("nonempty_bins:", nonempty_bins)
        gd = bin_count * nonempty_bins # GD
#         print("gd:",gd)
        gd = torch.clamp(gd, min=0.0001) # 对小于0.0001的值 放大至0.0001,避免除数为0
#         print("gd:",gd)
        beta = N / gd # 每个
#         print("beta:", beta)
        beta = beta.to(torch.float32)
        
#         print(beta[bin_idx])
        return self._custom_loss(x, target, beta[bin_idx]) # 求修正的损失


class GHMC_Loss(GHM_Loss):
    def __init__(self, bins, alpha):
        super(GHMC_Loss, self).__init__(bins, alpha)

    def _custom_loss(self, x, target, weight):
        weight = weight.cuda()
        return torch.sum(F.cross_entropy(x, target, reduction='none').mul(weight.detach()))/torch.sum(weight.detach())

    def _custom_loss_grad(self, x, target):
        
#         思考怎么求得每个样本的梯度
        # 先从原先的计算图中抛离出去，这个梯度不作为参数的更新，只是为了GHM使用而已。
        
        #x_ = x.cpu().detach()
        #x_.requires_grad = True

        #target_ = target.cpu().detach()

       # loss = F.cross_entropy(x_, target_)
        #loss.backward()

        # # 对每个x只获得正确的那个梯度
        #target = target.view(-1, 1)
        #idx = target.cpu().long()

        #one_hot_key = torch.FloatTensor(target.size(0), x_.size(1)).zero_()
        #one_hot_key = one_hot_key.scatter_(1, idx, 1)

        #x_grad = (one_hot_key * x_.grad).sum(1).cuda()
        p = F.softmax(x.detach(), dim=1)
        p_right = p[range(target.detach().size(0)), target.detach()]
        grad = p_right - 1
        return grad # 损失对整体的梯度
