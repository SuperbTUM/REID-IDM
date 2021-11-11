import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


warnings.filterwarnings("ignore")


def L2_distance(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


class DivLoss(nn.Module):
    def __init__(self):
        super(DivLoss, self).__init__()

    def forward(self, a):
        a_s, a_t = a[0].squeeze(), a[1].squeeze()
        return -(a_s.std() + a_t.std())


class BridgeFeatLoss(nn.Module):
    def __init__(self):
        super(BridgeFeatLoss, self).__init__()

    def forward(self, Fs, F_inter, Ft, a):
        dist_sinter = L2_distance(Fs, F_inter)
        dist_tinter = L2_distance(Ft, F_inter)
        loss = a[0] * dist_sinter + a[1] * dist_tinter
        return loss.mean()


class BridgeProbLoss(nn.Module):
    def __init__(self, num_class, epsilon=0.1):
        super(BridgeProbLoss, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.num_classes = num_class
        self.epsilon = epsilon

    def forward(self, y, P, a):
        batch_size = P.shape[0]
        Ps, P_inter, Pt = P[:batch_size // 3], P[batch_size // 3: 2*batch_size//3], P[2*batch_size//3:]
        inputs_ori = torch.cat([Ps, Pt], dim=1).view(-1, P.shape[-1])
        logprob_ori = self.logsoftmax(inputs_ori)
        logprob_inter = self.logsoftmax(P_inter)

        y = y.view(-1, y.size(-1))
        soft_targets = (1 - self.epsilon) * y + self.epsilon / self.num_classes
        y_s, y_t = y[:y.shape[0]//2], y[y.shape[0]//2:]

        a = a.view(-1, 1)
        soft_targets_mixed = a * y_s + (1. - a) * y_t
        soft_targets_mixed = (1 - self.epsilon) * soft_targets_mixed + self.epsilon / self.num_classes
        loss_ori = -(soft_targets * logprob_ori).mean(0).sum()
        loss_bridge_prob = -(soft_targets_mixed * logprob_inter).mean(0).sum()

        return loss_ori, loss_bridge_prob


class ClassificationLoss(nn.Module):
    def __init__(self, num_class, epsilon=0.1):
        super(ClassificationLoss, self).__init__()
        self.num_classes = num_class
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        probs_inputs = self.logsoftmax(inputs)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        return -(probs_inputs * targets).mean()


def _batch_hard(mat_distance, mat_similarity, indice=False):
    sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1, descending=True)
    hard_p = sorted_mat_distance[:, 0]
    hard_p_indice = positive_indices[:, 0]
    sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
    hard_n = sorted_mat_distance[:, 0]
    hard_n_indice = negative_indices[:, 0]
    if indice:
        return hard_p, hard_n, hard_p_indice, hard_n_indice
    return hard_p, hard_n


class TripletLoss(nn.Module):
    def __init__(self, margin=None, normalize=False):
        super(TripletLoss, self).__init__()
        self.normalize = normalize
        if margin is None:
            self.triplet = nn.SoftMarginLoss()
        else:
            self.triplet = nn.TripletMarginLoss(margin=margin)

    def forward(self, x, label):
        if self.normalize:
            x = F.normalize(x)
        mat_distance = L2_distance(x, x)
        N = mat_distance.size(0)
        mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()
        hard_p, hard_n = _batch_hard(mat_distance=mat_distance, mat_similarity=mat_sim)
        return self.triplet(x, hard_p, hard_n)


def IDMLoss(classi, tpl, div, brg_feat, brg_prob):
    reid = 0.3 * classi + tpl
    return reid + 0.7 * brg_prob + 0.1 * brg_feat + div
