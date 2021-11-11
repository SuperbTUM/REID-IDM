import torch
import torch.nn as nn
from torchvision import models
from sklearn.cluster import DBSCAN
import numpy as np

pretrained = False
resnet50 = models.resnet50(pretrained=pretrained)


class Stage0_resnet50(nn.Module):
    def __init__(self):
        super(Stage0_resnet50, self).__init__()
        conv = resnet50.conv1
        bn = resnet50.bn1
        relu = resnet50.relu
        pool = resnet50.maxpool
        self.stage0 = nn.Sequential(*[conv, bn, relu, pool])

    def forward(self, x):
        return self.stage0(x)


class Stage1_resnet50(nn.Module):
    def __init__(self):
        super(Stage1_resnet50, self).__init__()
        self.stage1 = resnet50.layer1

    def forward(self, x):
        return self.stage1(x)


class Stage2_resnet50(nn.Module):
    def __init__(self):
        super(Stage2_resnet50, self).__init__()
        self.stage2 = resnet50.layer2

    def forward(self, x):
        return self.stage2(x)


class Stage3_resnet50(nn.Module):
    def __init__(self):
        super(Stage3_resnet50, self).__init__()
        self.stage3 = resnet50.layer3

    def forward(self, x):
        return self.stage3(x)


class Stage4_resnet50(nn.Module):
    def __init__(self):
        super(Stage4_resnet50, self).__init__()
        self.stage4 = resnet50.layer4

    def forward(self, x):
        return self.stage4(x)


class GAP(nn.Module):
    def __init__(self):
        super(GAP, self).__init__()
        self.gap = resnet50.avgpool

    def forward(self, x):
        return self.gap(x)


def clustering(x, eps=0.6, num_samples=4):
    """
    calculate the clustering number of target domain: ct
    :param eps: critical parameters of DBSCAN
    :param num_samples: same as above
    :param x: input from target domain
    :return: clustering number
    """
    return len(np.unique(DBSCAN(eps=eps, num_samples=num_samples, n_jobs=-1).fit_predict(x)))


class HybridClassifier(nn.Module):
    def __init__(self, cs, ct):
        """
        This is a hybrid classifier after the global average pooling layer
        :param cs: the number of identities in the source domain
        :param ct: cluster number of target domain
        """
        super(HybridClassifier, self).__init__()
        input_channels = resnet50.fc.in_features
        self.bn = nn.BatchNorm1d(input_channels)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(input_channels, cs + ct, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)

    def forward(self, x):
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x


class IDM_MODULE(nn.Module):
    def __init__(self, input_channels=64):
        """
        input_channels consists of source domain or target domain
        :param input_channels:
        """
        super(IDM_MODULE, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avepool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(2 * input_channels, input_channels)
        self.fc2 = nn.Linear(input_channels, input_channels // 2)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(input_channels // 2, 2)  # why there is no non-linear activation between linear layers
        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal(m.weight, std=0.001)

    def forward(self, x):
        batch_size = x.shape[0]
        Gs, Gt = x[:batch_size//2], x[batch_size//2:]
        Gs_maxpool = self.maxpool(Gs).squeeze()
        Gs_avepool = self.avepool(Gs).squeeze()
        Gs_concat = torch.cat([Gs_maxpool, Gs_avepool], dim=-1)  # concatenate via channel
        Gs_fc1 = self.fc1(Gs_concat)
        Gt_maxpool = self.maxpool(Gt).squeeze()
        Gt_avepool = self.avepool(Gt).squeeze()
        Gt_concat = torch.cat([Gt_maxpool, Gt_avepool], dim=-1)
        Gt_fc1 = self.fc1(Gt_concat)
        summation = Gs_fc1 + Gt_fc1
        mlp = self.fc2(summation)
        mlp = self.relu(mlp)
        a = self.fc3(mlp)
        a = self.softmax(a)
        a = a.view(-1, 1, 1, 1)
        a_s, a_t = a[0], a[1]
        G_inter = a_s * Gs + a_t * Gt
        G = torch.cat([Gs, G_inter, Gt], dim=0)
        return G, a


class Model(nn.Module):
    def __init__(self, cs, ct, insert=0):
        super(Model, self).__init__()
        self.stage0 = Stage0_resnet50()
        if insert == 0:
            self.idm = IDM_MODULE(input_channels=64)
        self.stage1 = Stage1_resnet50()
        if insert == 1:
            self.idm = IDM_MODULE(input_channels=256)
        self.stage2 = Stage2_resnet50()
        if insert == 2:
            self.idm = IDM_MODULE(input_channels=512)
        self.stage3 = Stage3_resnet50()
        if insert == 3:
            self.idm = IDM_MODULE(input_channels=1024)
        self.stage4 = Stage4_resnet50()
        if insert == 4:
            self.idm = IDM_MODULE(input_channels=2048)
        self.avepool = GAP()
        self.classifier = HybridClassifier(cs=cs, ct=ct)

    def forward(self, x, insert=0, train=True):
        """
        :param x: the concatenation of batch-level inputs from source domain and target domain
        :param insert: the place where IDM module will be inserted
        :param train: if the model is in the training stage or not
        :return: prob -> phai(x); x -> f(x); a -> ...
        """
        a = None
        x = self.stage0(x)
        if insert == 0 and train:
            x, a = self.idm(x)
        x = self.stage1(x)
        if insert == 1 and train:
            x, a = self.idm(x)
        x = self.stage2(x)
        if insert == 2 and train:
            x, a = self.idm(x)
        x = self.stage3(x)
        if insert == 3 and train:
            x, a = self.idm(x)
        x = self.stage4(x)
        if insert == 4 and train:
            x, a = self.idm(x)
        x_ = self.avepool(x)
        prob = self.classifier(x_)
        return prob, x, a
