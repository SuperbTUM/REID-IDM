import torch
import torch.nn as nn
from model import Model
import torch.nn.functional as F


def load_trained_model(path, cs, ct):
    state_dict = torch.load(path)
    model = Model(cs, ct)
    model.load_state_dict(state_dict)
    return model


class TestModel(nn.Module):
    def __init__(self, path, cs, ct):
        super(TestModel, self).__init__()
        model = load_trained_model(path, cs, ct)
        self.layer0 = model.stage0
        self.layer1 = model.stage1
        self.layer2 = model.stage2
        self.layer3 = model.stage3
        self.layer4 = model.stage4
        for _, parameters in self.named_parameters():
            parameters.requires_grad = False
        self.batchnorm = nn.BatchNorm1d(2048)
        self.batchnorm.bias.requires_grad_(False)
        nn.init.constant_(self.batchnorm.weight, 1.)
        nn.init.constant_(self.batchnorm.bias, 0.)
        self.test_classifier = model.classifier

    def forward(self, x, output_prob=False):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(-1, 1)
        x = self.batchnorm(x)
        norm_x = F.normalize(x, p=2, dim=-1)
        if output_prob is False:
            return norm_x
        prob = self.test_classifier(x)
        return prob, x, norm_x
