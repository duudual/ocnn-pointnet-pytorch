import torch
import torch.nn as nn
import ocnn


class OCNNClassifier(nn.Module):

    def __init__(self, name='lenet', feature='ND', nempty=False,
                 stages=3, channel=4, nout=40, resblock_num=2, depth=5):
        super().__init__()
        self.name = name.lower()
        self.feature = feature
        self.nempty = nempty
        self.depth = depth

        if self.name == 'lenet':
            self.model = ocnn.models.LeNet(channel, nout, stages, nempty)
        elif self.name == 'resnet':
            self.model = ocnn.models.ResNet(channel, nout, resblock_num, stages, nempty)
        else:
            raise ValueError(f'Unknown model: {name}')

    def get_input_feature(self, octree):
        octree_feature = ocnn.modules.InputFeature(self.feature, self.nempty)
        data = octree_feature(octree)
        return data

    def loss_function(self, logit, label):
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logit, label.long())
        return loss

    def forward(self, batch):
        octree, label = batch['octree'].cuda(), batch['label'].cuda()
        data = self.get_input_feature(octree)
        logits = self.model(data, octree, octree.depth)
        loss = self.loss_function(logits, label)
        pred = torch.argmax(logits, dim=1)
        accu = pred.eq(label).float().mean()
        return loss, accu
