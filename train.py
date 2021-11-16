from torch.utils.data import Dataset, DataLoader
from model import Model, clustering, FeatureExtractor
import numpy as np
import pandas as pd
import cv2
import glob
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
from losses import *
import concurrent.futures
from torchvision import transforms


def fetch_data(path_image, path_label=None):
    label = None
    if path_label:
        label = pd.read_csv(path_label)  # read the label
        label = np.asarray(label).astype(np.float32)
    image_path = sorted(glob.glob(path_image + "/*.png"))

    def read_image(image_path):
        images = cv2.imread(image_path)
        return images

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executer:
        images = executer.map(read_image, image_path)
    return images, label


def load_model_data(path_image_source,
                    path_label_source,
                    label_target_train=None,
                    stage=0,
                    learning_rate=3.5e-4,
                    cuda=False):
    source_images, source_labels = fetch_data(path_image_source, path_label_source)
    cs = len(np.unique(source_labels))
    if label_target_train:
        ct = len(label_target_train)
    else:
        ct = 0
    if cuda:
        model = Model(cs, ct, insert=stage).cuda()
    else:
        model = Model(cs, ct)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    model_content = (model, optimizer, lr_scheduler)
    source_data = (source_images, source_labels)
    num_classes = cs + ct
    return model_content, source_data, num_classes


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class MyDataset(Dataset):
    def __init__(self, images, labels, train=True, transform=None):
        super(MyDataset, self).__init__()
        self.images = images
        self.labels = labels
        self.train = train
        self.transform = transform

    def __len__(self):
        if isinstance(self.images, list):
            return len(self.images)
        else:
            return self.images.shape[0]

    def __getitem__(self, item):
        image = self.images[item]
        if self.transform:
            image = self.transform(image)
        if self.train and self.labels:
            label = self.labels[item]
            return image, torch.tensor(label).int()
        else:
            label = None
            return image, label


def collate(batch):
    images = []
    labels = []
    for sample in batch:
        image, label = sample
        images.append(image)
        if label:
            labels.append(label)
    images = torch.Tensor(images).float()
    if labels:
        labels = torch.Tensor(labels).int()
        batch = images, labels
    else:
        batch = images, None
    return batch


class IterLoader:
    def __init__(self, loader, length=None):
        self.loader = loader
        self.length = length
        self.iter = None

    def __len__(self):
        if self.length:
            return self.length
        return len(self.loader)

    def new_epoch(self):
        self.iter = iter(self.loader)

    def next(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)


def train(path_image_source,
          path_label_source,
          path_image_target,
          path_label_target=None,
          train=True,
          max_epoch=50,
          batch_size=128,
          learning_rate=3.5e-4,
          cuda=False,
          stage=0,
          resized=(200, 100)):
    model_content, source_content, num_class = load_model_data(path_image_source,
                                                               path_label_source,
                                                               stage=stage,
                                                               learning_rate=learning_rate,
                                                               cuda=cuda)
    model, optimizer, lr_scheduler = model_content
    images_source, labels_source = source_content
    images_target, labels_target = fetch_data(path_image_target, path_label_target)
    transform = transforms.Compose([
        transforms.Resize(resized, interpolation=3),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Pad(10),
        transforms.Resize(resized),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5)
    ])
    source_dataset = MyDataset(images_source, labels_source, train, transform)
    target_dataset = MyDataset(images_target, labels_target, train, transform)

    for epoch in range(max_epoch):
        train_loader_source = IterLoader(DataLoaderX(source_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                                     pin_memory=True, collate_fn=collate))
        train_loader_target = IterLoader(DataLoaderX(target_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                                     pin_memory=True, collate_fn=collate))
        features, _ = FeatureExtractor(0, train_loader_target)
        labels_target = clustering(features)  # use this to construct a new dataset
        train_iters = len(target_dataset) // batch_size
        for _ in range(train_iters):
            optimizer.zero_grad()
            train_loader_source.new_epoch()
            train_loader_target.new_epoch()
            image_source, label_source = train_loader_source.next()
            image_target, _ = train_loader_target.next()

            C, H, W = image_source.shape
            inputs = torch.cat([image_source, image_target], dim=1).view(-1, C, H, W)
            if cuda:
                inputs = inputs.cuda()
            model.train()
            prob, features, a = model(inputs, stage=stage, train=train)
            size = features.shape[0]
            feat_s, feat_inter, feat_t = features[:size // 3], features[size // 3:2 * size // 3], features[
                                                                                                  2 * size // 3:]
            feat_ori = torch.cat([feat_s, feat_t], dim=0)
            targets = torch.repeat_interleave(label_source, 2).view(-1)
            triplet_loss = TripletLoss(margin=1.)(feat_ori, targets)
            div_loss = DivLoss()(a.detach())
            brgfeat_loss = BridgeFeatLoss()(feat_s, feat_inter, feat_t, a)
            classification_loss, brgprob_loss = BridgeProbLoss(num_class)(targets, prob, a.detach())
            loss = IDMLoss(classification_loss, triplet_loss, div_loss, brgfeat_loss, brgprob_loss)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
    torch.save(model.state_dict(), "model.pth")
