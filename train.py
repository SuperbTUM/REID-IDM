from torch.utils.data import Dataset, DataLoader
from model import Model, clustering
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
                    stage=0,
                    learning_rate=3.5e-4,
                    cuda=False):
    source_images, source_labels = fetch_data(path_image_source, path_label_source)
    cs = len(np.unique(source_labels))
    ct = clustering(source_images)
    if cuda:
        model = Model(cs, ct, insert=stage).cuda()
    else:
        model = Model(cs, ct)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    model_content = (model, optimizer, lr_scheduler)
    source_data = (source_images, source_labels)
    return model_content, source_data, cs + ct


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class MyDataset(Dataset):
    def __init__(self, images, labels, images_target, labels_target, train=True, transform=None):
        super(MyDataset, self).__init__()
        self.images_source = images
        self.labels_source = labels
        self.images_target = images_target
        self.labels_target = labels_target
        self.train = train
        self.transform = transform

    def __len__(self):
        if isinstance(self.images_source, list):
            return len(self.images_source)
        else:
            return self.images_source.shape[0]

    def __getitem__(self, item):
        image_source = self.images_source[item]
        label_source = self.labels_source[item]
        image_target = self.images_target[item]
        if self.transform:
            image_source = self.transform(image_source)
            image_target = self.transform(image_target)
        if self.train:
            label_target = self.labels_target[item]
            return image_source, torch.tensor(label_source).int(), image_target, torch.tensor(label_target).int()
        else:
            label_target = None
            return image_source, torch.tensor(label_source).int(), image_target, label_target


def collate(batch):
    image_sources = []
    label_sources = []
    image_targets = []
    label_targets = []
    for sample in batch:
        image_source, label_source, image_target, label_target = sample
        image_sources.append(image_source)
        label_sources.append(label_source)
        image_targets.append(image_target)
        if label_target:
            label_targets.append(label_target)
    image_sources = torch.Tensor(image_sources).float()
    label_sources = torch.Tensor(label_sources).int()
    image_targets = torch.Tensor(image_targets).float()
    if label_targets:
        label_targets = torch.Tensor(label_targets).int()
        batch = (image_sources, label_sources, image_targets, label_targets)
    else:
        batch = (image_sources, label_sources, image_targets)
    return batch


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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = MyDataset(images_source, labels_source, images_target, labels_target, train, transform)

    for epoch in range(max_epoch):
        train_loader = DataLoaderX(dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                   pin_memory=True, collate_fn=collate)
        for samples in tqdm(train_loader):
            optimizer.zero_grad()
            if len(samples) == 3:
                image_source, label_source, image_target = samples
            else:
                image_source, label_source, image_target, label_target = samples
            C, H, W = image_source.shape
            inputs = torch.cat([image_source, image_target], dim=1).view(-1, C, H, W)
            if cuda:
                inputs = inputs.cuda()
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
