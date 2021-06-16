```python
import os
import torch
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.manual_seed(777)  # reproducibility

# parameters
total_dog_classes = 133
learning_rate = 0.001
batch_size = 10
keep_prob = 0.7
data_dir="all_images/dogImages/"
TRAIN = 'train'
VAL = 'valid'
TEST = 'test'


# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = {TRAIN: transforms.Compose([
        transforms.RandomResizedCrop(224),    
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
                  VAL: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
                  TEST: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])}

image_datasets = {x: dsets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in [TRAIN, VAL, TEST]}



dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=10,
                                             shuffle=True, num_workers=0)
              for x in [TRAIN, VAL, TEST]}

dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, VAL, TEST]}
class_names = image_datasets[TRAIN].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("number of training images:", len(image_datasets[TRAIN]))
print("numbe of validation images:", len(image_datasets[VAL]))
print("number of testing images:", len(image_datasets[TEST]))
```

    number of training images: 6680
    numbe of validation images: 835
    number of testing images: 836



```python
import torch.nn as nn
import torch.nn.functional as F

# model
'''EfficientNet in PyTorch.
Paper: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks".
Reference: https://github.com/keras-team/keras-applications/blob/master/keras_applications/efficientnet.py
'''

import torch

def swish(x):
    return x * x.sigmoid()

def drop_connect(x, drop_ratio):
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x


class SE(nn.Module):
    '''Squeeze-and-Excitation block with Swish.'''

    def __init__(self, in_channels, se_channels):
        super(SE, self).__init__()
        self.se1 = nn.Conv2d(in_channels, se_channels,
                             kernel_size=1, bias=True)
        self.se2 = nn.Conv2d(se_channels, in_channels,
                             kernel_size=1, bias=True)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = swish(self.se1(out))
        out = self.se2(out).sigmoid()
        out = x * out
        return out


class Block(nn.Module):
    '''expansion + depthwise + pointwise + squeeze-excitation'''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 expand_ratio=1,
                 se_ratio=0.,
                 drop_rate=0.):
        super(Block, self).__init__()
        self.stride = stride
        self.drop_rate = drop_rate
        self.expand_ratio = expand_ratio

        # Expansion
        channels = expand_ratio * in_channels
        self.conv1 = nn.Conv2d(in_channels,
                               channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        # Depthwise conv
        self.conv2 = nn.Conv2d(channels,
                               channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=(1 if kernel_size == 3 else 2),
                               groups=channels,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        # SE layers
        se_channels = int(in_channels * se_ratio)
        self.se = SE(channels, se_channels)

        # Output
        self.conv3 = nn.Conv2d(channels,
                               out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Skip connection if in and out shapes are the same (MV-V2 style)
        self.has_skip = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        out = x if self.expand_ratio == 1 else swish(self.bn1(self.conv1(x)))
        out = swish(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))
        if self.has_skip:
            if self.training and self.drop_rate > 0:
                out = drop_connect(out, self.drop_rate)
            out = out + x
        return out


class EfficientNet(nn.Module):
    def __init__(self, cfg, num_classes=total_dog_classes):
        super(EfficientNet, self).__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv2d(3,
                               32,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_channels=32)
        self.linear = nn.Linear(cfg['out_channels'][-1], num_classes)

    def _make_layers(self, in_channels):
        layers = []
        cfg = [self.cfg[k] for k in ['expansion', 'out_channels', 'num_blocks', 'kernel_size',
                                     'stride']]
        b = 0
        blocks = sum(self.cfg['num_blocks'])
        for expansion, out_channels, num_blocks, kernel_size, stride in zip(*cfg):
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                drop_rate = self.cfg['drop_connect_rate'] * b / blocks
                layers.append(
                    Block(in_channels,
                          out_channels,
                          kernel_size,
                          stride,
                          expansion,
                          se_ratio=0.25,
                          drop_rate=drop_rate))
                in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = swish(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        dropout_rate = self.cfg['dropout_rate']
        if self.training and dropout_rate > 0:
            out = F.dropout(out, p=dropout_rate)
        out = self.linear(out)
        return out

cfg = {
    'num_blocks': [1, 2, 2, 3, 3, 4, 1],
    'expansion': [1, 6, 6, 6, 6, 6, 6],
    'out_channels': [16, 24, 40, 80, 112, 192, 320],
    'kernel_size': [3, 3, 5, 3, 5, 5, 3],
    'stride': [1, 2, 2, 2, 1, 2, 1],
    'dropout_rate': 0.2,
    'drop_connect_rate': 0.2,
}
    


model_scratch = EfficientNet(cfg)
print(model_scratch)

# move tensors to GPU if CUDA is available
use_cuda = torch.cuda.is_available()
if use_cuda:
    model_scratch.cuda()
else:
    print("cuda is not available.")
```

    EfficientNet(
      (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (layers): Sequential(
        (0): Block(
          (conv1): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (se): SE(
            (se1): Conv2d(32, 8, kernel_size=(1, 1), stride=(1, 1))
            (se2): Conv2d(8, 32, kernel_size=(1, 1), stride=(1, 1))
          )
          (conv3): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): Block(
          (conv1): Conv2d(16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)
          (bn2): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (se): SE(
            (se1): Conv2d(96, 4, kernel_size=(1, 1), stride=(1, 1))
            (se2): Conv2d(4, 96, kernel_size=(1, 1), stride=(1, 1))
          )
          (conv3): Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (2): Block(
          (conv1): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)
          (bn2): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (se): SE(
            (se1): Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))
            (se2): Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
          )
          (conv3): Conv2d(144, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (3): Block(
          (conv1): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(144, 144, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=144, bias=False)
          (bn2): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (se): SE(
            (se1): Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))
            (se2): Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
          )
          (conv3): Conv2d(144, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (4): Block(
          (conv1): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(240, 240, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=240, bias=False)
          (bn2): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (se): SE(
            (se1): Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))
            (se2): Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
          )
          (conv3): Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (5): Block(
          (conv1): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(240, 240, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=240, bias=False)
          (bn2): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (se): SE(
            (se1): Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))
            (se2): Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
          )
          (conv3): Conv2d(240, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (6): Block(
          (conv1): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)
          (bn2): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (se): SE(
            (se1): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
            (se2): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
          )
          (conv3): Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (7): Block(
          (conv1): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)
          (bn2): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (se): SE(
            (se1): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
            (se2): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
          )
          (conv3): Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (8): Block(
          (conv1): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(480, 480, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=480, bias=False)
          (bn2): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (se): SE(
            (se1): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
            (se2): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
          )
          (conv3): Conv2d(480, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (9): Block(
          (conv1): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=False)
          (bn2): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (se): SE(
            (se1): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
            (se2): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
          )
          (conv3): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (10): Block(
          (conv1): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=False)
          (bn2): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (se): SE(
            (se1): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
            (se2): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
          )
          (conv3): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (11): Block(
          (conv1): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(672, 672, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=672, bias=False)
          (bn2): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (se): SE(
            (se1): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
            (se2): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
          )
          (conv3): Conv2d(672, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (12): Block(
          (conv1): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)
          (bn2): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (se): SE(
            (se1): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
            (se2): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
          )
          (conv3): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (13): Block(
          (conv1): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)
          (bn2): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (se): SE(
            (se1): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
            (se2): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
          )
          (conv3): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (14): Block(
          (conv1): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)
          (bn2): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (se): SE(
            (se1): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
            (se2): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
          )
          (conv3): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (15): Block(
          (conv1): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(1152, 1152, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1152, bias=False)
          (bn2): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (se): SE(
            (se1): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
            (se2): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
          )
          (conv3): Conv2d(1152, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (linear): Linear(in_features=320, out_features=133, bias=True)
    )



```python
import torch.optim as optim
from torch.optim import lr_scheduler
### TODO: select loss function
criterion_scratch = nn.CrossEntropyLoss()

### TODO: select optimizer
optimizer_scratch = optim.SGD(model_scratch.parameters(), lr=learning_rate, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_scratch, step_size = 7, gamma=0.1)
```


```python
import matplotlib.pyplot as plt
from PIL import ImageFile
import numpy as np
import time
import copy
import asyncio

%matplotlib inline
ImageFile.LOAD_TRUNCATED_IMAGES = True

from tqdm import tqdm
import torch

batch_size = 10
train_data = image_datasets[TRAIN]
test_data = image_datasets[TEST]
# dataloaders
train_data_loader = dataloaders[TRAIN]
valid_data_loader = dataloaders[VAL]

# training function
def fit(model, dataloader, optimizer, criterion, train_data, device, use_amp):
    print('Training')
    if use_amp == 'yes':
        scaler = torch.cuda.amp.GradScaler() 
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    for i, (image,target) in tqdm(enumerate(dataloader), total=int(len(train_data)/batch_size)):
        image, target = image.to(device), target.to(device)
        optimizer.zero_grad()
        
        if use_amp == 'yes':
            with torch.cuda.amp.autocast():
                outputs = model(image)
                loss = criterion(outputs, target)
        
        elif use_amp == 'no':
            outputs = model(image)
            loss = criterion(outputs, target)
            
        train_running_loss += (1/(i+1))*(loss.item() - train_running_loss)
        _, preds = torch.max(outputs, 1)
        train_running_correct += (preds == target).sum().item()
        
        if use_amp == 'yes':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        elif use_amp == 'no':
            loss.backward()
            optimizer.step()
        
    train_loss = train_running_loss/dataset_sizes[TRAIN]
    train_accuracy = 100. * train_running_correct/dataset_sizes[TRAIN]    
    return train_loss, train_accuracy

# validation function
def validate(model, dataloader, optimizer, criterion, val_data, device, use_amp):
    print('Validating')
    if use_amp == True:
        scaler = torch.cuda.amp.GradScaler() 
        
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    with torch.no_grad():
        for i, (image,target) in tqdm(enumerate(dataloader), total=int(len(val_data)/batch_size)):
            image, target = image.to(device), target.to(device)
            
            if use_amp == 'yes':
                with torch.cuda.amp.autocast():
                    outputs = model(image)
                    loss = criterion(outputs, target)
        
            elif use_amp == 'no':
                outputs = model(image)
                loss = criterion(outputs, target)
            
            val_running_loss += (1/(i+1))*(loss.item() - val_running_loss)
            _, preds = torch.max(outputs, 1)
            val_running_correct += (preds == target).sum().item()
        
        val_loss = val_running_loss/dataset_sizes[VAL]
        val_accuracy = 100. * val_running_correct/dataset_sizes[VAL]        
        return val_loss, val_accuracy
    
def train(epochs, model, criterion, optimizer, scheduler):
    since = time.time()
     
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    use_amp = input("Do you want to use amp? (yes/no)")
    if use_amp == 'yes':
        print('Tranining and validating with Automatic Mixed Precision')
    else:
        print('Tranining and validating without Automatic Mixed Precision')
    
    start = time.time()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_accuracy = fit(model, train_data_loader, 
                                                     optimizer, criterion, 
                                                     train_data, device, use_amp)
        val_epoch_loss, val_epoch_accuracy = validate(model, valid_data_loader, 
                                                     optimizer, criterion, 
                                                     test_data, device, use_amp)
        train_loss.append(train_epoch_loss)
        train_acc.append(train_epoch_accuracy)
        val_loss.append(val_epoch_loss)
        val_acc.append(val_epoch_accuracy)
        print(f"Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_accuracy:.2f}")
        print(f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_accuracy:.2f}')
    end = time.time()
    print(f"Took {((end-start)/60):.3f} minutes to train for {epochs} epochs")

    # save model checkpoint
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f"./output0614_2021/efficient_amp_{use_amp}_model.pt")
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_acc, color='green', label='train accuracy')
    plt.plot(val_acc, color='blue', label='validataion accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"./output0614_2021/efficient_amp_{use_amp}_accuracy.png")
    plt.show()

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(val_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"./output0614_2021/efficient_amp_{use_amp}_loss.png")
    plt.show()
    


```


```python
# train the model
train(100, model_scratch, criterion_scratch, optimizer_scratch, exp_lr_scheduler)
```

    Do you want to use amp? (yes/no)yes


      0%|          | 0/668 [00:00<?, ?it/s]

    Tranining and validating with Automatic Mixed Precision
    Epoch 1 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:11,  6.91it/s]

    Validating


    84it [00:16,  5.04it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0007, Train Acc: 1.48
    Val Loss: 0.0057, Val Acc: 2.63
    Epoch 2 of 100
    Training


    100%|██████████| 668/668 [05:05<00:00,  2.18it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:16,  5.02it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0007, Train Acc: 3.35
    Val Loss: 0.0053, Val Acc: 4.31
    Epoch 3 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:13,  6.12it/s]

    Validating


    84it [00:16,  5.02it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0007, Train Acc: 4.22
    Val Loss: 0.0051, Val Acc: 6.95
    Epoch 4 of 100
    Training


    100%|██████████| 668/668 [05:05<00:00,  2.18it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:16,  5.03it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0006, Train Acc: 5.34
    Val Loss: 0.0049, Val Acc: 8.14
    Epoch 5 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:16,  5.04it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0006, Train Acc: 6.42
    Val Loss: 0.0047, Val Acc: 11.38
    Epoch 6 of 100
    Training


    100%|██████████| 668/668 [05:05<00:00,  2.18it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:16,  5.04it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0006, Train Acc: 7.71
    Val Loss: 0.0046, Val Acc: 9.46
    Epoch 7 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:16,  5.02it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0006, Train Acc: 9.13
    Val Loss: 0.0044, Val Acc: 13.53
    Epoch 8 of 100
    Training


    100%|██████████| 668/668 [05:05<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:13,  6.15it/s]

    Validating


    84it [00:16,  5.03it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0006, Train Acc: 10.42
    Val Loss: 0.0042, Val Acc: 15.09
    Epoch 9 of 100
    Training


    100%|██████████| 668/668 [05:05<00:00,  2.19it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:16,  5.03it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0006, Train Acc: 11.71
    Val Loss: 0.0041, Val Acc: 16.65
    Epoch 10 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:13,  5.98it/s]

    Validating


    84it [00:16,  5.02it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0006, Train Acc: 12.54
    Val Loss: 0.0040, Val Acc: 20.60
    Epoch 11 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:16,  5.02it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0005, Train Acc: 14.69
    Val Loss: 0.0038, Val Acc: 21.44
    Epoch 12 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:14,  5.73it/s]

    Validating


    84it [00:16,  5.02it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0005, Train Acc: 15.82
    Val Loss: 0.0036, Val Acc: 24.43
    Epoch 13 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:12,  6.49it/s]

    Validating


    84it [00:16,  5.02it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0005, Train Acc: 17.02
    Val Loss: 0.0035, Val Acc: 26.71
    Epoch 14 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:14,  5.59it/s]

    Validating


    84it [00:16,  5.03it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0005, Train Acc: 18.77
    Val Loss: 0.0034, Val Acc: 26.35
    Epoch 15 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:16,  5.10it/s]

    Validating


    84it [00:16,  5.01it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0005, Train Acc: 20.04
    Val Loss: 0.0033, Val Acc: 26.11
    Epoch 16 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:13,  5.93it/s]

    Validating


    84it [00:16,  5.01it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0005, Train Acc: 20.88
    Val Loss: 0.0032, Val Acc: 28.74
    Epoch 17 of 100
    Training


    100%|██████████| 668/668 [05:25<00:00,  2.05it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:19,  4.21it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0005, Train Acc: 23.43
    Val Loss: 0.0031, Val Acc: 29.34
    Epoch 18 of 100
    Training


    100%|██████████| 668/668 [05:42<00:00,  1.95it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:17,  4.93it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0005, Train Acc: 24.87
    Val Loss: 0.0029, Val Acc: 34.73
    Epoch 19 of 100
    Training


    100%|██████████| 668/668 [05:07<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:13,  6.28it/s]

    Validating


    84it [00:16,  5.02it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0004, Train Acc: 26.14
    Val Loss: 0.0029, Val Acc: 35.33
    Epoch 20 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:14,  5.62it/s]

    Validating


    84it [00:16,  5.02it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0004, Train Acc: 27.46
    Val Loss: 0.0027, Val Acc: 38.44
    Epoch 21 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:15,  5.36it/s]

    Validating


    84it [00:16,  5.02it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0004, Train Acc: 28.64
    Val Loss: 0.0026, Val Acc: 40.48
    Epoch 22 of 100
    Training


    100%|██████████| 668/668 [05:05<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:13,  6.25it/s]

    Validating


    84it [00:16,  5.03it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0004, Train Acc: 30.72
    Val Loss: 0.0023, Val Acc: 45.27
    Epoch 23 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:14,  5.77it/s]

    Validating


    84it [00:16,  5.02it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0004, Train Acc: 31.60
    Val Loss: 0.0024, Val Acc: 43.95
    Epoch 24 of 100
    Training


    100%|██████████| 668/668 [05:05<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:15,  5.25it/s]

    Validating


    84it [00:16,  5.04it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0004, Train Acc: 34.52
    Val Loss: 0.0022, Val Acc: 49.34
    Epoch 25 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:13,  6.20it/s]

    Validating


    84it [00:16,  5.02it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0004, Train Acc: 36.05
    Val Loss: 0.0022, Val Acc: 47.78
    Epoch 26 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:16,  5.03it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0004, Train Acc: 37.01
    Val Loss: 0.0021, Val Acc: 50.54
    Epoch 27 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:12,  6.76it/s]

    Validating


    84it [00:16,  5.02it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0004, Train Acc: 38.52
    Val Loss: 0.0021, Val Acc: 51.14
    Epoch 28 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:16,  5.02it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0003, Train Acc: 39.36
    Val Loss: 0.0019, Val Acc: 55.33
    Epoch 29 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:13,  6.21it/s]

    Validating


    84it [00:16,  5.02it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0003, Train Acc: 40.91
    Val Loss: 0.0019, Val Acc: 54.13
    Epoch 30 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:12,  6.61it/s]

    Validating


    84it [00:16,  5.03it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0003, Train Acc: 42.57
    Val Loss: 0.0019, Val Acc: 55.81
    Epoch 31 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:12,  6.34it/s]

    Validating


    84it [00:16,  5.03it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0003, Train Acc: 43.74
    Val Loss: 0.0018, Val Acc: 55.69
    Epoch 32 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:12,  6.41it/s]

    Validating


    84it [00:16,  5.03it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0003, Train Acc: 44.55
    Val Loss: 0.0017, Val Acc: 59.52
    Epoch 33 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:16,  5.05it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0003, Train Acc: 45.36
    Val Loss: 0.0018, Val Acc: 58.44
    Epoch 34 of 100
    Training


    100%|██████████| 668/668 [05:05<00:00,  2.18it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:16,  5.03it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0003, Train Acc: 46.74
    Val Loss: 0.0017, Val Acc: 58.32
    Epoch 35 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:14,  5.69it/s]

    Validating


    84it [00:16,  5.03it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0003, Train Acc: 47.47
    Val Loss: 0.0016, Val Acc: 60.72
    Epoch 36 of 100
    Training


    100%|██████████| 668/668 [05:05<00:00,  2.18it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:16,  5.03it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0003, Train Acc: 48.31
    Val Loss: 0.0015, Val Acc: 63.59
    Epoch 37 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:14,  5.82it/s]

    Validating


    84it [00:16,  5.02it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0003, Train Acc: 49.76
    Val Loss: 0.0016, Val Acc: 61.68
    Epoch 38 of 100
    Training


    100%|██████████| 668/668 [05:10<00:00,  2.15it/s]
      1%|          | 1/83 [00:00<00:14,  5.65it/s]

    Validating


    84it [00:16,  5.04it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0003, Train Acc: 50.79
    Val Loss: 0.0015, Val Acc: 65.27
    Epoch 39 of 100
    Training


    100%|██████████| 668/668 [05:05<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:12,  6.34it/s]

    Validating


    84it [00:16,  5.03it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0003, Train Acc: 52.23
    Val Loss: 0.0015, Val Acc: 63.83
    Epoch 40 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:13,  6.11it/s]

    Validating


    84it [00:16,  5.02it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0003, Train Acc: 52.40
    Val Loss: 0.0015, Val Acc: 65.03
    Epoch 41 of 100
    Training


    100%|██████████| 668/668 [05:05<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:14,  5.49it/s]

    Validating


    84it [00:16,  5.02it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0003, Train Acc: 52.63
    Val Loss: 0.0015, Val Acc: 63.71
    Epoch 42 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:16,  5.02it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0003, Train Acc: 53.64
    Val Loss: 0.0014, Val Acc: 62.87
    Epoch 43 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:12,  6.55it/s]

    Validating


    84it [00:16,  5.02it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0003, Train Acc: 53.94
    Val Loss: 0.0013, Val Acc: 68.50
    Epoch 44 of 100
    Training


     51%|█████     | 341/668 [02:36<02:24,  2.26it/s]IOPub message rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_msg_rate_limit`.
    
    Current values:
    NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    
    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:13,  6.07it/s]

    Validating


    84it [00:16,  5.01it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 55.88
    Val Loss: 0.0014, Val Acc: 65.03
    Epoch 46 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:13,  5.91it/s]

    Validating


    84it [00:16,  5.02it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 57.54
    Val Loss: 0.0013, Val Acc: 68.50
    Epoch 47 of 100
    Training


    100%|██████████| 668/668 [05:05<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:15,  5.30it/s]

    Validating


    84it [00:16,  5.02it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 58.20
    Val Loss: 0.0014, Val Acc: 67.43
    Epoch 48 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:13,  5.87it/s]

    Validating


    84it [00:16,  5.03it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 57.31
    Val Loss: 0.0013, Val Acc: 67.31
    Epoch 49 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:14,  5.74it/s]

    Validating


    84it [00:16,  5.02it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 59.48
    Val Loss: 0.0012, Val Acc: 69.94
    Epoch 50 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:16,  5.02it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 60.36
    Val Loss: 0.0013, Val Acc: 67.54
    Epoch 51 of 100
    Training


    100%|██████████| 668/668 [05:05<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:15,  5.27it/s]

    Validating


    84it [00:16,  5.04it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 61.02
    Val Loss: 0.0011, Val Acc: 72.22
    Epoch 52 of 100
    Training


    100%|██████████| 668/668 [05:05<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:13,  6.15it/s]

    Validating


    84it [00:16,  5.02it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 60.03
    Val Loss: 0.0013, Val Acc: 66.83
    Epoch 53 of 100
    Training


    100%|██████████| 668/668 [05:05<00:00,  2.18it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:16,  5.03it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 61.30
    Val Loss: 0.0011, Val Acc: 70.06
    Epoch 54 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:13,  6.29it/s]

    Validating


    84it [00:16,  5.04it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 61.21
    Val Loss: 0.0012, Val Acc: 71.14
    Epoch 55 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:16,  5.04it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 61.29
    Val Loss: 0.0012, Val Acc: 70.78
    Epoch 56 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:13,  6.03it/s]

    Validating


    84it [00:16,  5.02it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 62.59
    Val Loss: 0.0012, Val Acc: 69.70
    Epoch 57 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:14,  5.55it/s]

    Validating


    84it [00:16,  5.03it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 63.76
    Val Loss: 0.0012, Val Acc: 71.98
    Epoch 58 of 100
    Training


    100%|██████████| 668/668 [05:05<00:00,  2.18it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:16,  5.04it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 62.68
    Val Loss: 0.0011, Val Acc: 70.54
    Epoch 59 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:13,  6.13it/s]

    Validating


    84it [00:16,  5.04it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 63.46
    Val Loss: 0.0012, Val Acc: 72.69
    Epoch 60 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:16,  5.04it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 64.37
    Val Loss: 0.0011, Val Acc: 72.10
    Epoch 61 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:15,  5.44it/s]

    Validating


    84it [00:16,  5.03it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 66.11
    Val Loss: 0.0011, Val Acc: 73.89
    Epoch 62 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:16,  5.11it/s]

    Validating


    84it [00:16,  5.02it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 64.82
    Val Loss: 0.0011, Val Acc: 71.74
    Epoch 63 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:16,  5.03it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 64.84
    Val Loss: 0.0011, Val Acc: 71.86
    Epoch 64 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:13,  6.22it/s]

    Validating


    84it [00:16,  5.01it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 65.94
    Val Loss: 0.0012, Val Acc: 73.17
    Epoch 65 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:12,  6.43it/s]

    Validating


    84it [00:16,  5.04it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 66.12
    Val Loss: 0.0011, Val Acc: 73.05
    Epoch 66 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:13,  5.98it/s]

    Validating


    84it [00:16,  5.03it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 66.48
    Val Loss: 0.0011, Val Acc: 73.29
    Epoch 67 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:13,  6.10it/s]

    Validating


    84it [00:16,  5.03it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 68.02
    Val Loss: 0.0010, Val Acc: 73.89
    Epoch 68 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:15,  5.32it/s]

    Validating


    84it [00:16,  5.02it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 67.87
    Val Loss: 0.0010, Val Acc: 73.41
    Epoch 69 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:14,  5.82it/s]

    Validating


    84it [00:16,  5.03it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 67.31
    Val Loss: 0.0011, Val Acc: 73.05
    Epoch 70 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:16,  5.03it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 68.11
    Val Loss: 0.0011, Val Acc: 74.61
    Epoch 71 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:15,  5.34it/s]

    Validating


    84it [00:16,  5.03it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 68.76
    Val Loss: 0.0011, Val Acc: 74.37
    Epoch 72 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:16,  5.03it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 69.12
    Val Loss: 0.0010, Val Acc: 74.49
    Epoch 73 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:15,  5.14it/s]

    Validating


    84it [00:16,  5.04it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 69.13
    Val Loss: 0.0010, Val Acc: 74.85
    Epoch 74 of 100
    Training


    100%|██████████| 668/668 [05:10<00:00,  2.15it/s]
      1%|          | 1/83 [00:00<00:14,  5.60it/s]

    Validating


    84it [00:16,  5.04it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 69.81
    Val Loss: 0.0010, Val Acc: 75.81
    Epoch 75 of 100
    Training


    100%|██████████| 668/668 [05:05<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:14,  5.51it/s]

    Validating


    84it [00:16,  5.04it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 69.36
    Val Loss: 0.0010, Val Acc: 75.45
    Epoch 76 of 100
    Training


    100%|██████████| 668/668 [05:07<00:00,  2.17it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:16,  5.04it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 70.34
    Val Loss: 0.0011, Val Acc: 73.53
    Epoch 77 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:13,  5.99it/s]

    Validating


    84it [00:16,  5.03it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 70.52
    Val Loss: 0.0010, Val Acc: 73.89
    Epoch 78 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:14,  5.49it/s]

    Validating


    84it [00:16,  5.03it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 70.75
    Val Loss: 0.0010, Val Acc: 74.61
    Epoch 79 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:12,  6.38it/s]

    Validating


    84it [00:16,  5.03it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 69.93
    Val Loss: 0.0010, Val Acc: 76.05
    Epoch 80 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:14,  5.57it/s]

    Validating


    84it [00:16,  5.03it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 70.99
    Val Loss: 0.0011, Val Acc: 75.81
    Epoch 81 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:13,  6.09it/s]

    Validating


    84it [00:16,  5.03it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 70.79
    Val Loss: 0.0011, Val Acc: 72.93
    Epoch 82 of 100
    Training


     25%|██▍       | 165/668 [01:16<04:34,  1.83it/s]IOPub message rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_msg_rate_limit`.
    
    Current values:
    NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    
    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:14,  5.77it/s]

    Validating


    84it [00:16,  5.01it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0001, Train Acc: 73.23
    Val Loss: 0.0009, Val Acc: 76.05
    Epoch 90 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:12,  6.36it/s]

    Validating


    84it [00:16,  5.02it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0001, Train Acc: 73.91
    Val Loss: 0.0010, Val Acc: 75.09
    Epoch 91 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:14,  5.64it/s]

    Validating


    84it [00:16,  5.03it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0001, Train Acc: 74.70
    Val Loss: 0.0010, Val Acc: 75.81
    Epoch 92 of 100
    Training


    100%|██████████| 668/668 [05:06<00:00,  2.18it/s]
      1%|          | 1/83 [00:00<00:14,  5.83it/s]

    Validating


    84it [00:16,  5.02it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 72.80
    Val Loss: 0.0010, Val Acc: 74.49
    Epoch 93 of 100
    Training


     83%|████████▎ | 554/668 [04:13<00:53,  2.14it/s]IOPub message rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_msg_rate_limit`.
    
    Current values:
    NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    
    84it [00:16,  5.02it/s]                        


    Train Loss: 0.0001, Train Acc: 75.63
    Val Loss: 0.0010, Val Acc: 76.41
    Took 539.396 minutes to train for 100 epochs



    
![png](EfficientCNN_files/EfficientCNN_4_341.png)
    



    
![png](EfficientCNN_files/EfficientCNN_4_342.png)
    



```python
def test(model, criterion):
    since = time.time()
    # monitor test loss and accuracy
    print("Evaluating model")
    print("-" * 10)
    model.eval()
    
    test_loss = 0.
    test_acc = 0.
    correct = 0.
    total = 0.
    
    for i, data in enumerate(dataloaders[TEST]):
        images, target = data
        # move to GPU
        images, target = images.to(device), target.to(device)
                
        # forward pass: compute predicted outputs by passing inputs to the model
        with torch.no_grad():
            outputs = model(images)
            # calculate the loss
            _, preds = torch.max(outputs.data, 1)
            
            #pred = output.data.max(1, keepdim=True)[1]
            loss = criterion(outputs, target)
            # update average test loss  
            test_loss = test_loss + ((1 / (i + 1)) * (loss.item() - test_loss))
                       
            # compare predictions to true label
            total += target.size(0)
            correct += (preds == target).double().sum().item()
            
    avg_loss = test_loss
    elapsed_time = time.time() - since

    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("\nTest Loss: {:.6f}\n".format(avg_loss))
    print("\nTest Accuracy: {:.3f}% {:.0f}/{:.0f}".format(100. * correct / total, correct, total))
    print('-' * 10)
```


```python
# call test function    
test(model_scratch, criterion_scratch)
```

    Evaluating model
    ----------
    Evaluation completed in 0m 20s
    
    Test Loss: 0.858651
    
    
    Test Accuracy: 76.435% 639/836
    ----------



```python

```
