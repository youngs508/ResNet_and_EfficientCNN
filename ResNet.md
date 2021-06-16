```python
import os
import torch
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init as init
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.manual_seed(777)  # reproducibility

# parameters
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



print("number of training images:", len(image_datasets[TRAIN]))
print("numbe of validation images:", len(image_datasets[VAL]))
print("number of testing images:", len(image_datasets[TEST]))
```

    number of training images: 6680
    numbe of validation images: 835
    number of testing images: 836



```python
import torch.nn.functional as F
import torch.nn as nn
import torch
import argparse

total_dog_classes = 133

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # BatchNorm에 bias가 포함되어 있으므로, conv2d는 bias=False로 설정합니다.
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
        )

        # identity mapping, input과 output의 feature map size, filter 수가 동일한 경우 사용.
        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        # projection mapping using 1x1conv
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x


class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*BottleNeck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*BottleNeck.expansion)
            )
            
    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=total_dog_classes, init_weights=True):
        super().__init__()

        self.in_channels=64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # weights inittialization
        if init_weights:
            self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self,x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        x = self.conv3_x(output)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    # define weight initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def resnet18():
    return ResNet(BasicBlock, [2,2,2,2])

def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    return ResNet(BottleNeck, [3,4,6,3])

def resnet101():
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    return ResNet(BottleNeck, [3, 8, 36, 3])

model_scratch = resnet18()

print(model_scratch)

# move tensors to GPU if CUDA is available
use_cuda = torch.cuda.is_available()
if use_cuda:
    model_scratch.cuda()
else:
    print("cuda is not available.")
```

    ResNet(
      (conv1): Sequential(
        (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      )
      (conv2_x): Sequential(
        (0): BasicBlock(
          (residual_function): Sequential(
            (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (shortcut): Sequential()
          (relu): ReLU()
        )
        (1): BasicBlock(
          (residual_function): Sequential(
            (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (shortcut): Sequential()
          (relu): ReLU()
        )
      )
      (conv3_x): Sequential(
        (0): BasicBlock(
          (residual_function): Sequential(
            (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (shortcut): Sequential(
            (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (relu): ReLU()
        )
        (1): BasicBlock(
          (residual_function): Sequential(
            (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (shortcut): Sequential()
          (relu): ReLU()
        )
      )
      (conv4_x): Sequential(
        (0): BasicBlock(
          (residual_function): Sequential(
            (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (shortcut): Sequential(
            (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (relu): ReLU()
        )
        (1): BasicBlock(
          (residual_function): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (shortcut): Sequential()
          (relu): ReLU()
        )
      )
      (conv5_x): Sequential(
        (0): BasicBlock(
          (residual_function): Sequential(
            (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (shortcut): Sequential(
            (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (relu): ReLU()
        )
        (1): BasicBlock(
          (residual_function): Sequential(
            (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (shortcut): Sequential()
          (relu): ReLU()
        )
      )
      (avg_pool): AdaptiveAvgPool2d(output_size=(1, 1))
      (fc): Linear(in_features=512, out_features=133, bias=True)
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

%matplotlib inline
ImageFile.LOAD_TRUNCATED_IMAGES = True

from tqdm import tqdm
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
                }, f"./output0614_2021/resnet_transfer_amp_{use_amp}_model.pt")
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_acc, color='green', label='train accuracy')
    plt.plot(val_acc, color='blue', label='validataion accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"./output0614_2021/resnet_transfer_amp_{use_amp}_accuracy.png")
    plt.show()

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(val_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"./output0614_2021/resnet_transfer_amp_{use_amp}_loss.png")
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


    100%|██████████| 668/668 [01:47<00:00,  6.23it/s]
      1%|          | 1/83 [00:00<00:14,  5.63it/s]

    Validating


    84it [00:10,  7.70it/s]                        
      0%|          | 1/668 [00:00<01:59,  5.56it/s]

    Train Loss: 0.0007, Train Acc: 1.45
    Val Loss: 0.0056, Val Acc: 3.23
    Epoch 2 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.24it/s]
      1%|          | 1/83 [00:00<00:12,  6.52it/s]

    Validating


    84it [00:10,  7.67it/s]                        
      0%|          | 1/668 [00:00<01:46,  6.27it/s]

    Train Loss: 0.0007, Train Acc: 3.37
    Val Loss: 0.0054, Val Acc: 2.40
    Epoch 3 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.22it/s]
      1%|          | 1/83 [00:00<00:10,  7.55it/s]

    Validating


    84it [00:11,  7.64it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0007, Train Acc: 3.40
    Val Loss: 0.0054, Val Acc: 5.15
    Epoch 4 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.23it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:11,  7.63it/s]                        
      0%|          | 1/668 [00:00<02:10,  5.12it/s]

    Train Loss: 0.0007, Train Acc: 4.24
    Val Loss: 0.0051, Val Acc: 5.51
    Epoch 5 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.23it/s]
      1%|          | 1/83 [00:00<00:08,  9.93it/s]

    Validating


    84it [00:11,  7.60it/s]                        
      0%|          | 1/668 [00:00<01:43,  6.45it/s]

    Train Loss: 0.0006, Train Acc: 5.25
    Val Loss: 0.0052, Val Acc: 5.39
    Epoch 6 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.23it/s]
      2%|▏         | 2/83 [00:00<00:07, 10.72it/s]

    Validating


    84it [00:10,  7.64it/s]                        
      0%|          | 1/668 [00:00<02:03,  5.40it/s]

    Train Loss: 0.0006, Train Acc: 6.20
    Val Loss: 0.0049, Val Acc: 6.23
    Epoch 7 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.23it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:10,  7.67it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0006, Train Acc: 6.33
    Val Loss: 0.0047, Val Acc: 10.06
    Epoch 8 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.22it/s]
      1%|          | 1/83 [00:00<00:09,  9.05it/s]

    Validating


    84it [00:11,  7.59it/s]                        
      0%|          | 1/668 [00:00<01:44,  6.37it/s]

    Train Loss: 0.0006, Train Acc: 7.92
    Val Loss: 0.0049, Val Acc: 7.54
    Epoch 9 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.22it/s]
      2%|▏         | 2/83 [00:00<00:06, 12.20it/s]

    Validating


    84it [00:10,  7.77it/s]                        
      0%|          | 1/668 [00:00<01:45,  6.33it/s]

    Train Loss: 0.0006, Train Acc: 8.91
    Val Loss: 0.0046, Val Acc: 11.26
    Epoch 10 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.23it/s]
      1%|          | 1/83 [00:00<00:15,  5.26it/s]

    Validating


    84it [00:11,  7.59it/s]                        
      0%|          | 1/668 [00:00<01:46,  6.24it/s]

    Train Loss: 0.0006, Train Acc: 9.88
    Val Loss: 0.0045, Val Acc: 11.26
    Epoch 11 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.24it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:10,  7.67it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0006, Train Acc: 11.23
    Val Loss: 0.0047, Val Acc: 10.90
    Epoch 12 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.23it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:11,  7.59it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0006, Train Acc: 11.60
    Val Loss: 0.0045, Val Acc: 10.54
    Epoch 13 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.22it/s]
      2%|▏         | 2/83 [00:00<00:07, 10.95it/s]

    Validating


    84it [00:11,  7.59it/s]                        
      0%|          | 1/668 [00:00<02:02,  5.44it/s]

    Train Loss: 0.0006, Train Acc: 12.74
    Val Loss: 0.0043, Val Acc: 15.45
    Epoch 14 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.20it/s]
      1%|          | 1/83 [00:00<00:08,  9.43it/s]

    Validating


    84it [00:10,  7.66it/s]                        
      0%|          | 1/668 [00:00<01:46,  6.26it/s]

    Train Loss: 0.0005, Train Acc: 13.61
    Val Loss: 0.0040, Val Acc: 16.77
    Epoch 15 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.20it/s]
      1%|          | 1/83 [00:00<00:11,  6.83it/s]

    Validating


    84it [00:10,  7.68it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0005, Train Acc: 14.76
    Val Loss: 0.0040, Val Acc: 18.44
    Epoch 16 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.21it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:11,  7.57it/s]                        
      0%|          | 1/668 [00:00<01:54,  5.84it/s]

    Train Loss: 0.0005, Train Acc: 15.55
    Val Loss: 0.0042, Val Acc: 16.17
    Epoch 17 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.20it/s]
      1%|          | 1/83 [00:00<00:08,  9.14it/s]

    Validating


    84it [00:10,  8.08it/s]                        
      0%|          | 1/668 [00:00<01:37,  6.82it/s]

    Train Loss: 0.0005, Train Acc: 17.22
    Val Loss: 0.0039, Val Acc: 19.64
    Epoch 18 of 100
    Training


    100%|██████████| 668/668 [01:49<00:00,  6.10it/s]
      1%|          | 1/83 [00:00<00:09,  8.52it/s]

    Validating


    84it [00:10,  8.08it/s]                        
      0%|          | 1/668 [00:00<01:18,  8.55it/s]

    Train Loss: 0.0005, Train Acc: 17.47
    Val Loss: 0.0036, Val Acc: 24.67
    Epoch 19 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.21it/s]
      2%|▏         | 2/83 [00:00<00:07, 10.80it/s]

    Validating


    84it [00:11,  7.57it/s]                        
      0%|          | 1/668 [00:00<02:00,  5.53it/s]

    Train Loss: 0.0005, Train Acc: 18.29
    Val Loss: 0.0037, Val Acc: 22.28
    Epoch 20 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.19it/s]
      1%|          | 1/83 [00:00<00:12,  6.65it/s]

    Validating


    84it [00:11,  7.58it/s]                        
      0%|          | 1/668 [00:00<02:00,  5.53it/s]

    Train Loss: 0.0005, Train Acc: 19.63
    Val Loss: 0.0039, Val Acc: 20.96
    Epoch 21 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.20it/s]
      1%|          | 1/83 [00:00<00:13,  6.16it/s]

    Validating


    84it [00:11,  7.54it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0005, Train Acc: 19.93
    Val Loss: 0.0035, Val Acc: 23.83
    Epoch 22 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.21it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:11,  7.57it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0005, Train Acc: 21.59
    Val Loss: 0.0036, Val Acc: 24.19
    Epoch 23 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.22it/s]
      1%|          | 1/83 [00:00<00:11,  7.13it/s]

    Validating


    84it [00:10,  7.69it/s]                        
      0%|          | 1/668 [00:00<01:46,  6.24it/s]

    Train Loss: 0.0005, Train Acc: 23.58
    Val Loss: 0.0032, Val Acc: 31.14
    Epoch 24 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.22it/s]
      2%|▏         | 2/83 [00:00<00:07, 10.66it/s]

    Validating


    84it [00:10,  7.70it/s]                        
      0%|          | 1/668 [00:00<01:45,  6.32it/s]

    Train Loss: 0.0005, Train Acc: 24.76
    Val Loss: 0.0031, Val Acc: 32.22
    Epoch 25 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.23it/s]
      2%|▏         | 2/83 [00:00<00:07, 11.08it/s]

    Validating


    84it [00:11,  7.57it/s]                        
      0%|          | 1/668 [00:00<02:00,  5.51it/s]

    Train Loss: 0.0005, Train Acc: 25.42
    Val Loss: 0.0033, Val Acc: 30.18
    Epoch 26 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.20it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:11,  7.57it/s]                        
      0%|          | 1/668 [00:00<01:46,  6.29it/s]

    Train Loss: 0.0004, Train Acc: 26.90
    Val Loss: 0.0031, Val Acc: 33.41
    Epoch 27 of 100
    Training


     33%|███▎      | 221/668 [00:36<01:09,  6.45it/s]IOPub message rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_msg_rate_limit`.
    
    Current values:
    NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    
    100%|██████████| 668/668 [01:47<00:00,  6.20it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:11,  7.62it/s]                        
      0%|          | 1/668 [00:00<02:00,  5.56it/s]

    Train Loss: 0.0004, Train Acc: 35.76
    Val Loss: 0.0024, Val Acc: 44.07
    Epoch 35 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.20it/s]
      2%|▏         | 2/83 [00:00<00:08, 10.07it/s]

    Validating


    84it [00:10,  7.67it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0004, Train Acc: 35.85
    Val Loss: 0.0029, Val Acc: 38.56
    Epoch 36 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.19it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:11,  7.54it/s]                        
      0%|          | 1/668 [00:00<01:57,  5.69it/s]

    Train Loss: 0.0004, Train Acc: 36.65
    Val Loss: 0.0026, Val Acc: 41.32
    Epoch 37 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.20it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:11,  7.54it/s]                        
      0%|          | 1/668 [00:00<01:51,  5.99it/s]

    Train Loss: 0.0004, Train Acc: 38.23
    Val Loss: 0.0025, Val Acc: 45.99
    Epoch 38 of 100
    Training


     16%|█▌        | 106/668 [00:18<01:26,  6.51it/s]IOPub message rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_msg_rate_limit`.
    
    Current values:
    NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    
    100%|██████████| 668/668 [01:48<00:00,  6.18it/s]
      2%|▏         | 2/83 [00:00<00:07, 11.08it/s]

    Validating


    84it [00:11,  7.63it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0003, Train Acc: 45.75
    Val Loss: 0.0024, Val Acc: 47.07
    Epoch 45 of 100
    Training


    100%|██████████| 668/668 [01:48<00:00,  6.18it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:11,  7.55it/s]                        
      0%|          | 1/668 [00:00<02:05,  5.30it/s]

    Train Loss: 0.0003, Train Acc: 44.00
    Val Loss: 0.0022, Val Acc: 51.98
    Epoch 46 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.21it/s]
      2%|▏         | 2/83 [00:00<00:07, 10.54it/s]

    Validating


    84it [00:10,  7.65it/s]                        
      0%|          | 1/668 [00:00<01:37,  6.84it/s]

    Train Loss: 0.0003, Train Acc: 46.30
    Val Loss: 0.0025, Val Acc: 46.83
    Epoch 47 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.21it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:11,  7.59it/s]                        
      0%|          | 1/668 [00:00<02:01,  5.47it/s]

    Train Loss: 0.0003, Train Acc: 46.95
    Val Loss: 0.0021, Val Acc: 52.10
    Epoch 48 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.20it/s]
      1%|          | 1/83 [00:00<00:10,  7.91it/s]

    Validating


      8%|▊         | 7/83 [00:01<00:10,  7.01it/s]IOPub message rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_msg_rate_limit`.
    
    Current values:
    NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    
    100%|██████████| 668/668 [01:47<00:00,  6.21it/s]
      2%|▏         | 2/83 [00:00<00:07, 10.81it/s]

    Validating


    84it [00:11,  7.60it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0003, Train Acc: 47.72
    Val Loss: 0.0019, Val Acc: 56.41
    Epoch 50 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.20it/s]
      2%|▏         | 2/83 [00:00<00:07, 10.63it/s]

    Validating


    84it [00:10,  7.66it/s]                        
      0%|          | 1/668 [00:00<01:47,  6.18it/s]

    Train Loss: 0.0003, Train Acc: 49.12
    Val Loss: 0.0021, Val Acc: 50.90
    Epoch 51 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.21it/s]
      1%|          | 1/83 [00:00<00:10,  7.63it/s]

    Validating


    84it [00:11,  7.60it/s]                        
      0%|          | 1/668 [00:00<01:46,  6.27it/s]

    Train Loss: 0.0003, Train Acc: 50.09
    Val Loss: 0.0018, Val Acc: 57.01
    Epoch 52 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.20it/s]
      1%|          | 1/83 [00:00<00:14,  5.77it/s]

    Validating


    84it [00:11,  7.53it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0003, Train Acc: 49.90
    Val Loss: 0.0022, Val Acc: 53.29
    Epoch 53 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.20it/s]
      1%|          | 1/83 [00:00<00:09,  8.80it/s]

    Validating


    84it [00:10,  7.66it/s]                        
      0%|          | 1/668 [00:00<01:48,  6.16it/s]

    Train Loss: 0.0003, Train Acc: 51.00
    Val Loss: 0.0018, Val Acc: 57.96
    Epoch 54 of 100
    Training


    100%|██████████| 668/668 [01:48<00:00,  6.17it/s]
      1%|          | 1/83 [00:00<00:10,  7.48it/s]

    Validating


    84it [00:11,  7.53it/s]                        
      0%|          | 1/668 [00:00<01:50,  6.03it/s]

    Train Loss: 0.0003, Train Acc: 51.20
    Val Loss: 0.0017, Val Acc: 60.24
    Epoch 55 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.20it/s]
      2%|▏         | 2/83 [00:00<00:06, 12.59it/s]

    Validating


    84it [00:10,  7.68it/s]                        
      0%|          | 1/668 [00:00<01:48,  6.16it/s]

    Train Loss: 0.0003, Train Acc: 52.49
    Val Loss: 0.0019, Val Acc: 58.56
    Epoch 56 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.21it/s]
      1%|          | 1/83 [00:00<00:09,  8.88it/s]

    Validating


    84it [00:10,  7.68it/s]                        
      0%|          | 1/668 [00:00<01:49,  6.11it/s]

    Train Loss: 0.0003, Train Acc: 53.32
    Val Loss: 0.0018, Val Acc: 57.49
    Epoch 57 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.21it/s]
      1%|          | 1/83 [00:00<00:09,  8.20it/s]

    Validating


    84it [00:11,  7.55it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0003, Train Acc: 53.89
    Val Loss: 0.0018, Val Acc: 59.76
    Epoch 58 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.19it/s]
      2%|▏         | 2/83 [00:00<00:07, 11.06it/s]

    Validating


    84it [00:11,  7.58it/s]                        
      0%|          | 1/668 [00:00<01:50,  6.06it/s]

    Train Loss: 0.0003, Train Acc: 55.40
    Val Loss: 0.0018, Val Acc: 58.56
    Epoch 59 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.19it/s]
      1%|          | 1/83 [00:00<00:09,  8.81it/s]

    Validating


    84it [00:11,  7.54it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0003, Train Acc: 54.64
    Val Loss: 0.0016, Val Acc: 61.68
    Epoch 60 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.19it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:10,  7.64it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 55.40
    Val Loss: 0.0018, Val Acc: 57.37
    Epoch 61 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.19it/s]
      2%|▏         | 2/83 [00:00<00:07, 11.06it/s]

    Validating


    84it [00:11,  7.62it/s]                        
      0%|          | 1/668 [00:00<02:03,  5.42it/s]

    Train Loss: 0.0003, Train Acc: 55.09
    Val Loss: 0.0018, Val Acc: 58.56
    Epoch 62 of 100
    Training


    100%|██████████| 668/668 [01:48<00:00,  6.18it/s]
      1%|          | 1/83 [00:00<00:09,  8.49it/s]

    Validating


    84it [00:11,  7.62it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 56.47
    Val Loss: 0.0016, Val Acc: 62.51
    Epoch 63 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.19it/s]
      1%|          | 1/83 [00:00<00:08,  9.56it/s]

    Validating


    84it [00:11,  7.59it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 56.77
    Val Loss: 0.0017, Val Acc: 62.99
    Epoch 64 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.21it/s]
      2%|▏         | 2/83 [00:00<00:07, 11.30it/s]

    Validating


    84it [00:10,  7.66it/s]                        
      0%|          | 1/668 [00:00<02:06,  5.27it/s]

    Train Loss: 0.0002, Train Acc: 55.84
    Val Loss: 0.0017, Val Acc: 59.88
    Epoch 65 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.22it/s]
      1%|          | 1/83 [00:00<00:09,  8.97it/s]

    Validating


    84it [00:11,  7.59it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 58.19
    Val Loss: 0.0017, Val Acc: 60.12
    Epoch 66 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.20it/s]
      1%|          | 1/83 [00:00<00:10,  7.74it/s]

    Validating


    84it [00:11,  7.59it/s]                        
      0%|          | 1/668 [00:00<01:45,  6.32it/s]

    Train Loss: 0.0002, Train Acc: 58.49
    Val Loss: 0.0016, Val Acc: 61.08
    Epoch 67 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.20it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:11,  7.60it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 58.58
    Val Loss: 0.0019, Val Acc: 56.77
    Epoch 68 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.20it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:10,  7.66it/s]                        
      0%|          | 1/668 [00:00<01:36,  6.88it/s]

    Train Loss: 0.0002, Train Acc: 58.74
    Val Loss: 0.0017, Val Acc: 61.80
    Epoch 69 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.21it/s]
      1%|          | 1/83 [00:00<00:10,  8.12it/s]

    Validating


    84it [00:11,  7.58it/s]                        
      0%|          | 1/668 [00:00<01:41,  6.54it/s]

    Train Loss: 0.0002, Train Acc: 60.61
    Val Loss: 0.0017, Val Acc: 61.20
    Epoch 70 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.19it/s]
      1%|          | 1/83 [00:00<00:09,  8.56it/s]

    Validating


    84it [00:10,  7.66it/s]                        
      0%|          | 1/668 [00:00<01:44,  6.35it/s]

    Train Loss: 0.0002, Train Acc: 60.30
    Val Loss: 0.0017, Val Acc: 60.84
    Epoch 71 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.20it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:10,  8.05it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 59.25
    Val Loss: 0.0016, Val Acc: 64.55
    Epoch 72 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.20it/s]
      2%|▏         | 2/83 [00:00<00:07, 10.37it/s]

    Validating


    84it [00:10,  8.06it/s]                        
      0%|          | 1/668 [00:00<01:20,  8.32it/s]

    Train Loss: 0.0002, Train Acc: 60.76
    Val Loss: 0.0017, Val Acc: 61.32
    Epoch 73 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.19it/s]
      1%|          | 1/83 [00:00<00:12,  6.80it/s]

    Validating


    84it [00:10,  8.03it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 61.41
    Val Loss: 0.0016, Val Acc: 61.68
    Epoch 74 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.19it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:10,  7.65it/s]                        
      0%|          | 1/668 [00:00<01:38,  6.79it/s]

    Train Loss: 0.0002, Train Acc: 61.72
    Val Loss: 0.0016, Val Acc: 61.68
    Epoch 75 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.19it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:10,  7.67it/s]                        
      0%|          | 1/668 [00:00<01:51,  5.98it/s]

    Train Loss: 0.0002, Train Acc: 62.19
    Val Loss: 0.0015, Val Acc: 64.07
    Epoch 76 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.19it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:10,  7.66it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 62.25
    Val Loss: 0.0016, Val Acc: 64.07
    Epoch 77 of 100
    Training


    100%|██████████| 668/668 [01:48<00:00,  6.17it/s]
      1%|          | 1/83 [00:00<00:12,  6.47it/s]

    Validating


    84it [00:10,  7.65it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 62.71
    Val Loss: 0.0018, Val Acc: 58.80
    Epoch 78 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.19it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:11,  7.55it/s]                        
      0%|          | 1/668 [00:00<01:40,  6.64it/s]

    Train Loss: 0.0002, Train Acc: 62.87
    Val Loss: 0.0017, Val Acc: 61.56
    Epoch 79 of 100
    Training


    100%|██████████| 668/668 [01:48<00:00,  6.18it/s]
      1%|          | 1/83 [00:00<00:10,  7.99it/s]

    Validating


    84it [00:11,  7.57it/s]                        
      0%|          | 1/668 [00:00<01:55,  5.77it/s]

    Train Loss: 0.0002, Train Acc: 63.95
    Val Loss: 0.0014, Val Acc: 68.74
    Epoch 80 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.19it/s]
      2%|▏         | 2/83 [00:00<00:06, 11.82it/s]

    Validating


    84it [00:10,  7.66it/s]                        
      0%|          | 1/668 [00:00<01:45,  6.29it/s]

    Train Loss: 0.0002, Train Acc: 65.28
    Val Loss: 0.0017, Val Acc: 64.43
    Epoch 81 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.21it/s]
      1%|          | 1/83 [00:00<00:08,  9.48it/s]

    Validating


    84it [00:11,  7.57it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 65.15
    Val Loss: 0.0016, Val Acc: 65.51
    Epoch 82 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.22it/s]
      1%|          | 1/83 [00:00<00:09,  8.78it/s]

    Validating


    84it [00:11,  7.59it/s]                        
      0%|          | 1/668 [00:00<01:35,  7.02it/s]

    Train Loss: 0.0002, Train Acc: 65.04
    Val Loss: 0.0016, Val Acc: 62.63
    Epoch 83 of 100
    Training


    100%|██████████| 668/668 [01:48<00:00,  6.18it/s]
      1%|          | 1/83 [00:00<00:12,  6.41it/s]

    Validating


    84it [00:11,  7.59it/s]                        
      0%|          | 1/668 [00:00<02:00,  5.51it/s]

    Train Loss: 0.0002, Train Acc: 65.28
    Val Loss: 0.0017, Val Acc: 66.83
    Epoch 84 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.19it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:11,  7.55it/s]                        
      0%|          | 1/668 [00:00<01:50,  6.02it/s]

    Train Loss: 0.0002, Train Acc: 65.85
    Val Loss: 0.0016, Val Acc: 64.31
    Epoch 85 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.19it/s]
      1%|          | 1/83 [00:00<00:11,  7.29it/s]

    Validating


    84it [00:11,  7.60it/s]                        
      0%|          | 1/668 [00:00<02:04,  5.34it/s]

    Train Loss: 0.0002, Train Acc: 64.31
    Val Loss: 0.0017, Val Acc: 62.99
    Epoch 86 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.20it/s]
      1%|          | 1/83 [00:00<00:08,  9.30it/s]

    Validating


    84it [00:11,  7.62it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 66.56
    Val Loss: 0.0016, Val Acc: 64.67
    Epoch 87 of 100
    Training


    100%|██████████| 668/668 [01:48<00:00,  6.16it/s]
      1%|          | 1/83 [00:00<00:11,  6.84it/s]

    Validating


    84it [00:11,  7.57it/s]                        
      0%|          | 1/668 [00:00<02:12,  5.04it/s]

    Train Loss: 0.0002, Train Acc: 66.08
    Val Loss: 0.0015, Val Acc: 64.55
    Epoch 88 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.21it/s]
      1%|          | 1/83 [00:00<00:08,  9.69it/s]

    Validating


    84it [00:11,  7.60it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 66.75
    Val Loss: 0.0015, Val Acc: 66.23
    Epoch 89 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.21it/s]
      1%|          | 1/83 [00:00<00:09,  8.56it/s]

    Validating


    84it [00:11,  7.62it/s]                        
      0%|          | 1/668 [00:00<02:07,  5.23it/s]

    Train Loss: 0.0002, Train Acc: 67.75
    Val Loss: 0.0015, Val Acc: 68.50
    Epoch 90 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.20it/s]
      1%|          | 1/83 [00:00<00:09,  8.69it/s]

    Validating


    84it [00:11,  7.60it/s]                        
      0%|          | 1/668 [00:00<01:55,  5.77it/s]

    Train Loss: 0.0002, Train Acc: 68.05
    Val Loss: 0.0014, Val Acc: 67.43
    Epoch 91 of 100
    Training


    100%|██████████| 668/668 [01:48<00:00,  6.18it/s]
      1%|          | 1/83 [00:00<00:08,  9.44it/s]

    Validating


    84it [00:10,  7.69it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 68.82
    Val Loss: 0.0016, Val Acc: 66.23
    Epoch 92 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.20it/s]
      1%|          | 1/83 [00:00<00:08,  9.53it/s]

    Validating


    84it [00:11,  7.63it/s]                        
      0%|          | 1/668 [00:00<01:45,  6.33it/s]

    Train Loss: 0.0002, Train Acc: 68.37
    Val Loss: 0.0015, Val Acc: 66.11
    Epoch 93 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.21it/s]
      2%|▏         | 2/83 [00:00<00:07, 11.26it/s]

    Validating


    84it [00:11,  7.62it/s]                        
      0%|          | 1/668 [00:00<01:54,  5.84it/s]

    Train Loss: 0.0002, Train Acc: 68.29
    Val Loss: 0.0014, Val Acc: 68.02
    Epoch 94 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.20it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:11,  7.60it/s]                        
      0%|          | 1/668 [00:00<01:36,  6.93it/s]

    Train Loss: 0.0002, Train Acc: 68.82
    Val Loss: 0.0014, Val Acc: 69.82
    Epoch 95 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.21it/s]
      2%|▏         | 2/83 [00:00<00:07, 10.71it/s]

    Validating


    84it [00:11,  7.58it/s]                        
      0%|          | 1/668 [00:00<01:54,  5.84it/s]

    Train Loss: 0.0002, Train Acc: 68.95
    Val Loss: 0.0015, Val Acc: 68.26
    Epoch 96 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.21it/s]
      2%|▏         | 2/83 [00:00<00:06, 11.58it/s]

    Validating


    84it [00:11,  7.60it/s]                        
      0%|          | 0/668 [00:00<?, ?it/s]

    Train Loss: 0.0002, Train Acc: 68.88
    Val Loss: 0.0015, Val Acc: 67.54
    Epoch 97 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.21it/s]
      2%|▏         | 2/83 [00:00<00:07, 11.09it/s]

    Validating


    84it [00:11,  7.62it/s]                        
      0%|          | 1/668 [00:00<01:50,  6.04it/s]

    Train Loss: 0.0002, Train Acc: 69.76
    Val Loss: 0.0015, Val Acc: 66.71
    Epoch 98 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.21it/s]
      1%|          | 1/83 [00:00<00:09,  8.42it/s]

    Validating


    84it [00:10,  7.72it/s]                        
      0%|          | 1/668 [00:00<01:41,  6.54it/s]

    Train Loss: 0.0002, Train Acc: 70.13
    Val Loss: 0.0014, Val Acc: 67.90
    Epoch 99 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.22it/s]
      2%|▏         | 2/83 [00:00<00:06, 12.03it/s]

    Validating


    84it [00:11,  7.63it/s]                        
      0%|          | 1/668 [00:00<02:06,  5.25it/s]

    Train Loss: 0.0002, Train Acc: 70.04
    Val Loss: 0.0015, Val Acc: 65.39
    Epoch 100 of 100
    Training


    100%|██████████| 668/668 [01:47<00:00,  6.22it/s]
      1%|          | 1/83 [00:00<00:10,  7.65it/s]

    Validating


    84it [00:10,  7.67it/s]                        


    Train Loss: 0.0002, Train Acc: 70.25
    Val Loss: 0.0016, Val Acc: 65.03
    Took 197.836 minutes to train for 100 epochs



    
![png](ResNet_files/ResNet_4_349.png)
    



    
![png](ResNet_files/ResNet_4_350.png)
    



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
# load the model that got the best validation accuracy
#model_scratch.load_state_dict(torch.load("./output0614_2021/resnet_amp_yes_model.pt"))

# call test function    
test(model_scratch, criterion_scratch)
```

    Evaluating model
    ----------
    Evaluation completed in 0m 13s
    
    Test Loss: 1.329387
    
    
    Test Accuracy: 66.148% 553/836
    ----------



```python
import torchvision.models as models
import torch.nn as nn

## TODO: Specify model architecture 

model_transfer = models.resnet18(pretrained=True)
for param in model_transfer.parameters():
    param.requires_grad = True
    
num_ftrs = model_transfer.fc.in_features
model_transfer.fc = nn.Linear(num_ftrs, len(class_names))
```


```python
import torch.optim as optim

criterion_transfer = nn.CrossEntropyLoss()
    
optimizer_transfer = optim.SGD(model_transfer.parameters(), lr=0.001, momentum=0.9)
exp_transfer_scheduler = lr_scheduler.StepLR(optimizer_transfer, step_size=7, gamma=0.1)
```


```python
from PIL import Image, ImageFile
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True


# train the model
n_epochs = 2

train(n_epochs, model_transfer.cuda(), criterion_transfer, optimizer_transfer, exp_transfer_scheduler)
```

    Do you want to use amp? (yes/no)yes


      0%|          | 0/668 [00:00<?, ?it/s]

    Tranining and validating with Automatic Mixed Precision
    Epoch 1 of 2
    Training


    100%|██████████| 668/668 [02:18<00:00,  4.82it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:11,  7.13it/s]                        
      0%|          | 1/668 [00:00<01:56,  5.71it/s]

    Train Loss: 0.0005, Train Acc: 28.83
    Val Loss: 0.0015, Val Acc: 68.74
    Epoch 2 of 2
    Training


    100%|██████████| 668/668 [02:19<00:00,  4.79it/s]
      0%|          | 0/83 [00:00<?, ?it/s]

    Validating


    84it [00:11,  7.07it/s]                        


    Train Loss: 0.0003, Train Acc: 56.00
    Val Loss: 0.0010, Val Acc: 75.57
    Took 5.030 minutes to train for 2 epochs



    
![png](ResNet_files/ResNet_9_11.png)
    



    
![png](ResNet_files/ResNet_9_12.png)
    



```python
test(model_transfer, criterion_transfer)
```

    Evaluating model
    ----------
    Evaluation completed in 0m 12s
    
    Test Loss: 0.835666
    
    
    Test Accuracy: 76.555% 640/836
    ----------



```python

```
