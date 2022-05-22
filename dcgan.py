from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
dataroot = "data/celeba"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 0

'''
dataroot - 데이터세트 폴더의 루트 경로입니다. 다음 섹션에서 데이터세트에 대해 더 자세히 이야기하겠습니다.
workers - DataLoader로 데이터를 로드하기 위한 작업자 스레드 수
batch_size - 훈련에 사용되는 배치 크기. DCGAN 논문은 128의 배치 크기를 사용합니다.
image_size - 훈련에 사용되는 이미지의 공간적 크기입니다. 이 구현의 기본값은 64x64입니다. 다른 사이즈를 원하시면 D와 G의 구조를 변경하셔야 합니다. 자세한 내용은 여기 를 참조하십시오
nc - 입력 이미지의 색상 채널 수. 컬러 이미지의 경우 3입니다.
nz - 잠재 벡터의 길이
ngf - 생성기를 통해 전달되는 기능 맵의 깊이와 관련됩니다.
ndf - 판별자를 통해 전파되는 기능 맵의 깊이를 설정합니다.
num_epochs - 실행할 훈련 epoch의 수입니다. 더 오래 훈련하면 더 나은 결과를 얻을 수 있지만 훨씬 더 오래 걸립니다.
lr - 훈련을 위한 학습률. DCGAN 논문에 설명된 대로 이 숫자는 0.0002여야 합니다.
beta1 - Adam 옵티마이저용 beta1 하이퍼파라미터. 문서에 설명된 대로 이 숫자는 0.5여야 합니다.
ngpu - 사용 가능한 GPU 수. 이것이 0이면 코드는 CPU 모드에서 실행됩니다. 이 숫자가 0보다 크면 해당 GPU 수에서 실행됩니다.
'''

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))