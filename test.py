import matplotlib.pyplot as plt
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np

'''plt.plot([1, 2, 3, 4])
plt.ylabel('some numbers')
plt.show()'''

fashion = dset.FashionMNIST(root='data')

batchSize = 128
imageSize = 64
workers=0

dataroot="data/celeba"
dataset = dset.ImageFolder(root=dataroot, transform=transforms.Compose([
    transforms.Resize(imageSize),
    transforms.CenterCrop(imageSize),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ]))

dataLoader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=workers)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

#Iterates dataLoader and displays select images
realBatch = next(iter(dataLoader))

plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(realBatch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1,2,0)))
