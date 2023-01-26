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


#Source: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

dataroot = "data/celeba"
#dataroot = "data/van_Dyck_data_resized"

numGPU = 1
workers = 0
batchSize = 128
#batchSize = 10

imageSize = 64
colorNumber = 3 #nc
zLatent = 100 #nz
generatorFeatureMapSize = 64 #ngf
descriminatorFeatureMapSize = 64

epoch = 5
learningRate = 0.0002

beta1 = 0.5


#Data
dataset = dset.ImageFolder(root=dataroot, transform=transforms.Compose([
    transforms.Resize(imageSize),
    transforms.CenterCrop(imageSize),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ]))

dataLoader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=workers)

device = torch.device("cuda:0" if (torch.cuda.is_available() and numGPU > 0) else "cpu")

#Iterates dataLoader and displays select images
realBatch = next(iter(dataLoader))


def weightsInit(m):
    className = m.__class__.__name__
    if className.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif className.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, numGPU):
        super(Generator, self).__init__()
        self.numGPU = numGPU
        self.main = nn.Sequential(
            #Latent Z as input
            nn.ConvTranspose2d(zLatent, generatorFeatureMapSize * 8, 4,1,0, bias=False),
            nn.BatchNorm2d(generatorFeatureMapSize * 8),
            nn.ReLU(True),

            # Changing size, featureMap*8 x 4 x 4
            nn.ConvTranspose2d(generatorFeatureMapSize*8, generatorFeatureMapSize * 4, 4,2,1, bias=False),
            nn.BatchNorm2d(generatorFeatureMapSize * 4),
            nn.ReLU(True),

            # Changing size, featureMap*4 x 8 x 8
            nn.ConvTranspose2d(generatorFeatureMapSize*4, generatorFeatureMapSize * 2, 4,2,1, bias=False),
            nn.BatchNorm2d(generatorFeatureMapSize * 2),
            nn.ReLU(True),         

            # Changing size, featureMap*2 x 16 x 16
            nn.ConvTranspose2d(generatorFeatureMapSize*2, generatorFeatureMapSize, 4,2,1, bias=False),
            nn.BatchNorm2d(generatorFeatureMapSize),
            nn.ReLU(True),  

            # Changing size, featureMap x 32 x 32
            nn.ConvTranspose2d(generatorFeatureMapSize, colorNumber, 4,2,1, bias=False),
            nn.Tanh()

        )

    def forward(self, input):
        return self.main(input)

netGenerator = Generator(numGPU).to(device)

#if (device.type == 'cuda') and (numGPU > 1):
#    netGenerator = nn.DataParallel(netGenerator, list(range(numGPU)))

netGenerator.apply(weightsInit)

print(netGenerator)



#Discriminator
class Discriminator(nn.Module):
    def __init__(self, numGPU):
        super(Discriminator, self).__init__()
        self.numGPU = numGPU
        self.main = nn.Sequential(
            #input, colorNumber x 64 x 64
            nn.Conv2d(colorNumber, descriminatorFeatureMapSize, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            #State size, featureMap x 32 x 32
            nn.Conv2d(descriminatorFeatureMapSize, descriminatorFeatureMapSize*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(descriminatorFeatureMapSize*2),
            nn.LeakyReLU(0.2, inplace=True),

            #State size, featureMap*2 x 16 x 16
            nn.Conv2d(descriminatorFeatureMapSize*2, descriminatorFeatureMapSize*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(descriminatorFeatureMapSize*4),
            nn.LeakyReLU(0.2, inplace=True),

            #State size, featureMap*4 x 8 x 8
            nn.Conv2d(descriminatorFeatureMapSize*4, descriminatorFeatureMapSize*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(descriminatorFeatureMapSize*8),
            nn.LeakyReLU(0.2, inplace=True),

            #State size, featureMap*8 x 4 x 4
            nn.Conv2d(descriminatorFeatureMapSize*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()

        )

    def forward(self, input):
        return self.main(input)


netDiscriminator = Discriminator(numGPU).to(device)

#if (device.type == 'cuda') and (numGPU > 1):
#   netDiscriminator = nn.DataParallel(netDiscriminator, list(range(numGPU)))

netDiscriminator.apply(weightsInit)

print(netDiscriminator)


#Loss function
criterion = nn.BCELoss()

#Creates latent noise vectors
fixedNoise = torch.randn(64, zLatent, 1, 1, device=device)

REAL_LABEL = 1
FAKE_LABEL = 0

optimizerDiscriminator = optim.Adam(netDiscriminator.parameters(), lr=learningRate, betas=(beta1, 0.999))
optimizerGenerator = optim.Adam(netGenerator.parameters(), lr=learningRate, betas=(beta1, 0.999))



#Training

if __name__ == '__main__':

    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(realBatch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1,2,0)))
        
    imgList = []
    generatorLoss = []
    discriminatorLoss = []
    iterations = 0

    print("Starting Training")

    for tempEpoch in range(epoch):
        print("-------------------")
        for i, data in enumerate(dataLoader, 0):
            
            #1. Update D network, where D = probability generator generates "real img"
            #Maximize log(D(x)) + log(1 - D(G(z)))
            print("""""""""""")
            netDiscriminator.zero_grad()
            #Training with all-real batch
            print("====================")

            realCPU = data[0].to(device)
            batchSize = realCPU.size(0)
            label = torch.full((batchSize, ), REAL_LABEL, dtype=torch.float, device=device)
            
            #Forward pass real batch through D
            output = netDiscriminator(realCPU).view(-1)
            errorDiscriminatorReal = criterion(output, label)
            #Criterion finds gradient

            #Dalculates gradient for D in backward pass
            errorDiscriminatorReal.backward()
            D_x = output.mean().item()


            #Train with all fake batches

            #Generate noise/latent vectors
            noise = torch.randn(batchSize, zLatent, 1, 1, device=device)
            #Use Generator to generate fake images 
            fakeImages = netGenerator(noise)

            label.fill_(FAKE_LABEL)
            output = netDiscriminator(fakeImages.detach()).view(-1)

            errorDiscriminatorFake = criterion(output, label)
            errorDiscriminatorFake.backward()
            D_G_z1 = output.mean().item()

            errorDiscriminator = errorDiscriminatorReal + errorDiscriminatorFake
            #Update D
            optimizerDiscriminator.step()


            #2. 

            #Fake labels are assumed real for generator training
            netGenerator.zero_grad()
            label.fill_(REAL_LABEL)

            #Perform another forward pass of all-fake batch throuhg D
            output = netDiscriminator(fakeImages).view(-1)
            errorGenerator = criterion(output, label)

            errorGenerator.backward()
            D_G_z2 = output.mean().item()
            #Update G
            optimizerGenerator.step()

            #Output training statistics
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, epoch, i, len(dataLoader),
                        errorDiscriminator.item(), errorGenerator.item(), D_x, D_G_z1, D_G_z2))

            #Save losses for plotting
            generatorLoss.append(errorGenerator.item())
            discriminatorLoss.append(errorDiscriminator.item())

            if(iterations % 500 == 0) or ((epoch == epoch-1) and (i == len(dataLoader)-1)):
                with torch.no_grad():
                    fakeImages = netGenerator(fixedNoise).detach().cpu()

                imgList.append(vutils.make_grid(fakeImages, padding=2, normalize=True))

            iterations+=1


    GENERATOR_FILE = "models/generator.pth"
    DISCRIMINATOR_FILE = "models/discriminator.pth"

    torch.save(netGenerator.state_dict(), GENERATOR_FILE)
    torch.save(netDiscriminator.state_dict(), DISCRIMINATOR_FILE)
