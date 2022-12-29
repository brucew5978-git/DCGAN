import torch.nn as nn
import torch
#from google.colab import files

imageSize = 64
colorNumber = 3 #nc
zLatent = 100 #nz
generatorFeatureMapSize = 64 #ngf

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

GENERATOR_FILE = "generator.pth"

newGenerator = Generator(numGPU=0)
torch.save(newGenerator.state_dict(), GENERATOR_FILE)
#files.download(GENERATOR_FILE)