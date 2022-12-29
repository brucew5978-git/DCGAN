import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from DCGAN_tut import Generator


zLatent = 100
newGenerator = Generator(numGPU=0)

GENERATOR_FILE = "models/generator.pth"
#DISCRIMINATOR_FILE = "models/discriminator.pth"
newGenerator.load_state_dict(torch.load(GENERATOR_FILE, map_location=torch.device('cpu')))
newGenerator.eval()


fixedNoise = torch.randn(64, zLatent, 1, 1, device='cpu')
imgList = []
NEW_IMAGE = 'newImage.jpg'

fakeImage = newGenerator(fixedNoise).detach().cpu()
imgList.append(vutils.make_grid(fakeImage, padding=2, normalize=True))

plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(imgList[-1],(1,2,0)))
plt.show()
