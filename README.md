# DCGAN

## Introduction

Built a DCGAN model to generate new portraits in the style of one of the best 17th Century portrait painters, Anthony van Dyke.

The model for portrait generation can be consists of a generator-discriminator system. The generator utilizes five sets of convolutional transpose layers to upscale noise to generated images. Discriminators will utilize five CNNs to compare generator images with real painting images during training.

## Files
paintingData.py: Creates folder of van Dyck painting images scraped from wiki (https://en.wikipedia.org/wiki/List_of_paintings_by_Anthony_van_Dyck)

DCGAN_Train.py: Building and training the DCGAN model

DCGAN_Predictor.py: Generates new images using trained DCGAN model

Generator Architecture.png: Generator architecture image (https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

## Results and Next Steps
Due to limited training images and inadequate image labeling, the generator model was not able to achieve convergence. I will address these issues in my upcoming diffusion model project



<img width="604" alt="vab_Dyck_Training" src="https://user-images.githubusercontent.com/83440706/214950143-96ddbc96-e8f1-46c0-9d9c-58f6bea59652.png">
