# DCGAN

## Introduction

Built a DCGAN model to generate new portraits in the style of one of the best 17th Century portrait painters, Anthony van Dyke.

The model for portrait generation can be consists of a generator-discriminator system. The generator utilizes five sets of convolutional transpose layers to upscale noise to generated images. Discriminators will utilize five CNNs to compare generator images with real painting images during training.

## Files
DCGAN_Train.py: Building and training the DCGAN model

DCGAN_Predictor.py: Generates new images using trained DCGAN model
