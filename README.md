# DCGAN


<img width="391" alt="Screenshot 2023-01-16 at 10 33 08 PM" src="https://user-images.githubusercontent.com/83440706/214935588-fda47449-e50a-4cb9-9323-30ef05b2a19e.png">
(Source: https://www.google.com/url?sa=i&url=https%3A%2F%2Fcommons.wikimedia.org%2Fwiki%2FFile%3ASir_Anthony_van_Dyck_-_Self-portrait.jpg&psig=AOvVaw0yNV24acusc9OxpRUZkkyG&ust=1674848941504000&source=images&cd=vfe&ved=0CBAQjRxqFwoTCPCmtdOA5vwCFQAAAAAdAAAAABAJ)

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
![van_Dyck_Training](https://user-images.githubusercontent.com/83440706/214935011-45758b1d-61f7-46c0-b042-e8a9fd4a1521.png)
