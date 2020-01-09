# Face Generation
This is face generation project from DLND Udacity. In this project you will used DCGAN to generate new images of faces that 
look as realistic as possible! If you are new to GAN (Generative adversarial network) chek out first this simple [GAN](https://github.com/DishinGoyani/Deep-Learning/tree/master/Generative%20Adversarial%20Network-MNIST#generative-adversarial-network---mnist) 
using MNIST dataset project.  

The project will be broken down into a series of tasks from loading in data to defining and training adversarial networks. 
At the end of the notebook, you'll be able to visualize the results of your trained Generator to see how it performs; your 
generated samples should look like fairly realistic faces with small amounts of noise.  
## Dataset
You will be using [Large-scale CelebFaces Attributes (CelebA) Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
This dataset is more complex than the number datasets (like MNIST or SVHN). It is suggested that you utilize a GPU for training.

## Pre-processed Data
Since the project's main focus is on building the GANs, Udacity has done some of the pre-processing for you. Each of the 
CelebA images has been cropped to remove parts of the image that don't include a face, then resized down to `64x64x3` 
NumPy images. Some sample data is show below. you can download this data from [here](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/November/5be7eb6f_processed-celeba-small/processed-celeba-small.zip) 
This pre-processed dataset is a smaller subset of the very large CelebA data.  

<img src='Face Generation/assets/processed_face_data.png' width=60% />

## Define the Model
### Discriminator
```python
Discriminator(
  (conv1): Sequential(
    (0): Conv2d(3, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
  )
  (conv2): Sequential(
    (0): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (conv3): Sequential(
    (0): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (fc1): Linear(in_features=2048, out_features=1, bias=True)
)
```
### Generator
```python
Generator(
  (fc1): Linear(in_features=100, out_features=2048, bias=True)
  (deconv1): Sequential(
    (0): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (deconv2): Sequential(
    (0): ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (deconv3): Sequential(
    (0): ConvTranspose2d(32, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
  )
)
```
### Generator samples from training
<img src='Face Generation/assets/sample-image-from-generative-adversarial-network-face-generation.png' width=60% />
