## Generative Adversarial Network - MNIST
Simple Example of Generative Adversarial Network with MNIST dataset. In this notebook, we'll be building a generative adversarial network (GAN) trained on the MNIST dataset. From this, we'll be able to generate new handwritten digits!. This is part of exercise of Udacity deep learning nanodegree.  

## Models 

A GAN is comprised of two adversarial networks, a discriminator and a generator.  
### Discriminator model
```
Discriminator(
  (fc1): Linear(in_features=784, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=64, bias=True)
  (fc3): Linear(in_features=64, out_features=32, bias=True)
  (fc4): Linear(in_features=32, out_features=1, bias=True)
  (dropout): Dropout(p=0.3)
)
```
### Generator model
```
Generator(
  (fc1): Linear(in_features=100, out_features=32, bias=True)
  (fc2): Linear(in_features=32, out_features=64, bias=True)
  (fc3): Linear(in_features=64, out_features=128, bias=True)
  (fc4): Linear(in_features=128, out_features=784, bias=True)
  (dropout): Dropout(p=0.3)
)
```

## Training

Training will involve alternating between training the discriminator and the generator.  
Discriminator training

   - Compute the discriminator loss on real, training images
   - Generate fake images
   - Compute the discriminator loss on fake, generated images
   - Add up real and fake loss
   - Perform backpropagation + an optimization step to update the discriminator's weights

Generator training

   - Generate fake images
   - Compute the discriminator loss on fake images, using flipped labels!
   - Perform backpropagation + an optimization step to update the generator's weights  
   
[Here](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/gan-mnist) you can find udacity repo. for more.
