# **[Under Development]**
# GANs N' ROSES

Uses a Deep Convolutional Generative Adversial Network to generate images of roses using tensorflow.

The main of this repo is to provide users with a foundation to experiment on GANs with worrying about saving models, generated images and the Tensorboard
results. All the functions are well documented and instructions to get started on the
theory and the explaination is provided [here]().

## Installing the dependencies
 Install virtualenv and creating a new virtual environment:

    pip install virtualenv
    virtualenv -p /usr/bin/python3 gnr
    
 Install the dependencies
    
    pip3 install -r requirements.txt
    
*Note:*
* *Install tensorflow using the following [link](https://www.tensorflow.org/install/).*
* *It is recomended that you use the GPU during training.*

## Dataset
The roses dataset was downloaded from google images using the Chrome ImageSpark 
plugin.

The dataset that I have used can be downloaded from this [link](https://drive.google.com/open?id=0B068a_0Gq8kYSGZ3UmdveFczM0U).
* Extract the images into Dataset/Roses.

## Training the model

### mission_control.py
This file contains the dataset location and the hyperparameters that is used by the
model.

*Note:*
* *IMAGE_SIZE : currently buggy **don't** change it from 64. The Generator model
has to be modified in order to change the image size at the output of the generator. IMAGE_SIZE only 
changes the size of the real images feed to the Discriminator.*

### Training
Set the hyperparameters to the required values in the mission_control.py file
and run the following:

    python3 main.py
    
* Each run generates a new directory under Results/roses with the following format:
    
    <time_stamp>\_<Z_DIM>\_<BATCH_SIZE>\_<N_ITERATIONS>\_<LEARNING_RATE>_<BETA_1>

* This trains the model displaying the generated images after every 200 iternations
in the Results/roses/<time_stamp...>/Generated_Images directory. 

* The Discriminator and the Generator loss along with the Discriminator accuracy 
variations are plotted on Tensorboard which can be seen by running the code below.
    
    
    tensorboard --logdir="Results/roses/<time_stamp...>/Tensorboard"

### Examples of generated images after 30000 iterations:
The example images were generated using the default hyperparameters present in the 
mission_control.py file.

*Example 1*

![Example_1](https://raw.githubusercontent.com/Naresh1318/GANs_N_Roses/master/README/example_1.jpg)

*Example 2*

![Example_2](https://raw.githubusercontent.com/Naresh1318/GANs_N_Roses/master/README/example_2.jpg)

*Example 3 gif*

![Gif_1](https://raw.githubusercontent.com/Naresh1318/GANs_N_Roses/master/README/gif_1.gif)


## Credits
..


    


