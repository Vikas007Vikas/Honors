from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop
from keras.models import Model
from keras.datasets import mnist
from keras.models import load_model
from utils import load_images

import numpy as np
import math
import matplotlib.pyplot as plt
import os
import argparse

def build_generator(inputs,image_size):
    """
    #Arguments
        inputs (Layer): Input layer of the generator(the z-vector)
        image_size: Target size(size of depth image required)
    #Returns
        Model: Generator model
    """
    #Encoder
    kernel_size = 5
    layer_filters = [128,256,512,1024]

    x = inputs
    for filters in layer_filters:
        strides = 2
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding='same')(x)

    #decoder
    layer_filters = [512,256,128,64]
    kernel_size = 5
    for filters in layer_filters:
        strides = 2
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(filters=filters,kernel_size=kernel_size,strides=strides,padding='same')(x)

    x = Conv2D(filters=1,kernel_size=1,strides=1,padding='same')(x)
    x = Activation('tanh')(x)

    generator = Model(inputs,x,name='generator')
    return generator

def build_discriminator(inputs):
    """
    #Arguments
        inputs (Layer): Input layer of the discriminator
    #Returns
        Model: discriminator model
    """
    kernel_size = 5
    layer_filters = [128,256,512,1024]

    x = inputs
    for filters in layer_filters:
        strides = 2
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding='same')(x)

    x = Flatten()(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    discriminator = Model(inputs,x,name='discriminator')
    return discriminator

def train(models,images,x_train,params):
    """
    # Arguments
        models (list): Generator, Discriminator, Adversarial models
        images : input images for generator
        x_train (tensor): ground truth depth images
        params (list) : Networks parameters
    """
    # the GAN models
    generator, discriminator, adversarial = models
    # network parameters
    batch_size, latent_size, train_steps, model_name = params
    #generator image is saved every 500 steps
    save_interval = 500

    #number of elements in the train dataset
    train_size = x_train.shape[0]
    images_size = images.shape[0]
    for i in range(train_steps):
        rand_indexes = np.random.randint(0,train_size,size=batch_size)
        depth_images_real = x_train[rand_indexes]

        rand_indexes = np.random.randint(0,images_size,size=batch_size)
        input_images = images[rand_indexes]
        print('added:',input_images.shape)
        depth_images_fake = generator.predict(input_images)

        # real+fake images = 1 batch of train data
        x = np.concatenate((depth_images_real,depth_images_fake))
        #label real and fake images
        #real images label is 1.0
        y = np.ones([2*batch_size,1])
        #fake images label is 0.0
        y[batch_size:,:]=0.0
        #train discriminator network, log the loss and accuracy
        loss, acc = discriminator.train_on_batch(x,y)
        log = "%d: [discriminator loss: %f, acc: %f]" % (i, loss, acc)

        #train the adversarial network for 1 batch
        # 1 batch of fake images with label=1.0
        # since the discriminator weights are frozen in adversarial network
        # only the generator is trained
        rand_indexes = np.random.randint(0,images_size,size=batch_size)
        input_images = images[rand_indexes]
        #label fake_images as real or 1.0
        y = np.ones([batch_size,1])
        # train the adversarial network
        # note that unlike in discriminator training,
        # we do not save the fake images in a variable
        # the fake images go to the discriminator input of the adversarial
        # for classification
        # log the loss and accuracy
        loss, acc = adversarial.train_on_batch(input_images,y)
        log = "%s [adversarial loss: %f, acc: %f]" % (log, loss, acc)
        print(log)
        if (i + 1) % save_interval == 0:
            if (i + 1) == train_steps:
                show = True
            else:
                show = False

            # plot generator images on a periodic basis
            plot_images(generator,
                        noise_input=noise_input,
                        show=show,
                        step=(i + 1),
                        model_name=model_name)
    # save the model after training the generator
    # the trained generator can be reloaded for future depth generation
    generator.save(model_name + ".h5")

def plot_images(generator,
                noise_input,
                show=False,
                step=0,
                model_name="gan"):
    if not os.path.isdir(model_name):
        os.makedirs(model_name)
    filename = os.path.join(model_name, "%05d.png" % step)
    images = generator.predict(noise_input)
    plt.figure(figsize=(2.2, 2.2))
    num_images = images.shape[0]
    image_size = images.shape[1]
    rows = int(math.sqrt(noise_input.shape[0]))
    for i in range(num_images):
        plt.subplot(rows, rows, i + 1)
        image = np.reshape(images[i], [image_size, image_size])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.savefig(filename)
    if show:
        plt.show()
    else:
        plt.close('all')

def build_and_train_models():
    data = load_images('./images')
    #load rgb face images from dataset into images
    images = data['images']
    images = np.reshape(images, [-1, images.shape[1], images.shape[2], 1])
    print("images:",images.shape)
    #image size is (640,480,3) -> convert to greyscale (640,480,1)

    #load train images-> depth_images from dataset into x_train
    x_train = data['depth_maps']
    x_train = np.reshape(x_train, [-1, x_train.shape[1], x_train.shape[2], 1])
    print("depth_maps:",x_train.shape)

    #defining model
    model_name = "my_gan_implementation"
    #network parameters
    #the input vector or z vector is 480x640x1
    latent_size = (images.shape[1],images.shape[2],1)
    print("z-vector:",latent_size)
    batch_size = 1
    train_steps = 10000
    lr = 2e-4
    decay = 6e-8

    #build discriminator model
    input_shape = (x_train.shape[1],x_train.shape[2],1)
    print("discriminator input_shape:",input_shape)
    inputs = Input(shape=input_shape,name='discriminator_input')
    discriminator = build_discriminator(inputs)
    #using RMSprop as optimizer
    optimizer = RMSprop(lr=lr, decay=decay)
    discriminator.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    discriminator.summary()

    #build generator model
    input_shape = latent_size
    print("generator input_shape:",input_shape)
    inputs = Input(shape=input_shape,name='z_input')
    generator = build_generator(inputs,latent_size)
    generator.summary()

    #build adversarial model
    optimizer = RMSprop(lr=lr*0.5,decay=decay*0.5)
    #freeze the weights of discriminator during adversarial training
    discriminator.trainable = False
    #adversarial = generator+discriminator
    adversarial = Model(inputs,discriminator(generator(inputs)),name=model_name)
    adversarial.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    adversarial.summary()

    #train discriminator and adversarial networks
    models = (generator,discriminator,adversarial)
    params = (batch_size,latent_size,train_steps,model_name)
    train(models,images,x_train,params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load generator h5 model with trained weights"
    parser.add_argument("-g", "--generator", help=help_)
    args = parser.parse_args()
    if args.generator:
        generator = load_model(args.generator)
        test_generator(generator)
    else:
        build_and_train_models()
