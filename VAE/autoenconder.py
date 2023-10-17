# -*- coding: utf-8 -*-

#================================================================
#
#   File name   : autoenconder.py
#   Author      : Rigel89
#   Created date: 14/10/23
#   GitHub      : https://github.com/Rigel89/VAE
#   Description : VAE autoencoder architecture script
#
#================================================================

#%% Importing libraries

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Flatten, Reshape, Dense, Conv2DTranspose

#%% Creating encoder architecture

class encoder(Model):
    """Defines the encoder's layers.
    Args:
    inputs -- batch from the dataset
    latent_dim -- dimensionality of the latent space

    Returns:
    mu -- learned mean
    sigma -- learned standard deviation
    batch_2.shape -- shape of the features before flattening
    """
    def __init__(self, latent_dim=28, name='encoder'):
        super().__init__()
        self._name=name
        # Convolutions of the encoder
        self.conv_1 = Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation='relu', name='encode_conv1')
        self.bn_1 = BatchNormalization()
        self.conv_2 = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu', name='encode_conv2')
        self.bn_2 = BatchNormalization()
        # Translating the information to an array
        self.flat = Flatten(name='encode_flatten')
        self.den_1 = Dense(20, activation='relu', name='encode_dense')
        self.bn_3 = BatchNormalization()
        # Generating the latent space
        self.mu = Dense(latent_dim, name='latent_mu')
        self.sigma = Dense(latent_dim, name='latent_sigma')
    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.flat(x)
        x = self.den_1(x)
        x = self.bn_3(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        return  mu, sigma, self.bn_2.shape


#%% Creating bottle neck architecture

class sampling(Model):
    """Generates a random sample and combines with the encoder output
    
    Args:
      inputs -- output tensor from the encoder

    Returns:
      `inputs` tensors combined with a random sample
    """
    def __init__(self, name='bottle_neck'):
        super().__init__()
        self._name = name
    def call(self, inputs):
        # unpack the output of the encoder
        mu, sigma = inputs
        # get the size and dimensions of the batch
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]

        # generate a random tensor
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        # combine the inputs and noise
        return mu + tf.exp(0.5 * sigma) * epsilon


#%% Creating decoder architecture

class decoder(Model):
    """
    Defines the decoder layers.
    Args:
      inputs -- output of the bottleneck 
      conv_shape -- shape of the features before flattening

    Returns:
      tensor containing the decoded output
    """
    def __init__(self, conv_shape, name='decoder'):
        super().__init__()
        self._name=name
        self.units = conv_shape[1] * conv_shape[2] * conv_shape[3]
        # Array to conv
        self.den_1 = Dense(self.units, activation = 'relu', name="decode_dense1")
        self.bn_1 = BatchNormalization()
        self.reshape = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]), name="decode_reshape")
        # Convolutions of the decoder
        self.conv_1 = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu', name="decode_conv2d_2")
        self.bn_2 = BatchNormalization()
        self.conv_2 = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu', name="decode_conv2d_3")
        self.bn_3 = BatchNormalization()
        self.conv_3 = Conv2DTranspose(filters=1, kernel_size=3,strides=1, padding='same', activation='sigmoid', name="decode_final" )
        # Translating the information to an array
        self.flat = Flatten(name='encode_flatten')
        self.den_1 = Dense(20, activation='relu', name='encode_dense')
        self.bn_3 = BatchNormalization()
        # Generating the latent space
        self.mu = Dense(latent_dim, name='latent_mu')
        self.sigma = Dense(latent_dim, name='latent_sigma')
    def call(self, inputs):
        #Decoding linear array
        x = self.den_1(inputs)
        x = self.bn_1(x)
        
        # reshape output using the conv_shape dimensions
        x = self.reshape(x)

        # upsample the features back to the original dimensions
        x = self.conv_1(x)
        x = self.bn_2(x)
        x = self.conv_2(x)
        x = self.bn_3(x)
        x = self.conv_3(x)
        return x

# VAE model


