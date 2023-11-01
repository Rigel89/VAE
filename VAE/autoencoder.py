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
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Flatten, Reshape, Dense, Conv2DTranspose, Concatenate

#%% Creating encoder architecture

class Encoder(Model):
    """Defines the encoder's layers.
    Args:
    inputs -- batch from the dataset
    latent_dim -- dimensionality of the latent space

    Returns:
    mu -- learned mean
    sigma -- learned standard deviation
    batch_2.shape -- shape of the features before flattening
    """
    def __init__(self, input_shape=(28,28,1), latent_dim=28, name='encoder'):
        super().__init__()
        self._name=name
        # Convolutions of the encoder
        #self.input_layer = Input(shape=(input_shape))
        self.conv_1 = Conv2D(input_shape=input_shape,filters=32, kernel_size=3, strides=2, padding='same', activation='relu', name='encode_conv1')
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
        #x = self.input_layer(inputs)
        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        conv_shape = tf.shape(x)
        x = self.flat(x)
        x = self.den_1(x)
        x = self.bn_3(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        return  mu, sigma, conv_shape

class Encoder_CVAE(Model):
    """Defines the encoder's layers.
    Args:
    inputs_im -- images batch from the dataset
    inputs_c -- classes batch from the dataset
    latent_dim -- dimensionality of the latent space

    Returns:
    mu -- learned mean
    sigma -- learned standard deviation
    batch_2.shape -- shape of the features before flattening
    """
    def __init__(self, input_shape=(28,28,1), number_of_classes=10, latent_dim=28, name='encoder'):
        super().__init__()
        self._name=name
        # Convolutions of the encoder
        #self.input_layer = Input(shape=(input_shape))
        self.conv_1 = Conv2D(input_shape=input_shape,filters=32, kernel_size=3, strides=2, padding='same', activation='relu', name='encode_conv1')
        self.bn_1 = BatchNormalization()
        self.conv_2 = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu', name='encode_conv2')
        self.bn_2 = BatchNormalization()
        # Translating the information to an array
        self.flat = Flatten(name='encode_flatten')
        self.den_1 = Dense(number_of_classes, activation='sigmoid', name='encode_dense')
        self.den_2 = Dense(number_of_classes, activation='sigmoid', name='encode_dense')
        self.concat = Concatenate(axis=-1)
        self.den_3 = Dense(20, activation='relu', name='encode_dense')
        self.bn_3 = BatchNormalization()
        # Generating the latent space
        self.mu = Dense(latent_dim, name='latent_mu')
        self.sigma = Dense(latent_dim, name='latent_sigma')
    def call(self, inputs):
        #x = self.input_layer(inputs)
        img, classes = inputs
        x = self.conv_1(img)
        x = self.bn_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        conv_shape = tf.shape(x)
        x = self.flat(x)
        c = self.den_1(classes)
        c = self.den_2(c)
        x = self.concat([x,c])
        x = self.den_3(x)
        x = self.bn_3(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        return  mu, sigma, conv_shape

#%% Creating bottle neck architecture

class Sampling(Model):
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

#%% Creating the second head of the NN to decode within clasification

class classification_bottle_neck(Model):
    def __init__(self, number_of_classes=10, name='classification_bottle_neck'):
        super().__init__()
        self.number_of_classes = number_of_classes
        self._name = name
        self.dens_1 = Dense(number_of_classes, activation='sigmoid')
        self.dens_2 = Dense(10, activation='sigmoid')
        self.dens_3 = Dense(10, activation='sigmoid')
    def call(self, inputs):
        X = self.dens_1(inputs)
        X = self.dens_2(X)
        X = self.dens_3(X)
        return X

#%% Creating decoder architecture

class Decoder(Model):
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
        units = conv_shape[1] * conv_shape[2] * conv_shape[3]
        # Array to conv
        self.den_1 = Dense(units, activation = 'relu', name="decode_dense1")
        self.bnd_1 = BatchNormalization()
        self.den_2 = Dense(units, activation = 'relu', name="decode_dense1")
        self.bnd_2 = BatchNormalization()
        self.reshape = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]), name="decode_reshape")
        # Convolutions of the decoder
        self.conv_1 = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu', name="decode_conv2d_2")
        self.bn_2 = BatchNormalization()
        self.conv_2 = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu', name="decode_conv2d_3")
        self.bn_3 = BatchNormalization()
        self.conv_3 = Conv2DTranspose(filters=1, kernel_size=3,strides=1, padding='same', activation='sigmoid', name="decode_final" )

    def call(self, inputs):
        #Decoding linear array
        x = self.den_1(inputs)
        x = self.bnd_1(x)
        x = self.den_2(x)
        x = self.bnd_2(x)
        
        # reshape output using the conv_shape dimensions
        x = self.reshape(x)

        # upsample the features back to the original dimensions
        x = self.conv_1(x)
        x = self.bn_2(x)
        x = self.conv_2(x)
        x = self.bn_3(x)
        x = self.conv_3(x)
        return x

#%% VAE model

# Creating a MNIST images generator using VAE
def MNIST_generator(latent_dim_shape=28, conv_dim=(0,0,0,0)):
    # Initializating input layers
    mu = tf.keras.layers.Input(shape=(latent_dim_shape))
    sigma = tf.keras.layers.Input(shape=(latent_dim_shape))
    # Loading partial models
    model_sampler = Sampling()
    model_decoder = Decoder(conv_dim)
    # Calling the models
    sampler_output = model_sampler((mu, sigma))
    ouput_img =  model_decoder(sampler_output)
    # Creating the concatenated model
    model = tf.keras.models.Model(inputs=[mu,sigma], outputs=[ouput_img],name='MNIST_generator')
    return model

# Creating the VAE autoenconder full model
def VAE_autoencoder(latent_dim_shape=28, image_shape=(28,28,1)):
    # Input image layer
    img = Input(shape=image_shape, name="input_image")
    # Creating a partial model for the encoder
    model_encoder = Encoder(latent_dim=latent_dim_shape)
    # Calling the model
    mu, sigma, conv_dim = model_encoder(img)
    conv_dim= conv_dim._inferred_value
    model_MNIST_generator = MNIST_generator(latent_dim_shape=latent_dim_shape,conv_dim=conv_dim)
    ouput_MNIST_generator = model_MNIST_generator((mu, sigma))
    VAE = tf.keras.models.Model(inputs=[img], outputs=[ouput_MNIST_generator])
    return VAE, model_MNIST_generator, model_encoder

# Create a mu and sigma encoder
def mu_sig(latent_dim_shape=28, number_of_clases=10):
    inputs = Input(shape=(number_of_clases))
    mu = Sequential([
        Dense(latent_dim_shape, activation='sigmoid'),
        Dense(latent_dim_shape, activation='sigmoid'),
    ])
    sigma = Sequential([
        Dense(latent_dim_shape, activation='sigmoid'),
        Dense(latent_dim_shape, activation='sigmoid'),
    ])
    ouput_mu = mu(inputs)
    ouput_sigma = sigma(inputs)
    model = Model(inputs=[inputs],outputs=[ouput_mu,ouput_sigma])
    return model

#%% New VAE model with a atributes inputs

def class_generator(latent_dim_shape=28, number_of_classes=10, conv_dim=(0,0,0,0)):
    # Initializating input layers
    class_inputs = Input(shape=(number_of_classes))
    random_sample = Input(shape=(latent_dim_shape))
    # Loading partial models
    model_classifier = classification_bottle_neck(number_of_classes=number_of_classes)
    model_decoder = Decoder(conv_dim)
    # Calling the models
    classifier_output = model_classifier(class_inputs)
    concatenate_input = Concatenate(axis=-1)([random_sample, classifier_output])
    ouput_img =  model_decoder(concatenate_input)
    # Creating the concatenated model
    model = tf.keras.models.Model(inputs=[random_sample, class_inputs], outputs=[ouput_img],name='MNIST_generator')
    return model

# Creating the VAE autoenconder full model
def VAE_classes(latent_dim_shape=28, number_of_classes=10, image_shape=(28,28,1)):
    # Input image layer
    img = Input(shape=image_shape, name="input_image")
    class_inputs = Input(shape=(number_of_classes))
    # Creating a partial model for the encoder
    model_encoder = Encoder_CVAE(input_shape=image_shape, latent_dim=latent_dim_shape, number_of_classes=number_of_classes)
    model_sampler = Sampling()

    # Calling the model
    mu, sigma, conv_dim = model_encoder((img, class_inputs))
    output_sampler = model_sampler((mu, sigma))
    conv_dim= conv_dim._inferred_value
    model_class_generator = class_generator(latent_dim_shape=latent_dim_shape,
                                            number_of_classes=number_of_classes,
                                            conv_dim=conv_dim)
    ouput_img =  model_class_generator([output_sampler, class_inputs])
    VAE = tf.keras.models.Model(inputs=[img,class_inputs], outputs=[ouput_img])
    return VAE, model_class_generator, model_encoder
