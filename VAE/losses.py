# -*- coding: utf-8 -*-

#================================================================
#
#   File name   : losses.py
#   Author      : Rigel89
#   Created date: 17/10/23
#   GitHub      : https://github.com/Rigel89/VAE
#   Description : losses for VAE autoencoder architecture script
#
#================================================================

#%% Importing libraries
import tensorflow as tf

# Loss function

def kl_reconstruction_loss(mu, sigma):
    """ Computes the Kullback-Leibler Divergence (KLD)
    Args:
    mu -- mean
    sigma -- standard deviation

    Returns:
    KLD loss
    """
    kl_loss = 1 + sigma - tf.square(mu) - tf.exp(sigma)
    kl_loss = tf.reduce_mean(kl_loss)*-0.5

    return kl_loss

