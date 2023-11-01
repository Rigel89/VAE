# -*- coding: utf-8 -*-

#================================================================
#
#   File name   : postprocessing_tools.py
#   Author      : Rigel89
#   Created date: 20/10/23
#   GitHub      : https://github.com/Rigel89/VAE
#   Description : tool script to postprocess data
#
#================================================================

#%% Importing libraries
import tensorflow as tf
import os
import matplotlib.pyplot as plt

#%% Visualizating tools
@tf.function
def show_images(output_tensor,display_shape=(5,5),figure_size=(5,5), show=False):
    if output_tensor.shape[0] > display_shape[0]*display_shape[1]:
        fig, ax = plt.subplots(display_shape[0],display_shape[1], figsize =figure_size)
        for row in range(display_shape[0]):
            for col in range(display_shape[1]):
                imag_number = row*display_shape[1] + col
                ax[row, col].imshow(output_tensor[imag_number], cmap='gray')
                #ax.set_title(str(imag_number))
        if show:
            plt.show()
        return fig
    else:
        print('Try to increase the batch size (the number of images),\n show_images>tensor_images')