# -*- coding: utf-8 -*-

#================================================================
#
#   File name   : preprocessing_tools.py
#   Author      : Rigel89
#   Created date: 19/10/23
#   GitHub      : https://github.com/Rigel89/VAE
#   Description : tool script to preprocess data
#
#================================================================

#%% Importing libraries
import tensorflow as tf
import os

#%% Images tools

# Image normalization and cast format to float32 function, for dataset mapping
@tf.function
def img_norm(x):
    """Translate a image with 256 colors in uint8 to [0,1] in float32.
    Args:
    x -- image array with 256 colors

    Returns:
    x -- image array with [0,1] values in float32
    """
    x = tf.cast(x, dtype=tf.float32)/255.0
    return x

# Expand dimension for images with format (H,W) to (H,W,1), dataset mapping
@tf.function
def exp_dims(x):
    """Translate a image with 0 color channels to 1
    Args:
    x -- image array with dimension (H,W)

    Returns:
    x -- image array with dimension (H,W,1)
    """
    x = tf.expand_dims(x, axis=-1)
    return x

#%% One hot encoding tools

def one_hot_encoder(data):
    """Translate a data array to one hot encoding tensor
    Args:
    data -- list or array 1Dim, with names/numbers

    Returns:
    one_hot_encoded -- tensor with 0 and 1
    """
    if type(data) is list:
        length = len(data)
        encoded = tf.range(length+1)
        one_hot_encoded = tf.one_hot(indices=encoded, depth=length)
        return one_hot_encoded
    elif type(data) is np.ndarray:
        length = data.shape[0]
        encoded = tf.range(length+1)
        one_hot_encoded = tf.one_hot(indices=encoded, depth=length)
        return one_hot_encoded
    else:
        print('The format is incompatible use a list or an np array')

def create_name_file(names, one_hot_encoding, path='.\\', file_name='classes.names', overwrite=False):
    """Create a file wiht the name and hot encoding of each class
    Args:
    names -- list with names/numbers of the classes
    one_hot_encoding -- hot encoding resulting of the 'one_hot_encoder' function
    path -- path where the file with be created
    file_name -- name of the file
    overwrite --  if True delete the old file_name file and create a new one, other wise if the file with this name
                  already exist the file will not be created or edited by this function

    Returns:
    hot_encoding_dic -- dictionary with 'key'=name, 'value'=one_hot_encoding
    """
    if os.path.isfile(os.path.join(path,file_name)) and not overwrite:
        print('There is a file with this name! Skiping creation')
    else:
        f = open(os.path.join(path,file_name), "w")
        for n, name in enumerate(names):
            one_hot = one_hot_encoding[n].numpy().tolist()
            one_hot = ','.join(str(x) for x in one_hot)
            line = str(name) + ',' + one_hot + '\n'
            f.write(line)
        f.close()
        print('File created successfully')
    hot_encoding_dic = dict()
    for n, name in enumerate(names):
        hot_encoding_dic[str(name)] = one_hot_encoding[n]
    return hot_encoding_dic

@tf.function
def translate_to_hot_encoding(name, hot_encoding_dic):
    return hot_encoding_dic[tf.cast(name,tf.int32)]




