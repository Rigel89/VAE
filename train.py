# -*- coding: utf-8 -*-

#================================================================
#
#   File name   : train.py
#   Author      : Rigel89
#   Created date: 17/10/23
#   GitHub      : https://github.com/Rigel89/VAE
#   Description : training script
#
#================================================================

#%% Importing libraries

import tensorflow as tf

#This code must be here because has to be set before made other operations (quit, ugly solution!!)
print('SETTING UP GPUs')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            print('Setting up the GPUs done!')
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print('Setting up the GPUs Not done')

# Import system tools
import os
from shutil import rmtree

# Import modules
from VAE.config import *
from VAE.autoenconder import *
from VAE.losses import *
from VAE.config import *


#%% Main execution process

def main():
    # This must be a global variable
    global TRAIN_FROM_CHECKPOINT
    
    # Delete existing path and write a new log file
    print('Checking the existing path to the log:')
    if os.path.exists(os.path.join(MAIN_PATH,TRAIN_LOGDIR)):
        rmtree(TRAIN_LOGDIR)
        print('    Path existed and was deleted')
    else:
        print('    No existing path')

    print('Creating new log file')
    writer = tf.summary.create_file_writer(os.path.join(MAIN_PATH,TRAIN_LOGDIR))

    # Load the dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)
    
    train = tf.data.Dataset.from_tensor_slices((x_train,y_train))
    test = tf.data.Dataset.from_tensor_slices((x_test,y_test))

    #genetating the input data format
    train = train.batch(TRAIN_BATCH).shuffle(TRAIN_SHUFFLE).prefetch(TRAIN_PREFETCH)
    test = test.batch(TRAIN_BATCH).prefetch(TRAIN_PREFETCH)

    # Training variables for steps
    steps_per_epoch = len(train)
    global_steps = tf.Variable(0, trainable=False, dtype=tf.int64)
    warmup_steps = TRAIN_WARMUP_EPOCHS * steps_per_epoch
    total_steps = TRAIN_EPOCHS * steps_per_epoch
    print('Creating neuronal network')
    ##ME QUEDO AQUI



    yolo = Create_Yolov3(input_size=YOLO_INPUT_SIZE, channels=YOLO_CHANNELS,
                         anchors=ANCHORS,no_classes=NUMBER_OF_CLASSES)# training=True,

    print('Training from checkpoint: ' + str(TRAIN_FROM_CHECKPOINT))
    if TRAIN_FROM_CHECKPOINT:
        print("Trying to load weights from check points:")
        try:
            if os.path.exists("./checkpoints"):
                yolo.load_weights(f"./checkpoints/{TRAIN_MODEL_NAME}")
                print('    Succesfully load!')
            else:
                print('    There is not existing checkpoint path!')
        except ValueError:
            print("    Shapes are incompatible or there is not checkpoints")
            TRAIN_FROM_CHECKPOINT = False
    print('Setting up optimizer and iniciallicating training')
    optimizer = tf.keras.optimizers.Adam()

if __name__ == '__main__':
    main()