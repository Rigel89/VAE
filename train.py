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
from numpy import pi as PI

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
from VAE.preprocessing_tools import *
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

    #Labels names
    names = ['One','Two','Three','Four','Five','Six','Seven','Eigth','Nine']
    one_hot_names = one_hot_encoder(names)
    print('Creating ')
    hot_encoding_dic = create_name_file(names, one_hot_names, path='.\\', file_name='classes.names', overwrite=False)

    # Preprocessing dataset images and labels
    train = train.map(lambda x, y: (exp_dims(img_norm(x)), translate_to_hot_encoding(y, one_hot_names)),
                      num_parallel_calls=tf.data.AUTOTUNE)

    test = test.map(lambda x, y: (exp_dims(img_norm(x)), translate_to_hot_encoding(y, one_hot_names)),
                    num_parallel_calls=tf.data.AUTOTUNE)

    #genetating the input data format
    train = train.batch(TRAIN_BATCH).shuffle(TRAIN_SHUFFLE).prefetch(TRAIN_PREFETCH)
    test = test.batch(TRAIN_BATCH).prefetch(TRAIN_PREFETCH)

    # Training variables for steps
    steps_per_epoch = len(train)
    global_steps = tf.Variable(0, trainable=False, dtype=tf.int64)
    warmup_steps = TRAIN_WARMUP_EPOCHS * steps_per_epoch
    total_steps = TRAIN_EPOCHS * steps_per_epoch
    print('Creating neuronal network')
    

    VAE, mnist_gen_model, encoder_model  = VAE_autoencoder(latent_dim_shape=LATENT_DIMENSION, image_shape=INPUT_IMAGE_SIZE)

    print('Training from checkpoint: ' + str(TRAIN_FROM_CHECKPOINT))
    if TRAIN_FROM_CHECKPOINT:
        print("Trying to load weights from check points:")
        try:
            if os.path.exists("./checkpoints"):
                VAE.load_weights(f"./checkpoints/{TRAIN_MODEL_NAME}")
                print('    Succesfully load!')
            else:
                print('    There is not existing checkpoint path!')
        except ValueError:
            print("    Shapes are incompatible or there is not checkpoints")
            TRAIN_FROM_CHECKPOINT = False

    print('Setting up optimizer and iniciallicating training')
    optimizer = tf.keras.optimizers.Adam()
    loss_metric = tf.keras.metrics.Mean()
    bce_loss = tf.keras.losses.BinaryCrossentropy()

    image_pixels = float(INPUT_IMAGE_SIZE[0]*INPUT_IMAGE_SIZE[1]*INPUT_IMAGE_SIZE[2])

    def train_step(image_data, target):
        with tf.GradientTape() as tape:
            reconstruted_image = VAE(image_data, training=True) # There is BatchNormalization layers, so training=True to train the parameters
            mu, sigma, _ =  encoder_model(image_data, training=True)

            # compute reconstruction loss
            KL_loss = bce = 0
            KL_loss = kl_reconstruction_loss(mu, sigma)
            flattened_inputs = tf.reshape(image_data, shape=[-1])
            flattened_outputs = tf.reshape(reconstruted_image, shape=[-1])
            bce = bce_loss(flattened_inputs, flattened_outputs) * image_pixels
            total_loss = KL_loss + bce
            loss_mean = loss_metric(total_loss)

            # optimizing process
            gradients = tape.gradient(total_loss, VAE.trainable_variables)
            optimizer.apply_gradients(zip(gradients, VAE.trainable_variables))

            # update learning rate
            # about warmup: https://arxiv.org/pdf/1812.01187.pdf&usg=ALkJrhglKOPDjNt6SHGbphTHyMcT0cuMJg
            
            if global_steps < warmup_steps:# and not TRAIN_TRANSFER:
                lr = global_steps / warmup_steps * TRAIN_LR_INIT
            else:
                lr = TRAIN_LR_END + 0.5 * (TRAIN_LR_INIT - TRAIN_LR_END)*(
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * PI)))
            optimizer.lr.assign(lr.numpy())

            # writing summary data
            with writer.as_default():
                tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("loss/KL_loss", KL_loss, step=global_steps)
                tf.summary.scalar("loss/BCE_loss", bce, step=global_steps)
                tf.summary.scalar("loss/mean", loss_mean, step=global_steps)
            writer.flush()
            global_steps.assign_add(1)
        return global_steps.numpy(), optimizer.lr.numpy(), KL_loss.numpy(), bce.numpy(), loss_mean.numpy()

    validate_writer = tf.summary.create_file_writer(TRAIN_LOGDIR)
    def validate_step(image_data, target):
        reconstruted_image = VAE(image_data, training=True) # There is BatchNormalization layers, so training=True to train the parameters
        mu, sigma, _ =  encoder_model(image_data, training=True)

        # Losses process
        KL_loss = bce = 0
        KL_loss = kl_reconstruction_loss(mu, sigma)
        flattened_inputs = tf.reshape(image_data, shape=[-1])
        flattened_outputs = tf.reshape(reconstruted_image, shape=[-1])
        bce = bce_loss(flattened_inputs, flattened_outputs) * image_pixels
        total_loss = KL_loss + bce
        loss_mean = loss_metric(total_loss)
            
        return KL_loss.numpy(), bce.numpy(), loss_mean.numpy(), total_loss.numpy()

    print('Starting training process:')
    best_val_loss = 100000 # should be large at start
    for epoch in range(TRAIN_EPOCHS):
        for train_vars in train:
            
            image_data = train_vars[0]
            target = train_vars[1]
            results = train_step(image_data, target)
            cur_step = results[0]%steps_per_epoch
            print("epoch:{:2.0f} step:{:5.0f}/{}, lr:{:.6f}, KL_loss:{:4.4f}, BCE_loss:{:4.4f}, Mean_loss:{:4.4f}".format(epoch, cur_step, steps_per_epoch, results[1], results[2],
                          results[3], results[4]))

        if len(test) == 0:
            print("configure TEST options to validate model")
            #yolo.save_weights(os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME))
            #continue
        else:
            count, KL_val, BCE_val, Mean_val, total_val = 0, 0, 0, 0, 0
            for test_vars in test:
                image_data = test_vars[0]
                target = test_vars[1]
                results = validate_step(image_data, target)
                count += 1
                KL_val += results[0]
                BCE_val += results[1]
                Mean_val += results[2]
                total_val += results[3]
            # writing validate summary data
            with validate_writer.as_default():
                tf.summary.scalar("validate_loss/total_val", total_val/count, step=epoch)
                tf.summary.scalar("validate_loss/KL_val", KL_val/count, step=epoch)
                tf.summary.scalar("validate_loss/BCE_val", BCE_val/count, step=epoch)
                tf.summary.scalar("validate_loss/Mean_val", Mean_val/count, step=epoch)
            validate_writer.flush()
            
            print("\nValidation step-> KL_val_loss:{:7.2f}, BCE_val_loss:{:7.2f}, Mean_val_loss:{:7.2f}\n"
                  .format(KL_val/count, BCE_val/count, Mean_val/count))

        if TRAIN_SAVE_CHECKPOINT and not TRAIN_SAVE_BEST_ONLY:
            save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME+"_val_loss_{:7.2f}".format(total_val/count))
            VAE.save_weights(save_directory)
            print('\nWeights saved every epoch\n')
        if TRAIN_SAVE_BEST_ONLY and best_val_loss>total_val/count:
            save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME)
            VAE.save_weights(save_directory)
            best_val_loss = total_val/count
            print('\nThe weights are being saved this epoch!\n')

if __name__ == '__main__':
    main()