# -*- coding: utf-8 -*-

#================================================================
#
#   File name   : config.py
#   Author      : Rigel89
#   Created date: 14/10/23
#   GitHub      : https://github.com/Rigel89/VAE
#   Description : training configuration and global variable script
#
#================================================================

#%% Variables

# Global parameters

from tkinter import N


NN_NAME                     = 'VAE_v1'
KEEP_LOG_EVENTS             = False

# NN parameters
LATENT_DIMENSION            = 28

# Dataset parameters

INPUT_IMAGE_SIZE            = (28,28,1)
CLASSES_NAMES               = ['Zero','One','Two','Three','Four','Five','Six','Seven','Eigth','Nine']
NUMBER_OF_CLASSES           = len(CLASSES_NAMES)

#Training parameters

MAIN_PATH                   = '.\\'
TRAIN_LOGDIR                = "log"
#DATASET_DIR                 = 'MNIST_dataset'
#TRAIN_DIR                   = 'train'
#TEST_DIR                    = 'test'
TRAIN_CHECKPOINTS_FOLDER    = "checkpoints"
TRAIN_MODEL_NAME            = f"{NN_NAME}_custom"
TRAIN_FROM_CHECKPOINT       = False
TRAIN_SAVE_CHECKPOINT       = False
TRAIN_SAVE_BEST_ONLY        = True
TRAIN_WARMUP_EPOCHS         = 1
TRAIN_EPOCHS                = 20
TRAIN_BATCH                 = 60
TRAIN_PREFETCH              = -1
TRAIN_SHUFFLE               = 1000
TRAIN_LR_INIT               = 0.5e-3
TRAIN_LR_END                = 1e-4
LAMBDA_LK                   = 20.0
# LR_VARIATION_EPOCH          = int(0.9*TRAIN_EPOCHS)
# REGULARIZATION_START_EPOCH  = 1
# REGULARIZATION_END_EPOCH    = 10
# REGULARIZATION_MAX_VALUE    = 0.1