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

NN_NAME                     = 'VAE_v1'
KEEP_LOG_EVENTS             = False

# NN parameters
# NUMBER_OF_CLASSES           = 10
# LAMBDA_OBJ                  = 5.0
# LAMBDA_NOOBJ                = 0.5

# Dataset parameters

# INPUT_IMAGE_SIZE            = (105,105,3)
# NUM_SAMPLES                 = 8385+8138
# TRAIN_SIZE                  = int(0.7*NUM_SAMPLES)
# TEST_SIZE                   = int(0.2*NUM_SAMPLES)
# VAL_SIZE                    = int(NUM_SAMPLES-TRAIN_SIZE-TEST_SIZE)

# Dataset creation configuration

# PREP_PATH                   = '.\\dataset\\preprocess'
# FORCE_PREP_DATASET          = False
# BATCH_DATASET               = 5
# FETCH_DATASET               = 6
# SHUFFLE_DATASET             = 600
# NUMBER_OF_TRAINING_IMAGES   = 4000
# NUMBER_OF_TEST_IMAGES       = 500
# MIN_SCALE                   = 0.4
# MAX_SCALE                   = 14
# ROTATION_ANG                = 30

#Training parameters

MAIN_PATH                   = '.\\'
TRAIN_LOGDIR                = "log"
#DATASET_DIR                 = 'MNIST_dataset'
#TRAIN_DIR                   = 'train'
#TEST_DIR                    = 'test'
# TRAIN_CHECKPOINTS_FOLDER    = "checkpoints"
# TRAIN_MODEL_NAME            = f"{NN_NAME}_custom"
# TRAIN_FROM_CHECKPOINT       = False
# TRAIN_SAVE_CHECKPOINT       = False
# TRAIN_SAVE_BEST_ONLY        = True
TRAIN_WARMUP_EPOCHS         = 1
TRAIN_EPOCHS                = 150
TRAIN_BATCH                 = 120
TRAIN_PREFETCH              = -1
TRAIN_SHUFFLE               = 960
# TRAIN_LR_INIT               = 0.8e-3
# TRAIN_LR_END                = 1e-5
# LR_VARIATION_EPOCH          = int(0.9*TRAIN_EPOCHS)
# REGULARIZATION_START_EPOCH  = 1
# REGULARIZATION_END_EPOCH    = 10
# REGULARIZATION_MAX_VALUE    = 0.1