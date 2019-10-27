#!/usr/bin/env bash

export PYTHONPATH=`pwd`:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

WORK_DIR=$(pwd)
ATTACK_DIR="${WORK_DIR}/src"
echo ${ATTACK_DIR}


###########1
if [ ! -d "drive_rotate_11_20w_unet/" ];then
mkdir drive_rotate_11_20w_unet
else
echo "Dir already existing"
fi

nohup python -u "${ATTACK_DIR}"/train_rotation_new_alone.py \
   --configfile=config_1.txt > drive_rotate_11_20w_unet/training.nohup


python "${ATTACK_DIR}"/predict_patch_2.py --testdir drive_rotate_11_20w_unet --start 2





