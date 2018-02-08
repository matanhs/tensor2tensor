#!/bin/bash
##This script allows new users to run the Transformer big model on wmt english to deutsch dataset,
#examin the results on tensorboard and benchmark training performance.

##ARGS-
#$1 arg is the desired base path for the script to write auxilary files (optional)
#$2 arg is number of gpus to use for training. (optional)
#$3 arg is the wanted batch size. (optional)
#$4 arg is the number of training steps between reports. (optional)
#$5 arg is used to specify the data-set location, if not used output variable is set to <auxilary dir>/data. (optional)
set -e

#script home directory
ORIGIN="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
#where to save logs and checkpoints, if no dataset dir is provided the data will be written here
BASE_DIR=${1:-$ORIGIN"/tmp/Transformer"}
#how many gpus to use for training/inferencing
NUM_GPU=${2:-1}
#specify batch size - stands for how many tokens per gpu (source + target)
BATCH_SIZE=${3:-4096}
#report freq
SUMMARY_FREQ=${4:-1000}
#data set directory
DATA_DIR=${5:-$BASE_DIR/data}

echo 'base dir is '$BASE_DIR
echo 'data dir is '$DATA_DIR

#create base directory
if [ ! -d "$BASE_DIR"  ]; then
  mkdir -p $BASE_DIR
fi

#setup the data set in the data directory
if [ ! -d "$DATA_DIR"  ]; then
  mkdir -p $DATA_DIR
  t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$BASE_DIR/datagen_tmp \
  --problem=translate_ende_wmt32k

  else
  echo 'designted data folder '"$DATA_DIR"' already exists, assuming data is available'
fi

LOG_DIR="$BASE_DIR""/Transformer_""$NUM_GPU""_GPUs""_batchsize_""$BATCH_SIZE"

#clear previous data
if [ -d "$LOG_DIR"  ]; then
  rm -rf $LOG_DIR

fi

mkdir -p $LOG_DIR/model-eval
echo 'logs can be found at '"$LOG_DIR"
echo 'launching tensor-board'
tensorboard --logdir $LOG_DIR &
echo 'start training Transfomer with '"$NUM_GPU"' GPUs, and batch size of '"$BATCH_SIZE"''
#uncomment one for profiling/tracing
#nvprof \
t2t-trainer --data_dir=$DATA_DIR --problems=translate_ende_wmt32k --model=transformer \
  --hparams_set=transformer_big --worker_gpu=$NUM_GPU --output_dir=$LOG_DIR \
  --hparams='batch_size='"$BATCH_SIZE" --local_eval_frequency=$SUMMARY_FREQ #--dbgprofile

#tracing can be viewed by opening chrome://tracing and loading the timeline from the log folder

