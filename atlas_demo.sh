#!/bin/bash
#SBATCH --mail-user=tirupativenkatasrisairama.penmatsa@sjsu.edu
#SBATCH --mail-user=/dev/null
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=gpuTest_016037047
#SBATCH --output=Batchsize_%j.out
#SBATCH --error=Batchsize_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00     
##SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu   
#SBATCH --mem=40G

# on coe-hpc1 cluster load
# module load python3/3.8.8
#
# on coe-hpc2 cluster load:


module load python-3.10.8-gcc-11.2.0-c5b5yhp slurm
export CUDA_VISIBLE_DEVICES=0,1,2,3
export http_proxy=http://172.16.1.2:3128; export https_proxy=http://172.16.1.2:3128


cd /home/016037047/atlas

DATA_DIR=/home/016037047/atlas/atlas_data
port=$(shuf -i 15000-16000 -n 1)
TRAIN_FILE="${DATA_DIR}/nq_data/team2-split25.jsonl"
EVAL_FILES="${DATA_DIR}/nq_data/evaluation2split1.jsonl"
SAVE_DIR=${DATA_DIR}/experiments/
EXPERIMENT_NAME=alt_base
PASSAGE_FILE=unified-passage-set-v1.jsonl
TRAIN_STEPS=100
SIZE=base
TARGET_MAXLEN=16
LOG_FREQUENCY=1
PRECISION=bf16
TEMPARATURE_GOLD=1
DROPOUT=0.01
WEIGHT_DECAY=0.001
LR=4e-5
LR_RETRIEVER=4e-5
TEXT_MAXLENGTH=512
PER_GPU_BATCH_SIZE=1
N_CONTEXT=10
RETRIEVER_N_CONTEXT=10
WARMUP_STEPS=50
TOTAL_STEPS=50
EVAL_FREQ=10
SAVE_FREQ=50



# submit your code to Slurm 
python3 /home/016037047/atlas/train.py --shuffle  --train_retriever  --gold_score_mode pdist   --use_gradient_checkpoint_reader --use_gradient_checkpoint_retriever  --precision ${PRECISION}   --shard_optim --shard_grads   --temperature_gold ${TEMPARATURE_GOLD}   --refresh_index -1   --query_side_retriever_training  --target_maxlength ${TARGET_MAXLEN}   --reader_model_type google/t5-${SIZE}-lm-adapt --dropout ${DROPOUT} --weight_decay ${WEIGHT_DECAY} --lr ${LR} --lr_retriever ${LR_RETRIEVER} --scheduler linear   --text_maxlength ${TEXT_MAXLENGTH}   --model_path "/home/016037047/atlas/atlas_data/models/atlas/${SIZE}/"  --train_data ${TRAIN_FILE}   --eval_data ${EVAL_FILES}   --per_gpu_batch_size ${PER_GPU_BATCH_SIZE}  --n_context ${N_CONTEXT}   --retriever_n_context ${RETRIEVER_N_CONTEXT}   --name ${EXPERIMENT_NAME}   --checkpoint_dir ${SAVE_DIR}   --eval_freq ${EVAL_FREQ}   --log_freq ${LOG_FREQUENCY}   --total_steps ${TRAIN_STEPS}   --warmup_steps ${WARMUP_STEPS}  --save_freq ${TRAIN_STEPS}   --main_port $port   --write_results --task qa   --index_mode flat   --passages "/home/016037047/atlas/atlas_data/corpora/wiki/enwiki-dec2018/${PASSAGE_FILE}"  --save_index_path ${SAVE_DIR}/${EXPERIMENT_NAME}/saved_index