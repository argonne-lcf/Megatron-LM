#!/bin/bash --login

USER=$(whoami)

# SCRIPT_PATH="${BASH_SOURCE[0]}"
# while [ -L "$SCRIPT_PATH" ]; do
#   SCRIPT_DIR="$(cd -P "$(dirname "$SCRIPT_PATH")" >/dev/null 2>&1 && pwd)"
#   SCRIPT_PATH="$(readlink "$SCRIPT_PATH")"
#   [[ ${SCRIPT_PATH} != /* ]] && SCRIPT_PATH="${SCRIPT_DIR}/${SCRIPT_PATH}"
# done
# SCRIPT_PATH="$(readlink -f "$SCRIPT_PATH")"
# SCRIPT_DIR="$(cd -P "$(dirname -- "$SCRIPT_PATH")" >/dev/null 2>&1 && pwd)"

SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
  SOURCE=$(readlink "$SOURCE")
  [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
PARENT=$(dirname "$DIR")

# echo "------------------------"
# echo "SCRIPT_DIR=$SCRIPT_DIR"
# echo "SCRIPT_PATH=$SCRIPT_PATH"
echo "------------------------"
echo "SOURCE=$SOURCE"
echo "DIR=$DIR"
echo "------------------------"

sourceSetup() {
  SETUP_FILE="${DIR}/setup.sh"
  if [[ -f "$SETUP_FILE" ]]; then
    echo "source-ing ${SETUP_FILE}"
    # shellcheck source=./setup.sh
    source "$SETUP_FILE"
    setupMPI
  else
    echo "ERROR: UNABLE TO SOURCE ${SETUP_FILE}"
  fi
}

setupModel() {
  MODEL_SIZE="$1"
  MODEL_FILE="${DIR}/model.sh"
  if [[ -f "$MODEL_FILE" ]]; then
    echo "source-ing ${MODEL_FILE}"
    # shellcheck source=./model.sh
    source "$MODEL_FILE"
    getModel "${MODEL_SIZE}"
  else
    echo "ERROR: UNABLE TO SOURCE ${MODEL_FILE}"
  fi

}

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Model Architecture definition ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
sourceSetup
MODEL_SIZE="175B"
setupModel "${MODEL_SIZE}"

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Model Parallel / Pipeline Parallel ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# ----------
# Originals
# MPSIZE=8
# PPSIZE=16
# ----------
DDP_IMPL="local"   # FSDP | local | torch
USE_FLASH_ATTN=1  # 1 | 0
USE_ACTIVATION_CHECKPOINTING=1  # 1 | 0
SEQ_LEN=2048
MPSIZE=8
PPSIZE=16
MICRO_BATCH=2
ZERO_STAGE=0  # 0 | 1 | 2 | 3
GRADIENT_ACCUMULATION_STEPS=1

WORLD_SIZE="${NGPUS}"
export WORLD_SIZE="${WORLD_SIZE}"
GLOBAL_BATCH=$(( NGPUS * MICRO_BATCH * GRADIENT_ACCUMULATION_STEPS / (MPSIZE * PPSIZE) ))

# +-----------------------------------------------+
# | DEAL WITH SEQUENCE PARALLELISM / ENV VAR HERE |
export CUDA_DEVICE_MAX_CONNECTIONS=1
# +-----------------------------------------------+
#
# GB=NGPU*MB*GAS
# NGPUS=$((${NHOSTS}*${NGPU_PER_HOST}))
export SEQ_LEN="${SEQ_LEN}"

export DDP_IMPL="${DDP_IMPL}"
export USE_FLASH_ATTN="${USE_FLASH_ATTN}"
export ZERO_STAGE="${ZERO_STAGE}"
export USE_ACTIVATION_CHECKPOINTING="${USE_ACTIVATION_CHECKPOINTING}"

export NHOSTS="${NHOSTS}"
export NGPUS="${NGPUS}"
export MPSIZE="${MPSIZE}"
export PPSIZE="${PPSIZE}"
export MICRO_BATCH="${MICRO_BATCH}"
export GLOBAL_BATCH="${GLOBAL_BATCH}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}"

# ┏━━━━━━━━━━━━┓
# ┃ Data paths ┃
# ┗━━━━━━━━━━━━┛
# - [ ] [TODO] 1T with Pile data
# DATA_PATH="/lus/eagle/projects/datascience/venkatv/datasets/pile_bin/pile_text_document"
# ------------------------------------------------------------------------------------------------
# DATA_PATH="${PARENT}/dataset/BookCorpusDataset_text_document"
# DATA_PATH=/lus/grand/projects/datascience/foremans/genslm_megatron_preprocess/genslm-subsample_sequence_document/genslm-subsample_sequence_document
# DATA_PATH="/lus/eagle/projects/datascience/venkatv/datasets/pile_bin/pile_text_document"
DATA_PATH=/lus/grand/projects/datascience/vsastry/genslm_subsample_200k_sequence_document/genslm_subsample_200k_sequence_document
VOCAB_FILE="${PARENT}/dataset/gpt2-vocab.json"
MERGE_FILE="${PARENT}/dataset/gpt2-merges.txt"

# ┏━━━━━━━━━━━━━━━━━━━┓
# ┃ FILE I/O SETTINGS ┃
# ┗━━━━━━━━━━━━━━━━━━━┛
RUN_STR="gb${GLOBAL_BATCH}_mb${MICRO_BATCH}_gas${GRADIENT_ACCUMULATION_STEPS}"
RUN_STR="nl${NLAYERS}_hs${HIDDEN}_${RUN_STR}"
RUN_STR="mp${MPSIZE}_pp${PPSIZE}_${RUN_STR}"
RUN_STR="z${ZERO_STAGE}_seqlen${SEQ_LEN}_${RUN_STR}"
RUN_STR="${MODEL_SIZE}_${RUN_STR}"


# TODO:+------------------------------------------------------------+
# TODO:| Deal with this, following assertions from                  |
# TODO:| ../megatron/arguments.py                                   |
# TODO:| related to `tensor_model_parallel` and `sequence_parallel` |
# TODO:+------------------------------------------------------------+
#
# if [[ $MPSIZE == 1 ]]; then
#   SEQ_PARALLEL=0
# elif [[ $SEQ_PARALLEL == 1 ]]; then
#   ASYNC_TMP_ALLREDUCE=0
# fi


if [[ $USE_FLASH_ATTN == 1 ]] ; then
  RUN_STR="flashAttn_${RUN_STR}"
fi
if [[ $DDP_IMPL == 'FSDP' ]]; then
  RUN_STR="FSDP_${RUN_STR}"
fi
if [[ $USE_ACTIVATION_CHECKPOINTING == 1 ]] ;then
  RUN_STR="actCkpt_${RUN_STR}"
fi

RUN_STR="GPT3_${RUN_STR}"


OUTPUT_DIR="${PARENT}/outputs/${RUN_STR}"
CHECKPOINT_DIR="${PARENT}/checkpoints/$RUN_STR"
TENSORBOARD_DIR="${PARENT}/outputs/${RUN_STR}/tensorboard"

export MODEL_SIZE="$MODEL_SIZE"
export TENSORBOARD_DIR=$TENSORBOARD_DIR
export OUTPUT_DIR=$OUTPUT_DIR
mkdir -p "$OUTPUT_DIR/tensorboard/wandb"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$TENSORBOARD_DIR"
mkdir -p "$OUTPUT_DIR"
echo "OUTPUT TO: ${OUTPUT_DIR}"

# if [[ -z "${NVME_PATH}" ]]; then
#   echo "NVME_PATH: $NVME_PATH"
# else
#   if [[ $(hostname) == x* ]]; then
#     export NVME_PATH="/local/scratch/"
#   elif [[ $(hostname) == theta* ]]; then
#     export NVME_PATH="/raid/scratch/"
#   else
#     export NVME_PATH="/tmp/"
#   fi
# fi

# echo "NVME_PATH: ${NVME_PATH}"

# ┏━━━━━━━━━━━━━━━━━━┓
# ┃ DeepSpeed Config ┃
# ┗━━━━━━━━━━━━━━━━━━┛
# DS_CONFIG=${PARENT}/ds_config-gpt.json
# cat <<EOT > "$DS_CONFIG"
# {
#   "train_batch_size" : $GLOBAL_BATCH,
#   "train_micro_batch_size_per_gpu": $MICRO_BATCH,
#   "gradient_accumulation_steps": $GRADIENT_ACCUMULATION_STEPS,
#   "steps_per_print": 1,
#   "wall_clock_breakdown" : true,
#   "zero_optimization": {
#     "stage": $ZERO_STAGE,
#     "allgather_partitions": true,
#     "reduce_scatter": true,
#     "allgather_bucket_size": 5e8,
#     "overlap_comm": true,
#     "contiguous_gradients": true,
#     "offload_param": {
#       "device": "cpu",
#       "nvme_path": "/raid/scratch",
#       "pin_memory": true
#     }
#   },
#   "fp16": {
#     "enabled": true,
#     "initial_scale_power": 12
#   },
#   "flops_profiler": {
#     "enabled": true,
#     "profile_step": 1,
#     "module_depth": -1,
#     "top_modules": 3,
#     "detailed": true,
#     "output_file": null
#   },
#   "comms_logger": {
#     "enabled": true,
#     "verbose": false,
#     "prof_all": false,
#     "debug": false
#   },
#   "wandb": {
#     "enabled": true,
#     "project": "megatron-DS"
#   }
# }
# EOT

# ┏━━━━━━━━━━━━━━━━━━━━━┓
# ┃ DeepSpeed Arguments ┃
# ┗━━━━━━━━━━━━━━━━━━━━━┛
# if [[ "$DDP_IMPL" != "FSDP" ]] ; then
#   ds_args=""
#   ds_args=" --deepspeed ${ds_args}"
#   ds_args=" --deepspeed_mpi ${ds_args}"
#   ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
#   ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
#   if [[ "$PPSIZE" == 1 ]]; then
#     ds_args="--no-pipeline-parallel ${ds_args}"
#   else
#     ds_args=" --pipeline-model-parallel-size ${PPSIZE} ${ds_args}"
#   fi
#   # if [[ "$USE_ACTIVATION_CHECKPOINTING" == 1 ]]; then
#   #   ds_args=" --deepspeed-activation-checkpointing ${ds_args}"
#   # fi
# fi


# ┏━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ MEGATRON-LM SETTINGS ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━┛
gpt_args="\
  --seed ${RANDOM} \
  --DDP-impl ${DDP_IMPL} \
  --pipeline-model-parallel-size ${PPSIZE} \
  --tensor-model-parallel-size ${MPSIZE} \
  --num-layers ${NLAYERS} \
  --hidden-size ${HIDDEN} \
  --num-attention-heads ${ATEN_HEADS} \
  --micro-batch-size ${MICRO_BATCH} \
  --global-batch-size ${GLOBAL_BATCH} \
  --seq-length ${SEQ_LEN} \
  --max-position-embeddings ${SEQ_LEN} \
  --train-iters 5 \
  --lr-decay-iters 320000 \
  --data-path $DATA_PATH \
  --vocab-file $VOCAB_FILE \
  --merge-file $MERGE_FILE \
  --data-impl mmap \
  --split 949,50,1 \
  --distributed-backend nccl \
  --lr 0.00015 \
  --lr-decay-style cosine \
  --min-lr 1.0e-5 \
  --weight-decay 1e-2 \
  --clip-grad 1.0 \
  --lr-warmup-fraction .01 \
  --log-interval 1 \
  --save-interval 1000 \
  --eval-interval 1000 \
  --eval-iters 1 \
  --tensorboard-dir ${TENSORBOARD_DIR} \
  --log-timers-to-tensorboard \
  --num-workers 1 \
  --tensorboard-log-interval 1"

if [[ "$USE_ACTIVATION_CHECKPOINTING" == 1 ]]; then
  # gpt_args="\
  #   --checkpoint-activations \
  #   ${gpt_args}"
  gpt_args="\
    --recompute-activations \
    ${gpt_args}"
fi


if [[ "$DDP_IMPL" != "FSDP" ]] ; then
  gpt_args="${gpt_args} --fp16"
else
  gpt_args="${gpt_args} --bf16"
fi

if [[ "$USE_FLASH_ATTN" == 1 ]] ; then
  gpt_args="\
    --use-flash-attn \
    ${gpt_args}"
fi

export gpt_args="$gpt_args"

echo "megatron args: ${gpt_args}"
# echo "ds_config: ${DS_CONFIG}"
