#!/bin/bash

# --- 스크립트 설정 ---
# set -e: 명령어가 실패하면 즉시 스크립트를 중단합니다.
# set -x: 실행되는 명령어를 터미널에 출력하여 디버깅을 돕습니다.
set -e
set -x
# export CUDA_VISIBLE_DEVICES=0,1,2,3 # 필요시 주석 해제하여 특정 GPU만 사용

# --- 사용자 설정 변수 ---
# 이 부분의 값들을 필요에 맞게 수정하여 사용하세요. (하이퍼파라미터 튜닝 시 이 부분을 변경)
export CUDA_VISIBLE_DEVICES=0,1,2
# 사용할 GPU 개수 (시스템에 맞게 설정)
N_GPUS=3

# 데이터셋을 캐시할 경로 (절대 경로 권장)
DATA_PATH="$HOME/.cache/huggingface/datasets"

# TensorBoard 로그 및 모델 체크포인트를 저장할 기본 경로
OUTPUT_DIR="./training_output"

# Python 스크립트 파일 이름
SCRIPT_NAME="vit.py"

# 학습 기본 하이퍼파라미터
EPOCHS=90
BATCH_SIZE_PER_GPU=256 # GPU 메모리에 맞춰 조절
WORKERS=4              # 데이터 로딩에 사용할 CPU 워커 수

# [변경] 옵티마이저 및 스케줄러 하이퍼파라미터 (튜닝 대상)
BASE_LR=0.0013
WARMUP_STEPS=7510
WEIGHT_DECAY=0.0005      # [추가] Weight Decay
BETA1=0.95             # [추가] Beta1 (Momentum)

# [추가] 데이터 증강 하이퍼파라미터 (Algoperf ViT 기본값)
MIXUP=0.2            # [추가] Mixup alpha
LABEL_SMOOTHING=0.1   # [추가] Label Smoothing

# --- 실행 설정 ---
# 로그 및 체크포인트 저장을 위한 디렉토리 생성
# [변경] RUN_NAME에 주요 하이퍼파라미터 포함하여 실험 식별 용이하게 함
RUN_NAME="vit_shampoo_LR${BASE_LR}_WD${WEIGHT_DECAY}_B1${BETA1}_$(date +%Y%m%d_%H%M%S)"
LOG_PATH="$OUTPUT_DIR/$RUN_NAME/logs"
# SAVE_PATH는 vit.py 내에서 직접 사용되지는 않지만 폴더 구조를 위해 유지
SAVE_PATH="$OUTPUT_DIR/$RUN_NAME/checkpoints"
mkdir -p $LOG_PATH
mkdir -p $SAVE_PATH

# --- 분산 학습 실행 ---
echo "========================================================"
echo "Vision Transformer on ImageNet-1k Training (Algoperf Spec)"
echo "Optimizer: Distributed Shampoo"
echo "GPUs: $N_GPUS"
echo "Total Batch Size: $(($N_GPUS * $BATCH_SIZE_PER_GPU))"
echo "LR: $BASE_LR, WD: $WEIGHT_DECAY, Beta1: $BETA1"
echo "Augmentations: RandAugment(m15-n2), Mixup($MIXUP), LS($LABEL_SMOOTHING)"
echo "Log Path: $LOG_PATH"
echo "========================================================"

# [변경] torchrun 명령어에 새로운 인자 추가
torchrun --standalone --nnodes=1 --nproc_per_node=$N_GPUS $SCRIPT_NAME \
    --data-path $DATA_PATH \
    --log-dir $LOG_PATH \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE_PER_GPU \
    --workers $WORKERS \
    --base-lr $BASE_LR \
    --warmup-steps $WARMUP_STEPS \
    --weight-decay $WEIGHT_DECAY \
    --beta1 $BETA1 \
    --mixup $MIXUP \
    --label-smoothing $LABEL_SMOOTHING \
    --log-interval 200 \
    --save-interval 10

echo "Training finished successfully."