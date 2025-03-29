#!/bin/bash
#SBATCH --job-name=hyperparam_tuning
#SBATCH --output=/u/student/2021/cs21resch15002/DomainShift/Outputs/ISIC_Dataset/logs/NormalClassification/slurm_output.log
#SBATCH --error=/u/student/2021/cs21resch15002/DomainShift/Outputs/ISIC_Dataset/logs/NormalClassification/slurm_error.log
#SBATCH --partition=cse-gpu-all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

source activate vptta  

DATASET_PATH="/u/student/2021/cs21resch15002/DomainShift/Datasets/ISIC_Dataset"
LOG_PATH1="/u/student/2021/cs21resch15002/DomainShift/Outputs/ISIC_Dataset/logs/NormalClassification/bcn_vs_ham_loc_palms_soles"
LOG_PATH2="/u/student/2021/cs21resch15002/DomainShift/Outputs/ISIC_Dataset/logs/NormalClassification/bcn_vs_ham_loc_head_neck"
LOG_PATH3="/u/student/2021/cs21resch15002/DomainShift/Outputs/ISIC_Dataset/logs/NormalClassification/ham_vs_msk_age_u30"
LOG_PATH4="/u/student/2021/cs21resch15002/DomainShift/Outputs/ISIC_Dataset/logs/NormalClassification/bcn_vs_ham_age_u30"
LOG_PATH5="/u/student/2021/cs21resch15002/DomainShift/Outputs/ISIC_Dataset/logs/NormalClassification/bcn_vs_msk_loc_head_neck"
LOG_PATH6="/u/student/2021/cs21resch15002/DomainShift/Outputs/ISIC_Dataset/logs/NormalClassification/ham_vs_msk_age_u30"
LOG_PATH7="/u/student/2021/cs21resch15002/DomainShift/Outputs/ISIC_Dataset/logs/NormalClassification/ham_vs_msk_loc_head_neck"
LOG_PATH8="/u/student/2021/cs21resch15002/DomainShift/Outputs/ISIC_Dataset/logs/NormalClassification/msk_vs_bcn_age_u30"


MODEL="resnet50"
EPOCHS=20
ITER=500
LR=0.001
BATCH_SIZE=32
NUM_CLASSES=2
PHASE="validate"
# srun python Classification.py \
#   --arch $MODEL \
#   --data ISIC \
#   --root $DATASET_PATH \
#   --source bcn_loc_palms_soles \
#   --target ham_loc_palms_soles \
#   --num_classes $NUM_CLASSES \
#   --batch_size $BATCH_SIZE \
#   --epochs $EPOCHS \
#   --iters_per_epoch $ITER \
#   --lr $LR \
#   --phase $PHASE \
#   --output_dir $LOG_PATH1/lr_$LR

# srun python Classification.py \
#   --arch $MODEL \
#   --data ISIC \
#   --root $DATASET_PATH \
#   --source bcn_loc_head_neck \
#   --target ham_loc_head_neck \
#   --num_classes $NUM_CLASSES \
#   --batch_size $BATCH_SIZE \
#   --epochs $EPOCHS \
#   --iters_per_epoch $ITER \
#   --lr $LR \
#   --phase $PHASE \
#   --output_dir $LOG_PATH2/lr_$LR




# srun python Classification.py \
#   --arch $MODEL \
#   --data ISIC \
#   --root $DATASET_PATH \
#   --source ham_age_u30 \
#   --target msk_age_u30 \
#   --num_classes $NUM_CLASSES \
#   --batch_size $BATCH_SIZE \
#   --epochs $EPOCHS \
#   --iters_per_epoch $ITER \
#   --lr $LR \
#   --phase $PHASE \
#   --output_dir $LOG_PATH3/lr_$LR

srun python Classification.py \
  --arch $MODEL \
  --data ISIC \
  --root $DATASET_PATH \
  --source msk_age_u30 \
  --target bcn_age_u30 \
  --num_classes $NUM_CLASSES \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --iters_per_epoch $ITER \
  --lr $LR \
  --phase $PHASE \
  --output_dir $LOG_PATH8/lr_$LR

# srun python Classification.py \
#   --arch $MODEL \
#   --data ISIC \
#   --root $DATASET_PATH \
#   --source bcn_loc_head_neck \
#   --target msk_loc_head_neck \
#   --num_classes $NUM_CLASSES \
#   --batch_size $BATCH_SIZE \
#   --epochs $EPOCHS \
#   --iters_per_epoch $ITER \
#   --lr $LR \
#   --phase $PHASE \
#   --output_dir $LOG_PATH5/lr_$LR

# srun python Classification.py \
#   --arch $MODEL \
#   --data ISIC \
#   --root $DATASET_PATH \
#   --source ham_age_u30 \
#   --target msk_age_u30 \
#   --num_classes $NUM_CLASSES \
#   --batch_size $BATCH_SIZE \
#   --epochs $EPOCHS \
#   --iters_per_epoch $ITER \
#   --lr $LR \
#   --phase $PHASE \
#   --output_dir $LOG_PATH6/lr_$LR

# srun python Classification.py \
#   --arch $MODEL \
#   --data ISIC \
#   --root $DATASET_PATH \
#   --source ham_loc_head_neck \
#   --target msk_loc_head_neck \
#   --num_classes $NUM_CLASSES \
#   --batch_size $BATCH_SIZE \
#   --epochs $EPOCHS \
#   --iters_per_epoch $ITER \
#   --lr $LR \
#   --phase $PHASE \
#   --output_dir $LOG_PATH7/lr_$LR
