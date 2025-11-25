#!/bin/bash
#SBATCH --job-name=ssl_resnet18
#SBATCH --account=csci_ga_2572-2025fa     <-- YOUR CLASS ACCOUNT 
#SBATCH --partition=c12m85-a100-1         <-- A100 GPU Partition [cite: 83, 94]
#SBATCH --gres=gpu:1                      <-- Request 1 GPU
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00                   <-- Reduced to 12h (Spot instances are often capped)
#SBATCH --requeue                         <-- REQUIRED: Auto-restart if Google interrupts you 
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# 1. Environment Setup
source activate ssl_project

# 2. Fast Data Copy to SSD (Crucial for Speed)
echo "Step 1: Copying data to local SSD ($SLURM_TMPDIR)..."
mkdir -p $SLURM_TMPDIR/data
# Update this path if your zips are elsewhere!
unzip -q -o "./data/cc3m_all/*.zip" -d $SLURM_TMPDIR/data/cc3m_all

# 3. Run Training
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "Step 2: Starting Training..."

python src/train.py \
    --dataset project_data \
    --arch resnet18 \
    --epochs 100 \
    --batch_size 256 \
    --lr 1e-3

echo "Job finished at $(date)"