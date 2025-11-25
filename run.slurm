#!/bin/bash
#SBATCH --job-name=ssl_resnet18
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# 1. Environment Setup
source activate dl_env

# 2. Fast Data Copy (The "Pro" Move)
# This copies data from slow /scratch to the fast local SSD ($SLURM_TMPDIR)
echo "Step 1: Copying data to local SSD ($SLURM_TMPDIR)..."
mkdir -p $SLURM_TMPDIR/data

# Unzip directly to the fast drive
unzip -q -o "../data/cc3m_all/*.zip" -d $SLURM_TMPDIR/data/cc3m_all

echo "Step 2: Data copy finished. Files in SSD:"
ls $SLURM_TMPDIR/data/cc3m_all | head -n 5

# 3. Run Training
# We add the root folder to python path so it finds 'src'
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Step 3: Starting Training..."
python src/train.py \
    --dataset project_data \
    --arch resnet18 \
    --epochs 100 \
    --batch_size 256 \
    --lr 1e-3

echo "Job finished at $(date)"