#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH -u

#activate conda env
conda activate master

# debugging flags (optional)

#export NCCL_DEBUG=INFO
#export PYTHONFAULTHANDLER=1
#export PL_TORCH_DISTRIBUTED_BACKEND=gloo
module load cuda/10.0

CUDA_DEVICE=$(echo "$CUDA_VISIBLE_DEVICES," | cut -d',' -f $((SLURM_LOCALID + 1)) );

T_REGEX='^[0-9]$';

if ! [[ "$CUDA_DEVICE" =~ $T_REGEX ]]; then

        echo "error no reserved gpu provided" 

        #exit 1;

fi
echo "Process $SLURM_PROCID of Job $SLURM_JOBID withe the local id $SLURM_LOCALID using gpu id $CUDA_DEVICE (we may use gpu: $CUDA_VISIBLE_DEVICES on $(hostname))" 

srun python run_clip_filtering.py
