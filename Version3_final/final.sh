#!/bin/bash
#SBATCH -A nlp
#SBATCH -n 20
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=koushik.viswanadha@students.iiit.ac.in
#SBATCH --mail-type=ALL

module add cuda/10.0

echo "Activating conda environment"
source activate project

echo "training the model"
python train.py
