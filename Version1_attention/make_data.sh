#!/bin/bash
#SBATCH -A nlp
#SBATCH -n 15
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=koushik.viswanadha@students.iiit.ac.in
#SBATCH --mail-type=ALL

module add cuda/10.0

echo "Activating conda environment"
~/anaconda3/bin/conda activate project

echo "making the training data"
python data.py train
echo "Done with the training data"

echo "making the testing data"
python data.py test
echo "Done with the testing data"

echo "making the validation data"
python data.py eval
echo "Done with the validation data"
