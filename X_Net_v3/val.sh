#!/bin/bash
# Submission script for Manneback
# Resource request steps
#SBATCH --job-name=xnet_val
#SBATCH --time=30:00:00 # hh:mm:ss
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres="gpu:TeslaA100:1"
#SBATCH --mem-per-cpu=8000 # megabytes
#SBATCH --partition=gpu
#SBATCH --mail-user=nils.boulanger@student.uclouvain.be
#SBATCH --mail-type=ALL
#SBATCH --comment=xnet_val
#SBATCH --output="val_output.txt"
#SBATCH --error="val_error.txt"

# "gpu:TeslaA100:1"

# Job steps
module load releases/2019b
module load Python/3.7.4-GCCcore-8.3.0

mkdir -p /auto/home/users/n/b/nboulang/X_Net/my_venv    # Loading the env.
python3 -m pip install virtualenv    # Installing the env.
virtualenv -p python3 --system-site-packages /auto/home/users/n/b/nboulang/X_Net/my_venv
source /auto/home/users/n/b/nboulang/X_Net/my_venv/bin/activate


pip install --upgrade pip
pip install sklearn
pip install nibabel
pip install torchio
pip install IPython
pip install tensorboardX
pip install SimpleITK
pip install PYMIC
pip install -U setuptools
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

# Files to run
mkdir result
python xnet/pymic/train_infer/train_infer.py test /auto/home/users/n/b/nboulang/X_Net_v3/xnet/examples/miccai/config/train_test.cfg
# mkdir predictions
# python xnet/pymic/util/evaluation.py /auto/home/users/n/b/nboulang/X_Net_v3/xnet/examples/miccai/config/evaluation.cfg
