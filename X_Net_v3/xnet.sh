#!/bin/bash
# Submission script for Manneback
# Resource request steps
#SBATCH --job-name=xnet
#SBATCH --time=30:00:00 # hh:mm:ss
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gres="gpu:TeslaA100:1"
#SBATCH --mem-per-cpu=8000 # megabytes
#SBATCH --partition=gpu
#SBATCH --mail-user=nils.boulanger@student.uclouvain.be
#SBATCH --mail-type=ALL
#SBATCH --comment=unet_3d
#SBATCH --output="xnet_output.txt"
#SBATCH --error="xnet_error.txt"

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
pip install -U setuptools
pip install tensorboardX
pip install SimpleITK
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

# Files to run
export PYTHONPATH=$PYTHONPATH:/auto/home/users/n/b/nboulang/Head-Neck-GTV-master/HeadNeck_GTV
python xnet/pymic/train_infer/train_infer.py train /auto/home/users/n/b/nboulang/X_Net_v3/xnet/examples/miccai/config/train_test.cfg
deactivate