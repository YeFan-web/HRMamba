#!/bin/bash

set -e

echo "=== Setting up environment for HRMamba evaluation ==="


echo ">>> Upgrading pip, setuptools, wheel"
pip install setuptools==79.0.1

pip install numpy==1.26.0

echo ">>> Installing PyTorch 2.1.1 with CUDA 11.8"
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
conda install -c conda-forge cudatoolkit=11.8 -y

echo ">>> Installing base dependencies"
pip install opencv-python==4.11.0.86
pip install pytorch-msssim
pip install einops==0.8.0
pip install timm==1.0.9
pip install transformers==4.37.2
pip install matplotlib==3.6.0
pip install thop
pip install scikit-image==0.22.0



echo ">>> Installing ninja"
pip install ninja


echo ">>> Installing causal-conv1d and mamba_ssm"
pip install causal-conv1d==1.1.1 --no-build-isolation
pip install mamba_ssm==1.1.1 --no-build-isolation


echo "=== Environment setup completed successfully ==="
echo "You may now run python eval.py"
