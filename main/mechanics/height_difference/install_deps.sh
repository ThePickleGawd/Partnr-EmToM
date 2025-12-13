#!/bin/bash
# Script to install all dependencies for habitat-llm

# Activate the environment
source /data4/miniconda3/etc/profile.d/conda.sh
conda activate habitat-llm

# Navigate to project root
cd /data4/parth/Partnr-EmToM

echo "Initializing git submodules..."
git submodule sync
git submodule update --init --recursive

echo "Installing PyTorch with CUDA..."
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y

echo "Installing habitat-sim..."
conda install habitat-sim=0.3.3 withbullet headless -c conda-forge -c aihabitat -y

echo "Installing habitat-lab and habitat-baselines..."
pip install -e ./third_party/habitat-lab/habitat-lab
pip install -e ./third_party/habitat-lab/habitat-baselines

echo "Installing transformers-CFG..."
pip install -e ./third_party/transformers-CFG

echo "Installing requirements.txt..."
pip install -r requirements.txt

echo "Installing habitat-llm library..."
pip install -e .

echo "Done! Testing imports..."
python -c "import habitat; import habitat_llm; print('✓ All imports work!')"
