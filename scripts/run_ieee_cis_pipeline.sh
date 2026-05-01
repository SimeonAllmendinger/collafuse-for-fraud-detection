#!/bin/bash -l
#SBATCH --job-name=collafuse-ieee-cis
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --export=ALL

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80

cd $HOME/collafuse-for-fraud-detection

module purge
module load cuda/12.1.1
module load openmpi/4.1.6-gcc11.2.0-cuda
module load python/3.12-conda

export STORAGE_DIR="$WORK"
export TMPDIR=${SLURM_TMPDIR:-/tmp}

export TORCH_HOME="$TMPDIR/torch"
mkdir -p $TORCH_HOME


if ! conda info --envs | grep -q "$HOME/.conda/envs/collafuse-fraud"; then
    conda env create -f environment.yml
fi

conda activate $HOME/.conda/envs/collafuse-fraud

set -e

# python3 -m src.cli --config src/config_files/config_baf.yaml run-all-stages
# python3 -m src.cli --config src/config_files/config_credit_card_fraud.yaml run-all-stages
# python3 -m src.cli --config src/config_files/config_elliptic.yaml run-all-stages
# python3 -m src.cli --config src/config_files/config_ieee_cis.yaml run-all-stages
python3 -m src.cli --config src/config_files/config_paysim.yaml run-all-stages