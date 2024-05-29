#!/bin/sh
#SBATCH -J MutatedCooperators
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dbrewster@g.harvard.edu
#SBATCH --mem-per-cpu=4G
#SBATCH --ntasks=100
module load python/3.10.13-fasrc01
pip3 install --user numpy scipy seaborn pandas networkx
python3 martins-dream.py