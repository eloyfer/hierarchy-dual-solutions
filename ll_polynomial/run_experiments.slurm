#! /bin/bash

#SBATCH --time=1-00:00:00
#SBATCH --mem=32G
#SBATCH --array=13,15,18,21,22,23,24,25
#SBATCH --chdir=/cs/labs/nati/eloyfer/projects/LL-poly-solutions/outputs

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/gurobi/9.0.0/lib/
export PYTHONPATH=$PYTHONPATH:/usr/local/gurobi/9.0.0/python/python3.7/site-packages/gurobipy/:/cs/labs/nati/eloyfer/projects/multivariate-krawchouks/
export GRB_LICENSE_FILE=/cs/share/etc/license/gurobi/gurobi.lic
source /cs/labs/nati/eloyfer/envs/py39/bin/activate

/cs/labs/nati/eloyfer/envs/py39/bin/python ../manage_experiments.py run_multiple_experiments ${SLURM_ARRAY_TASK_ID} 2 6 ./
