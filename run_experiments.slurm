#! /bin/bash

#SBATCH --time=05:00:00
#SBATCH --ntasks=2
#SBATCH --mem=1G

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/gurobi/9.0.0/lib/
export PYTHONPATH=$PYTHONPATH:/usr/local/gurobi/9.0.0/python/python3.7/site-packages/gurobipy/:/cs/labs/nati/eloyfer/projects/multivariate-krawchouks/
export GRB_LICENSE_FILE=/cs/share/etc/license/gurobi/gurobi.lic
source /cs/labs/nati/eloyfer/envs/py39/bin/activate


/cs/labs/nati/eloyfer/envs/py39/bin/python manage_experiments.py run_multiple_experiments 21 2 6 
