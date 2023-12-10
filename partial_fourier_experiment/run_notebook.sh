#! /bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/gurobi/9.0.0/lib/
export PYTHONPATH=$PYTHONPATH:/usr/local/gurobi/9.0.0/python/python3.7/site-packages/gurobipy/:/cs/labs/nati/eloyfer/projects/multivariate-krawchouks/
export GRB_LICENSE_FILE=/cs/share/etc/license/gurobi/gurobi.lic
source /cs/labs/nati/eloyfer/envs/py39/bin/activate
jupyter notebook --ip 0.0.0.0 --no-browser
