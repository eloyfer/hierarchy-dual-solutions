
import sys
sys.path.append('/cs/labs/nati/eloyfer/projects/hierarchy-dual-solutions/ll_polynomial')
sys.path.append('/cs/labs/nati/eloyfer/projects/multivariate-krawchouks/')

from dual_sol_experiment import DualSolExperiment

DualSolExperiment(10, 4, {1: [2,4,6], 2: [4,6]})
