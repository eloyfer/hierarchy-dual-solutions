import numpy as np


try:
  import gurobipy as gp
except ModuleNotFoundError:
  import sys
  import os
  os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH','') + ':/usr/local/gurobi/9.0.0/lib/'
  os.environ['GRB_LICENSE_FILE'] = '/cs/share/etc/license/gurobi/gurobi.lic'
  sys.path.append('/usr/local/gurobi/9.0.0/python/python3.7/site-packages/gurobipy/')
  import gurobipy as gp

from ll_polynomial_utils import get_Phi, get_Phi_nonlinear

def solve_ilp(n,ell,d,m,K_mat,config_set):
  """
  construct the ILP and solve it
  """
  Phi = get_Phi(n,ell,d,m,K_mat,config_set)
  M = K_mat.T @ np.diag(Phi) @ K_mat.T
  model = gp.Model()
  x = model.addMVar(len(config_set), vtype=gp.GRB.BINARY)
  model.setObjective(K_mat[:,0] @ x , gp.GRB.MINIMIZE)
  model.addConstr(sum(x) >= 1)
  model.addConstr((M - np.eye(len(M))) @ x >= 0)
  model.optimize()
  return model.x

def solve_ilp_nonlinear(n,ell,d,m,K_mat,config_set):
  """
  construct the ILP and solve it
  """
  Phi = get_Phi_nonlinear(n,ell,d,m,K_mat,config_set)
  M = K_mat.T @ np.diag(Phi) @ K_mat.T
  model = gp.Model()
  x = model.addMVar(len(config_set), vtype=gp.GRB.BINARY)
  model.setObjective(K_mat[:,0] @ x , gp.GRB.MINIMIZE)
  model.addConstr(sum(x) >= 1)
  model.addConstr((M - np.eye(len(M))) @ x >= 0)
  model.optimize()
  return model.x
