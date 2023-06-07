
import gorobipy as gp
from ll_polynomial_utils import get_Phi

def solve_ilp(n,ell,d,m,K_mat,config_set):
  """
  construct the ILP and solve it
  """
  Phi = get_Phi(n,ell,d,m,K_mat,config_set)
  M = K_mat.T @ np.diag(Phi) @ K_mat.T
  model = gurobipy.Model()
  x = model.addMVar(len(config_set), vtype=gp.GRB.BINARY)
  model.setObjective(K_mat[:,0] @ x , gp.GRB.MINIMIZE)
  model.addConstr(K[0] @ x >= 1, "c0")
  model.addConstr(M @ x >= 0)
  model.optimize()
  return model.x
