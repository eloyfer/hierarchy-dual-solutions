
import os
import json
import fire
import numpy as np
from ll_integer_lp import solve_ilp

try:
  from create_krawchouk import get_krawchouk_column, index_set_generator, get_krawchouk_recurrence_coeffs
  from create_phi import get_L1_lin_matrix, get_coefficients_lin, get_index_set
  from fast_walsh_hadamard_transform import fwht
 except ModuleNotFoundError:
  import sys
  sys.path.append('/cs/labs/nati/eloyfer/projects/multivariate-krawchouks/')
  from create_krawchouk import get_krawchouk_column, index_set_generator, get_krawchouk_recurrence_coeffs
  from create_phi import get_L1_lin_matrix, get_coefficients_lin, get_index_set
  from fast_walsh_hadamard_transform import fwht

def get_krawtchouk_matrix(n,ell):
    I = get_index_set(n,1<<ell)
    kraw_recurrence_coeffs = {a: get_krawchouk_recurrence_coeffs(a) for a in tqdm(I, desc='coeffs getter')}
    def coeffs_getter(a):
        return kraw_recurrence_coeffs[a]
    K = [get_krawchouk_column(a, coeffs_getter=coeffs_getter) for a in tqdm(I, desc='create krawtchouk')]
    K = [[col[a] for a in I] for col in K]
    K = np.array(K).T
    return K

def save_result(n,d,ell,m,sol_json,out_dir):
  filename = 'result_n{n}_d{d}_ell{ell}_m{m}.json'
  with open(os.path.join(out_dir, filename), 'w') as fid:
    fid.write(sol_json)

def solution_to_support(sol_vec, K_mat, config_set):
  support_idx = np.nonzero(np.round(sol_vec))
  support = [config_set[i] for i in support_idx]
  support_size = K_mat[support_idx,0].sum()
  return support,support_size

def solution_to_json(n,d,ell,m,support,support_size):
  sol_value = support_size**(1./ell)
  result = {
    'n': n, 'd':d, 'ell':ell,'m':m,'support':[list(x) for x in support],
    'support_size': support_size,
    'sol_value': sol_value
  }
  return json.dumps(result, sort_keys=True, indent=4)
  
 
def run_single_experiment(n,d,ell,m,out_dir):
  
  config_set = get_index_set(n,1<<ell)
  K_mat = get_krawtchouk_matrix(n,ell)
  sol = solve_ilp(n,ell,d,m,K_mat,config_set)
  support,support_size = solution_to_support(sol, K_mat, config_set)
  sol_json = solution_to_json(n,d,ell,m,support,support_size)
  print(sol_json)
  save_result(n,d,ell,m,sol_json,out_dir)
  
  
  
if __name__ == '__main__':
  fire.Fire()
