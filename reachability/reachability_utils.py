import numpy as np
import pandas as pd
import sys
import tqdm
from collections import Counter
from itertools import product
# from IPython.display import display, Math


def print_table(dat):
  rows_str = []
  for index, row in dat.iterrows():
#         bias = row[-1]
#         row = row[:-1]
    nonzero = row.apply(lambda x : x != 0)
    row_vars = dat.columns[nonzero]
    var_coeff = row[nonzero].to_numpy()
    signs = ['+'*int(coeff>0) + '-'*int(coeff<0) for coeff in var_coeff]
    var_coeff_str = [str(x) * int(x != 1) for x in np.abs(var_coeff)]
    row_parts = [sign + coeff + str(col) for sign, coeff, col in zip(signs,var_coeff_str,row_vars)]
    row_str = ' '.join(row_parts)
    row_str = row_str.strip()
    if len(row_str) == 0:
      row_str = '0'
    row_str += f' = 0'
    row_str = row_str.lstrip('+')
    row_str = row_str.replace('$','')
    rows_str.append(row_str)
  return '\n'.join(rows_str)


def pivot_df(df,row,col,row0=None):
    M = df.to_numpy().astype(np.float32)
    row = df.index.get_loc(row)
    col = df.columns.get_loc(col)
    if row0 is not None:
        row0 = df.index.get_loc(row0)
    pivot(M,row,col,row0=row0)
    if (M.round() == M).all():
        M = M.astype(np.int32)
    return pd.DataFrame(M, index=df.index, columns=df.columns)


def pivot(mat,row,col,row0=None):
  if row0 is not None:
    mat[[row0,row]] = mat[[row,row0]]
    row = row0
  mat[row] /= mat[row,col]
  mat -= (mat[[row]] * mat[:,[col]]) * (np.arange(len(mat)) != row).astype(np.float32).reshape(-1,1)

def reduce_to_row_echelon_form(mat):
  mat = np.copy(mat).astype(np.float32)
  rows,cols = mat.shape
  r0 = 0
  for c in range(cols):
    r = np.nonzero(mat[r0:,c])[0]
    if len(r) == 0:
      continue
    else:
      r = r0 + r[0]
      pivot(mat,r,c,r0)
      r0 += 1
  return mat


def uv_label(u,v,letter='s'):
    return f'$s({u},{v})$'

def delta_label(v):
    return f'$\Delta({v})$'

def generate_walk_conditions(U,ell):
  """
  Create the eqaution system for a walk between conf0 and conf1 that consists of
  steps from the set U
  """
  U = Counter(U)
  dat = pd.DataFrame(
    data=np.zeros([len(U)+2**ell-1, len(U) * 2**ell + 2**ell], dtype=np.int32),
    columns=[uv_label(u,v) for u in U for v in range(2**ell)] + ['m'] + [delta_label(v) for v in range(1,2**ell)]
  )
  row = 0
  for u in U:
    for v in range(2**ell):
      dat.loc[row, uv_label(u,v)] = 1
    dat.loc[row,'m'] = -U[u]
    row += 1
  for v in range(1,2**ell):
    for u in U:
      dat.loc[row, uv_label(u,v)] = -1
      dat.loc[row, uv_label(u,v^u)] = 1
    dat.loc[row, delta_label(v)] = -1
    row += 1
  return dat
