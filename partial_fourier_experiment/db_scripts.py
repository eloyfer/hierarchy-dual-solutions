import os
import subprocess
import pandas as pd
import fire
from datetime import datetime
import manage_db
from krawtchouk_getter import get_krawtchouk
from level_sol_row_worker import LevelSolRowWorker
from full_solution_row_worker import FullSolutionRowWorker
from experiment_config import slurm_dir


def populate_lvl_sols(n,d):
    phi_name = 'balanced'
    phi_params = [f'm={m}' for m in range(2,n//2,2)]
    for lvl in [1,2]:
        K = get_krawtchouk(n,lvl)
        configs_list = [[c] for c in K.get_config_set()]
        for pp in phi_params:
            manage_db.add_lvl_sols_to_table(n,d,lvl,phi_name,pp,configs_list)

def compute_all_lvl_sols():
    """
    compute all lvl sols that were not computed yet
    """
    query = f"SELECT MAX(rowid) FROM {manage_db.tbl_lvl_sols}"
    con = manage_db.get_con()
    cur = con.cursor()
    res = cur.execute(query).fetchone()
    con.close()
    max_rowid = res[res.keys()[0]]
    worker = LevelSolRowWorker()
    worker.process_rows(list(range(1,max_rowid+1)))

def compute_lvl_sols_batch(batch_num, batch_size):
    start_idx = batch_size*batch_num
    end_idx = batch_size*(batch_num+1)
    print(f'computing level solutions from {start_idx} to {end_idx}')
    batch_indices = list(range(start_idx, end_idx))
    worker = LevelSolRowWorker()
    worker.process_rows(batch_indices)

def run_sbatch(script, args):
    cmd = ["sbatch"] + args + [script]
    print(' '.join(cmd))
    #subprocess.run(cmd)

parallelize_compute_lvl_sols_script = """#!/bin/bash
PYTHON="/cs/labs/nati/eloyfer/envs/py39/bin/python"
SCRIPT="/cs/labs/nati/eloyfer/projects/hierarchy-dual-solutions/partial_fourier_experiment/db_scripts.py"
export PYTHONPATH=$PYTHONPATH:/cs/labs/nati/eloyfer/projects/multivariate-krawchouks/

env TQDM_DISABLE=1 $PYTHON $SCRIPT compute_lvl_sols_batch --batch_num=$SLURM_ARRAY_TASK_ID --batch_size={batch_size} 
""".format


def parallelize_compute_lvl_sols(batch_size):
    idx0_nm = "start_idx"
    idx1_nm = "end_idx"
    query = (f"SELECT MIN(rowid) AS {idx0_nm}, MAX(rowid) AS {idx1_nm} "
            f"FROM {manage_db.tbl_lvl_sols} WHERE is_feasible IS NULL")
    conn = manage_db.get_con()
    dat = pd.read_sql(query, conn)
    conn.close()
    print('min and max idx:')
    print(dat)
    start_idx = dat.loc[0,idx0_nm]
    end_idx = dat.loc[0,idx1_nm]

    dir_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = os.path.join(slurm_dir, dir_name)
    
    os.makedirs(dir_name)

    script = parallelize_compute_lvl_sols_script(batch_size=batch_size)
    script_file = os.path.join(dir_name, 'run.slurm')
    with open(script_file,'w') as fid:
        fid.write(script)

    args = [
        f'--chdir={dir_name}',
        '-c1',
        '--mem=16G',
        f'--array={start_idx//batch_size:d}-{end_idx//batch_size:d}',
        '--time=1:0:0'
    ]
    run_sbatch(script_file, args)


def print_best(n, d, count=10):
    conn = manage_db.get_con()
    query = f"SELECT * FROM {manage_db.tbl_full_sols} WHERE n = {n} AND d = {d} ORDER BY value"
    dat = pd.read_sql(query,conn)
    dat = dat.head(count)
    lvl_sol_columns = [col for col in dat.columns if col.endswith('_sol') and col.startswith('lvl')]
    for _,row in dat.iterrows():
        print('value:',row['value'])
        rowids = ', '.join([str(row[key]) for key in lvl_sol_columns])
        query = f"SELECT rowid,* FROM {manage_db.tbl_lvl_sols} WHERE rowid IN ({rowids})"
        sol_info = pd.read_sql(query,conn)
        print(sol_info)
    conn.close()


if __name__ == '__main__':
    fire.Fire()
