import os
import json
import fire
import glob
from tqdm import tqdm
import pandas as pd

def collect_results(outdir):
  result_files = glob.glob(os.path.join(outdir, 'result_n*.json'))
  results = []
  required_keys = ['n','d','ell','m','sol_value']
  for rf in tqdm(result_files):
    with open(rf,'r') as fid:
      res = json.loads(fid.read())
    assert set(required_keys) <= set(res.keys())
    results.append([res[key] for key in required_keys])
  dat = pd.DataFrame(results, columns=required_keys)
  dat.to_csv(os.path.join(outdir, 'collected_results.csv'), index=False)

  
if __name__ == '__main__':
  fire.Fire(collect_results)
