import os
import pickle
import gzip
import fire
import yaml
from krawtchouk import Krawtchouk
from experiment_config import krawtchouks_dir

def get_krawtchouks_dir():
    return krawtchouks_dir

def get_file_name(n,lvl):
    return f'kraw.{n:03d}.{lvl}.pkl.gz'

def get_krawtchouk(n, lvl, cache_dir=None, save=False):
    cache_dir = cache_dir or get_krawtchouks_dir()
    filename = os.path.join(cache_dir, get_file_name(n,lvl))
    if os.path.isfile(filename):
        print(f'loading Krawtchouk from {filename}')
        with gzip.open(filename,'rb') as fid:
            K = pickle.load(fid)
    else:
        K = Krawtchouk(n,lvl)
        if save:
            print(f'saving Krawtchouk as {filename}')
            with gzip.open(filename,'wb') as fid:
                fid.write(pickle.dumps(K))
    return K

if __name__ == '__main__':
    fire.Fire(get_krawtchouk)
