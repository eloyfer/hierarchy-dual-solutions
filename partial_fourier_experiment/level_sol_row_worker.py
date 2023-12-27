
import os
import gzip
import pickle
from tqdm import tqdm
from krawtchouk_getter import get_krawtchouk
from level_sol import LevelSol
from phi_func_base import get_phi
import manage_db

class LevelSolRowWorker:

    def __init__(self):
        self.krawtchouks = {}
        self.phi_funcs = {}

    def get_rows(self, rowid_list):
        con = manage_db.get_con()
        cur = con.cursor()
        rowid_str = ', '.join(map(str,rowid_list))
        query = f"SELECT rowid,* FROM {manage_db.tbl_lvl_sols} WHERE rowid IN ({rowid_str}) and is_feasible IS NULL"
        res = cur.execute(query)
        rows = res.fetchall()
        con.close()
        return rows
    
    def get_krawtchouk(self, n, lvl):
        key = (n,lvl)
        if key not in self.krawtchouks:
            self.krawtchouks[key] = get_krawtchouk(n,lvl)
        return self.krawtchouks[key]

    def get_phi(self, phi_name, phi_params, n, d, lvl):
        key = (phi_name, phi_params, n, d, lvl)
        if key not in self.phi_funcs:
            params = phi_params.split(';')
            params = [param.split('=') for param in params]
            params = {str(key): int(value) for key,value in params}
            params.update(dict(n=n,d=d,lvl=lvl,K=self.get_krawtchouk(n,lvl)))
            self.phi_funcs[key] = get_phi(phi_name, params)
        return self.phi_funcs[key]

    def process_rows(self, rowid_list):
        rows = self.get_rows(rowid_list)
        
        is_feasible = {}
        for row in tqdm(rows):
            filename = manage_db.level_solution_path(row['rowid'])
            if not os.path.isfile(filename):
                # create lvl_sol
                lvl_sol = self.create_lvl_sol(row)
                if not os.path.exists(os.path.dirname(filename)):
                    os.makedirs(os.path.dirname(filename))
                with gzip.open(filename,'wb') as fid:
                    fid.write(pickle.dumps(lvl_sol))
            else:
                # load the lvl_sol to update is_feasible
                with gzip.open(filename,'rb') as fid:
                    lvl_sol = pickle.loads(fid.read())
            is_feasible[row['rowid']] = int(lvl_sol.is_feasible())

        self.update_feasible(is_feasible)

    def update_feasible(self, is_feasible):
        query = f"UPDATE {manage_db.tbl_lvl_sols} SET is_feasible=? WHERE rowid=?"
        con = manage_db.get_con()
        cur = con.cursor()
        cur.executemany(query, [(val,key) for key,val in is_feasible.items()])
        con.commit()
        con.close()

    def create_lvl_sol(self, row):
        n = row['n']
        d = row['d']
        lvl = row['lvl']

        phi = self.get_phi(
            row['phi_name'],
            row['phi_params'],
            n,d,lvl,
        )
        
        lvl_sol = LevelSol(
            n=n,
            d=d,
            lvl=lvl,
            configs=row['configs'].configs,
            phi=phi, 
            K = self.get_krawtchouk(n,lvl)
        )
        return lvl_sol
        
        
    

