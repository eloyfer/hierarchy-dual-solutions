import os
import pickle
import gzip
import fire
import tqdm
from level_sol import LevelSol
import manage_db
from load_level_sol import load_level_sol
from total_sol import TotalSol

class FullSolutionRowWorker:

    def __init__(self):
        self.lvl_sols = {} # a dict of {rowid: lvl_sol object}

    def get_rows(self, rowid_list):
        con = manage_db.get_con()
        cur = con.cursor()
        rowids_str = ', '.join(map(str, rowid_list))
        query = f"SELECT rowid,* FROM {manage_db.tbl_full_sols} WHERE rowid IN ({rowids_str}) AND value IS NULL"
        res = cur.execute(query)
        res = res.fetchall()
        con.close()
        return res

    def get_lvl_sol(self, rowid):
        if rowid not in self.lvl_sols:
            filename = manage_db.level_solution_path(rowid)
            with gzip.open(filename, 'rb') as fid:
                self.lvl_sols[rowid] = pickle.load(fid)
        return self.lvl_sols[rowid]

    def process_all_rows(self):
        con = manage_db.get_con()
        cur = con.cursor()
        query = f"SELECT rowid FROM {manage_db.tbl_full_sols} WHERE value IS NULL"
        res = cur.execute(query)
        res = res.fetchall()
        con.close()
        res = [row['rowid'] for row in res]
        self.process_rows(res)

    def process_rows(self, rowid_list):
        rows = self.get_rows(rowid_list)

        values = {}
        for row in tqdm.tqdm(rows):
            sol_id = row['rowid']
            values[sol_id] = self.process_row(row)

        self.update_values(values)

    def process_row(self, row):
        sol_id = row['rowid']
        lvl_sol_ids = {lvl: row[f'lvl{lvl}_sol'] for lvl in range(1,4)}
        lvl_sols = {lvl: self.get_lvl_sol(rid) for lvl,rid in lvl_sol_ids.items() if rid > 0}

        n = row['n']
        max_lvl = max(lvl_sols)
        full_sol = TotalSol(n, max_lvl)
        for lvl in sorted(lvl_sols, reverse=True):
            full_sol.add_level(lvl, lvl_sols[lvl])

        # save pickle
        filename = manage_db.full_solution_path(sol_id)
        if not os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
            
        with gzip.open(filename,'wb') as fid:
            fid.write(pickle.dumps(full_sol))

        value = full_sol.compute_value()
        value = float(value)**(1./max_lvl)
        return value

    def update_values(self, sol_values):
        query = f"UPDATE {manage_db.tbl_full_sols} SET value=? WHERE rowid=?"
        con = manage_db.get_con()
        cur = con.cursor()
        cur.executemany(query, [(val,key) for key,val in sol_values.items()])
        con.commit()
        con.close()

if __name__ == '__main__':
    fire.Fire(FullSolutionRowWorker)
