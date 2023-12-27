import os
import sqlite3
import fire
import pandas as pd
from config_list import ConfigList
from itertools import product
from experiment_config import db_file, working_dir


tbl_lvl_sols = 'level_solution'
tbl_full_sols = 'full_solution'


"""
level_sol_table contains level solutions for different 
values of n,d,lvl, and phi functions.
"""

create_table_query = {
    tbl_lvl_sols: f"""
CREATE TABLE {tbl_lvl_sols} (
    n INTEGER,
    d INTEGER,
    lvl INTERGER,
    phi_name TEXT,
    phi_params TEXT,
    configs ConfigList,
    is_feasible INTEGER,
    UNIQUE (n,d,lvl,phi_name,phi_params,configs)
)
""",
    tbl_full_sols: f"""
CREATE TABLE {tbl_full_sols} (
    n INTEGER,
    d INTEGER,
    value REAL,
    lvl1_sol INTEGER DEFAULT 0 NOT NULL,
    lvl2_sol INTEGER DEFAULT 0 NOT NULL,
    lvl3_sol INTEGER DEFAULT 0 NOT NULL,
    UNIQUE (lvl1_sol, lvl2_sol, lvl3_sol)
)
"""
}

def get_con():
    sqlite3.register_converter("ConfigList", ConfigList.convert_from_sql)
    con = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES)
    con.row_factory = sqlite3.Row
    return con

def level_solution_path(rowid):
    filename = f'{rowid:010d}.pkl.gz'
    dirname = f'lvl_sols/{rowid//1000:06d}'
    path = os.path.join(working_dir,dirname,filename)
    return path

def full_solution_path(rowid):
    filename = f'{rowid:010d}.pkl.gz'
    dirname = f'full_sols/{rowid//1000:06d}'
    path = os.path.join(working_dir,dirname,filename)
    return path

def full_solution_description(rowid):
    con = get_con()
    cur = con.cursor()
    query = f"SELECT * FROM {tbl_full_sols} WHERE rowid = {rowid}"
    res = cur.execute(query)
    row = res.fetchone()
    lvl_sol_ids = [row[key] for key in row.keys() if key.endswith('_sol')]
    query = f"SELECT * FROM {tbl_lvl_sols} WHERE rowid in ({', '.join(map(str,lvl_sol_ids))}) ORDER BY lvl"
    dat = pd.read_sql(query, con)
    con.close()
    return dat
    
def create_db():
    con = get_con()
    cur = con.cursor()

    # get existing tables
    query = "SELECT name FROM sqlite_master WHERE type='table'"
    res = cur.execute(query)
    existing_tables = [x['name'] for x in res.fetchall()]

    # create missing tables
    for tbl_name in create_table_query:
        if tbl_name not in existing_tables:
            print(f'creating table {tbl_name}')
            cur.execute(create_table_query[tbl_name])
    con.close()

def validate_configs(n,lvl,configs):
    return all(sum(conf) == n and len(conf) == 2**lvl for conf in configs)

def add_lvl_sols_to_table(n,d,lvl,phi_name,phi_params,configs_list):
    column_list = ['n','d','lvl','phi_name','phi_params','configs']

    assert all(validate_configs(n,lvl,configs) for configs in configs_list), \
        f"Some configurations do not match the parameters n={n} and lvl={lvl}"

    entry_list = [
        (n,d,lvl,phi_name,phi_params, ConfigList(configs)) for configs in configs_list
    ]
    con = sqlite3.connect(DB_NAME)
    cur = con.cursor()

    print(f"expected # new entries: {len(configs_list)}")
    res = cur.execute(f"SELECT COUNT(rowid) FROM {tbl_lvl_sols}")
    print("db size before query: ", res.fetchone())

    col_names = ','.join(column_list)
    values = ','.join(['?']*len(column_list))
    cmd = f"INSERT OR IGNORE INTO {tbl_lvl_sols}({col_names}) VALUES({values})"
    cur.executemany(cmd, entry_list)
    con.commit()
    res = cur.execute(f"SELECT COUNT(rowid) FROM {tbl_lvl_sols}")
    print("db size after query: ", res.fetchone())

    # upadte also the full solution table
    add_rows_to_full_solution_table(n,d,cur)
    con.commit()
    con.close()

def add_rows_to_full_solution_table(n,d,cur):

    # get available levels
    query = f"SELECT DISTINCT lvl FROM {tbl_lvl_sols} WHERE n = n AND d = d AND is_feasible = 1" 
    res = cur.execute(query)
    levels = sorted([row['lvl'] for row in res.fetchall()])
    assert set(levels) == set(range(min(levels), max(levels)+1))

    # get all feasible level-solutions by level
    query_func = lambda lvl: \
            (f"SELECT rowid,phi_name FROM {tbl_lvl_sols} "
            f"WHERE n = {n} AND d = {d} AND lvl = {lvl} AND is_feasible = 1")
    level_rows = {
        lvl: cur.execute(query_func(lvl)).fetchall()
        for lvl in levels 
    }
    level_rowids = {
        lvl: {row['rowid']: row['phi_name'] for row in rows} 
        for lvl,rows in level_rows.items()
    }

    def filter_func(ids):
        """ match level-solutions only on these conditions.
        Args:
            ids - list/tuple of rowid's, representing level solutions
        """
        all_balanced = all(level_rowids[lvl][i] == 'balanced' for lvl, i in enumerate(ids,1))
        even_odd = all(level_rowids[lvl][i] in ['even','odd'] for lvl, i in enumerate(ids,1))
        return all_balanced or even_odd

    for max_lvl in levels:
        columns = ['n','d'] + [f'lvl{lvl}_sol' for lvl in range(1,max_lvl+1)]
        values_str = ['?'] * len(columns)
        columns = ', '.join(columns)
        values_str = ', '.join(values_str)
        query = f"INSERT OR IGNORE INTO {tbl_full_sols}({columns}) VALUES({values_str})"

        values = product(*[level_rowids[lvl].keys() for lvl in range(1,max_lvl+1)])
        values = filter(filter_func, values) 
        values = ((n,d) + tuple(row) for row in values)
        values = list(values)
        print('query:', query)
        print('num rows:', len(values))
        cur.executemany(query, values)

def populate_full_solution_table():

    con = get_con()
    cur = con.cursor()

    # get existing pairs of n,d
    query = f"SELECT DISTINCT n,d FROM {tbl_lvl_sols}"
    res = cur.execute(query)
    for row in res.fetchall():
        n = row['n']
        d = row['d']
        print(f'n: {n}, d: {d}')
        add_rows_to_full_solution_table(n,d,cur)
        con.commit()
    con.close()
    

if __name__ == '__main__':
    fire.Fire()
