
from collections import defaultdict
from itertools import product
import pandas as pd

from solution_factory2 import SolutionFactory2
from total_sol import TotalSol


class DualSolExperiment:
    
    def __init__(self, n, d, m_vals):
        assert len(m_vals) == max(m_vals)
        assert min(m_vals) == 1
        self.n = n
        self.d = d
        self.ell = max(m_vals)
        self.m_vals = m_vals
        self.sol_factory = SolutionFactory2(n,d)
        
        self.init_all_level_solutions()
        print(self.report_num_feasible())
        self.init_final_sols()
        print(self.report_best_sols())
        
    def init_all_level_solutions(self):
        self.lvl_sols = {}
        for lvl in range(1, self.ell+1):
            self.lvl_sols[lvl] = []
            for m in self.m_vals[lvl]:
                self.add_level_solutions(lvl, m)

    def add_level_solutions(self, lvl, m):
        self.lvl_sols[lvl] += self.sol_factory.get_all_level_sols(lvl, m)
    
    def report_num_feasible(self):
        dat = pd.DataFrame(
            index=[(lvl,m) for lvl in range(1,self.ell+1) for m in self.m_vals[lvl]],
            columns=['total', 'feasible']
        )
        dat['total'] = 0
        dat['feasible'] = 0
        for lvl in range(1, self.ell+1):
            for sol in self.lvl_sols[lvl]:
                dat.at[(lvl,sol.m), 'total'] += 1
                dat.at[(lvl,sol.m), 'feasible'] += int(sol.is_feasible())
        return dat
    
    def init_final_sols(self):
        self.final_sols = []
        for max_lvl in range(1, self.ell+1):
            sols = [TotalSol(self.n, max_lvl)]
            for lvl in range(max_lvl, 0, -1):
                print('lvl', lvl)
                sols_new = []
                for sol in sols:
                    for lvl_sol in filter(lambda x: x.is_feasible(), self.lvl_sols[lvl]):
                        new_sol = sol.get_copy()
                        new_sol.add_level(lvl, lvl_sol)
                        sols_new.append(new_sol)
                sols = sols_new
            self.final_sols += sols
        
    def report_best_sols(self):
        index_col = []
        for max_lvl in range(1,self.ell+1):
            index_col += [
                tuple(ms) for ms in product(*[self.m_vals[lvl] for lvl in range(1,max_lvl+1)])
            ]
        data = {
            'value': [2.**(self.n)] * len(index_col),
            'config': None
        }
        dat = pd.DataFrame(data, index=index_col)
        for sol in self.final_sols:
            key = tuple(sol.lvl_sols[lvl].m for lvl in range(1,sol.max_lvl+1))
            value = float(sol.compute_value())**(1./sol.max_lvl)
            if value < dat.at[key,'value']:
                dat.at[key,'value'] = value
                dat.at[key,'config'] = repr(sol) # tuple(sol.lvl_sols[lvl].configs for lvl in range(1,sol.max_lvl+1))
        return dat
    
