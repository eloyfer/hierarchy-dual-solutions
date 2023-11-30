
from collections import defaultdict

from solution_factory2 import SolutionFactory2
from total_sol import TotalSol


class DualSolExperiment:
    
    def __init__(self, n, d, ell, m_vals):
        assert set(range(1,ell+1)) <= set(m_vals.keys())
        self.n = n
        self.d = d
        self.ell = ell
        self.m_vals = m_vals
        self.sol_factory = SolutionFactory2(n,d)
        
        self.init_level_solutions()
        self.print_num_feasible()
        self.init_final_sols()
        self.print_best_sols()
        
    def init_level_solutions(self):
        self.lvl_sols = {}
        for lvl in range(1, self.ell+1):
            self.lvl_sols[lvl] = []
            for m in self.m_vals:
                self.lvl_sols[lvl] += self.sol_factory.get_all_level_sols(lvl, m)
    
    def report_num_feasible(self):
        num_feasible = {
            lvl: {m: 0 for m in self.m_vals[lvl]}
            for lvl in range(1, self.ell+1)
        }
        for lvl in range(1, self.ell+1):
            for sol in self.lvl_sols[lvl]:
                num_feasible[lvl][sol.m] += int(sol.is_feasible())
        return num_feasible
    
    def print_num_feasible(self):
        print('num feasible:')
        num_feasible = self.report_num_feasible()
        for lvl in sorted(num_feasible):
            print(f'level {lvl}')
            for m in sorted(num_feasible[lvl]):
                print(f'\tm={m}: {num_feasible[lvl][m]}/{len([sol for sol in self.lvl_sols[lvl] if sol.m == m])}')
    
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
        
#         self.min_value = min([float(sol.compute_value())**(1./self.ell) for sol in self.final_sols])
#         print('min value:', self.min_value)
    
    def report_best_sols(self):
        best_sols = {
            lvl: {m: 2**(self.n*lvl) for m in m_vals[lvl]} 
            for lvl in range(1,self.ell+1)
        }
        for sol in self.final_sols:
            best_sols[sol.lvl][sol.m] = min(
                best_sols[sol.lvl][sol.m],
                float(sol.compute_value())**(1./sol.lvl)
            )
        return best_sols
    
    def print_best_sols(self):
        print('best solutions:')
        best_sols = self.report_best_sols()
        for lvl in sorted(best_sols):
            print(f'level {lvl}')
            for m in sorted(best_sols[lvl]):
                print(f'\tm={m}: {best_sols[lvl]}')
