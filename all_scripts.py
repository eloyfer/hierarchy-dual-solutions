%%writefile krawtchouk.py

from tqdm import tqdm
from create_krawchouk import get_krawchouk_column, index_set_generator, get_krawchouk_recurrence_coeffs

class Krawtchouk:
    
    def __init__(self, n ,ell):
        self.n = n
        self.ell = ell
        self.config_set = list(index_set_generator(n,2**ell))
        self.init_matrix()
        self.init_symmetries()
        
    def get_config_set(self):
        return self.config_set[:]
    
    def config2index(self, config):
        return self.config_set.index(config)
    
    def index2config(self, index):
        return self.config_set[index]
    
    def init_matrix(self):
        kraw_recurrence_coeffs = {
            a: get_krawchouk_recurrence_coeffs(a) 
            for a in tqdm(self.config_set, desc='coeffs getter')
        }
        def coeffs_getter(a):
            return kraw_recurrence_coeffs[a]
        KT = [
            get_krawchouk_column(a, coeffs_getter=coeffs_getter) 
            for a in tqdm(self.config_set, desc='create krawtchouk')
        ]
        KT = [[col[a] for a in self.config_set] for col in KT]
        self.KT = KT
        
        # transpose
        n_conf = len(self.config_set)
        self.K = [[self.KT[i][j] for i in range(n_conf)] for j in range(n_conf)]
    
    def init_symmetries(self):
        # !!!! this solution only works for ell <= 2 !!!!
        self.config_to_orbit = {a: (a[0],) + tuple(sorted(a[1:])) for a in self.config_set}
        self.orbit_list = sorted(set(self.config_to_orbit.values()), key=lambda a: self.config_set.index(a))
        self.orbit_map = {}
        for orbit_rep, conf in self.config_to_orbit.items():
            self.orbit_map[orbit_rep] = self.orbit_map.get(orbit_rep,[]) + [conf]
    
    def get_orbit_list(self):
        return self.orbit_list
    
    def __call__(self, conf1, conf2):
        row = self.config_set.index(conf1)
        col = self.config_set.index(conf2)
        return self.K[row][col]
    
    def get_row(self, conf):
        row = self.config_set.index(conf)
        return self.K[row]
    
    def get_column(self, conf):
        col = self.config_set.index(conf)
        return self.KT[col]
    
    def get_slice_rows(self, lvl):
        """
        Return all the rows that correspond to a configuration of degree lvl
        Args:
            lvl: int, between 0 and n
        Returns:
            A list of lists
        """
        configs = [a for a in self.config_set if a[0] == self.n - lvl]
        return [self.get_row(a) for a in configs]
    
    def get_orbit_rows(self, config):
        """
        Return all the rows that correspond to a configuration in the orbit of config
        """
        orbit_rep = self.config_to_orbit[config]
        orbit_configs = self.orbit_map[orbit_rep]
        return [self.get_row(a) for a in orbit_configs]
    
    def matmul(self, other, transposed=False):
        dim1 = len(self.K)
        dim2 = dim1 # == len(self.KT)
        dim3 = len(other[0])
        assert len(other) == dim1
        result = []
        mat = self.KT if transposed else self.K
        for i in range(dim1):
            row = []
            for j in range(dim3):
                val = sum(mat[i][k]*other[k][j] for k in range(dim2))
                row.append(val)
            result.append(row)
        return result
    
    def vec_mul(self, vec, transposed=False):
        result = self.kraw.matmul([[x] for x in vec], transposed)
        # flatten
        result = list(itertools.chain(*result))
        return result

    
    @staticmethod
    def convolve(n, ell, func1, func2):
        config_set = list(index_set_generator(n,2**ell))
        assert len(func1) == len(func2)
        assert len(func1) == len(config_set)
        # TODO: complete this function


%%writefile solution_factory.py


from krawtchouk import Krawtchouk
from level_sol import LevelSol

class SolutionFactory:
    
    def __init__(self, n, d, m, lvl):
        self.n = n
        self.d = d
        
        self.kraw = Krawtchouk(n,lvl) # Krawtchouk matrices, indexed by level
        self.phi = self.compute_phi() # phi functions, indexed by (level,m)
        self.valid_region = self. # lists of valid configurations, indexed by level
    
    def get_kraw(self, lvl):
        if lvl not in self.kraw:
            self.kraw[lvl] = Krawtchouk(self.n,lvl)
        return self.kraw[lvl]
    
    def get_phi(self, lvl, m):
        key = (lvl,m)
        if key not in self.phi:
            self.phi[key] = self.compute_phi(lvl, m)
        return self.phi[key]
    
    def compute_phi(self):
        K = self.kraw
        K1 = K.get_slice_rows(1)
        num_configs = len(K.get_config_set())
        def phi_func_eval(index):
            if m % 2 == 0:
                return sum((K1[i][index] + self.d)**m - (self.n - self.d)**m for i in range(len(K1)))
            else:
                return sum(K1[i][index]**m - (self.n - 2 * self.d)**m for i in range(len(K1)))
        phi = [phi_func_eval(j) for j in range(num_configs)]
        
        assert self.check_nonpos_constraint(phi, lvl)
        
        return phi
    
    def check_nonpos_constraint(self, func, lvl):
        """
        Verify the func is non-positive in the valid region
        """
        return all([func[self.get_kraw(lvl).config2index(a)] <= 0 for a in self.get_valid_region(lvl)])
    
    def get_valid_region(self, lvl):
        if lvl not in self.valid_region:
            self.valid_region[lvl] = self.compute_valid_region(lvl)
        return self.valid_region[lvl]

    def compute_valid_region(self, lvl):
        K = self.get_kraw(lvl)
        K1 = K.get_slice_rows(1)
        
        def filter_func(idx):
            return all(K1[row][idx] <= self.n-2*self.d for row in range(len(K1)))
        
        config_set = K.get_config_set()
        valid_region = [config for idx,config in enumerate(config_set) if filter_func(idx)]
        return valid_region
    
    def get_sol(self, lvl, m, configs):
        """
        Args:
            lvl: int, level of the solution
            m: int, a parameter of the phi function
            configs: list of configurations
        Returns:
            A LevelSol object 
        """
        # create solution object and store in a dict
        sol = LevelSol(
            lvl=lvl,
            n=self.n,
            d=self.d,
            m=m,
            configs=configs,
            phi=self.get_phi(lvl,m), 
            kraw=self.get_kraw(lvl), 
            valid_region=self.get_valid_region(lvl)
        )
        return sol
    
    def get_all_level_sols(self, lvl, m):
        K = self.get_kraw(lvl)
        singles = [self.get_sol(lvl, m, [config]) for config in K.get_orbit_list()]
        orbits = [self.get_sol(lvl, m, K.orbit_map[config]) for config in K.get_orbit_list()]
        return singles + orbits


%%writefile solution_factory2.py


from krawtchouk import Krawtchouk
from level_sol import LevelSol

class SolutionFactory:
    
    def __init__(self, n, d):
        self.n = n
        self.d = d
        
        self.kraw = {} # Krawtchouk matrices, indexed by level
        self.phi = {} # phi functions, indexed by (level,m)
        self.valid_region = {} # lists of valid configurations, indexed by level
    
    def get_kraw(self, lvl):
        if lvl not in self.kraw:
            self.kraw[lvl] = Krawtchouk(self.n,lvl)
        return self.kraw[lvl]
    
    def get_phi(self, lvl, m):
        key = (lvl,m)
        if key not in self.phi:
            self.phi[key] = self.compute_phi(lvl, m)
        return self.phi[key]
    
    def compute_phi(self, lvl, m):
        K = self.get_kraw(lvl)
        K1 = K.get_slice_rows(1)
        num_configs = len(K.get_config_set())
        def phi_func_eval(index):
            if m % 2 == 0:
                return sum((K1[i][index] + self.d)**m - (self.n - self.d)**m for i in range(len(K1)))
            else:
                return sum(K1[i][index]**m - (self.n - 2 * self.d)**m for i in range(len(K1)))
        phi = [phi_func_eval(j) for j in range(num_configs)]
        
        assert self.check_nonpos_constraint(phi, lvl)
        
        return phi
    
    def check_nonpos_constraint(self, func, lvl):
        """
        Verify the func is non-positive in the valid region
        """
        return all([func[self.get_kraw(lvl).config2index(a)] <= 0 for a in self.get_valid_region(lvl)])
    
    def get_valid_region(self, lvl):
        if lvl not in self.valid_region:
            self.valid_region[lvl] = self.compute_valid_region(lvl)
        return self.valid_region[lvl]

    def compute_valid_region(self, lvl):
        K = self.get_kraw(lvl)
        K1 = K.get_slice_rows(1)
        
        def filter_func(idx):
            return all(K1[row][idx] <= self.n-2*self.d for row in range(len(K1)))
        
        config_set = K.get_config_set()
        valid_region = [config for idx,config in enumerate(config_set) if filter_func(idx)]
        return valid_region
    
    def get_sol(self, lvl, m, configs):
        """
        Args:
            lvl: int, level of the solution
            m: int, a parameter of the phi function
            configs: list of configurations
        Returns:
            A LevelSol object 
        """
        # create solution object and store in a dict
        sol = LevelSol(
            lvl=lvl,
            n=self.n,
            d=self.d,
            m=m,
            configs=configs,
            phi=self.get_phi(lvl,m), 
            kraw=self.get_kraw(lvl), 
            valid_region=self.get_valid_region(lvl)
        )
        return sol
    
    def get_all_level_sols(self, lvl, m):
        K = self.get_kraw(lvl)
        singles = [self.get_sol(lvl, m, [config]) for config in K.get_orbit_list()]
        orbits = [self.get_sol(lvl, m, K.orbit_map[config]) for config in K.get_orbit_list()]
        return singles + orbits

%%writefile level_sol.py

import math
from fractions import Fraction
import itertools

class LevelSol:
    
    def __init__(self, lvl, n, d, m, configs, phi, kraw, valid_region):
        
        assert lvl == int(math.log2(len(configs[0])))
        assert n == sum(configs[0])
        
        self.lvl = lvl
        self.n = n
        self.d = d
        self.m = m
        self.configs = configs

        self.phi = phi
        self.kraw = kraw
        self.valid_region = valid_region
        
        self.init_sol()
        
    def init_sol(self):
        """
        """
        gamma_sqr = self.get_gamma_squared()
        raw_sol = [x * y for x,y in zip(self.phi, gamma_sqr)]
        self.raw_sol = raw_sol
        
        
        # compute the Fourier transform by multiplying with the Krawtchouk matrix
        func_fourier = self.kraw.matmul([[x] for x in raw_sol], transposed=True)

        # flatten
        func_fourier = list(itertools.chain(*func_fourier))

        # divide by 2^(ell*n)
        norm_fac = 2**(self.lvl * self.n)
#         assert all(x % norm_fac == 0 for x in func_fourier)
        func_fourier = [Fraction(x, norm_fac) for x in func_fourier]
        self.raw_sol_fourier = func_fourier
        
        # keep useful quantities
        self.max_in_valid_region = max(self.raw_sol[self.kraw.config2index(a)] for a in self.valid_region)
        self.min_fourier_val = min(func_fourier)
        
        # compute the actual solution
        if self.raw_sol_fourier[0] > 0:
            denom = self.raw_sol_fourier[0]
            self.sol = [Fraction(x, denom) - 1 for x in self.raw_sol]
        
    
    def get_gamma_squared(self):
        K_rows = [self.kraw.get_row(a) for a in self.configs]
        gamma_sqr = [Fraction(sum(col), len(col))**2 for col in zip(*K_rows)]
        
        # symmetrize
        gamma_sqr_on_robit = {
            a0: Fraction(sum(gamma_sqr[self.kraw.config2index(a)] for a in orbit), len(orbit))
            for a0, orbit in self.kraw.orbit_map.items()
        }
        
        gamma_sqr = [
            gamma_sqr_on_robit[self.kraw.config_to_orbit[a]] 
            for a in self.kraw.config_set
        ]
        
        return gamma_sqr
        
    
    def is_feasible(self):
        fourier_feasible = self.min_fourier_val >= 0
        valid_region_feasible = self.max_in_valid_region <= 0
        
        # this one is not strictly necessary but is convenient
        non_zero_fourier_at_0 = self.raw_sol_fourier[0] > 0
        
        return fourier_feasible and valid_region_feasible and non_zero_fourier_at_0
    
    def __repr__(self):
        return f'LevelSol(n={self.n},lvl={self.level},d={self.d},m={self.m},configs={self.configs})'


%%writefile total_sol.py

from fractions import Fraction

class TotalSol:
    """
    An object that stores the full dual solution, including the functions
    and the coefficients.
    """
    
    def __init__(self, n, ell):
        self.n = n
        self.ell = ell
        self.lambdas = {}
        self.lvl_sols = {}
    
    def get_copy(self):
        self_copy = TotalSol(self.n, self.ell)
        self_copy.lambdas = self.lambdas.copy()
        self_copy.lvl_sols = self.lvl_sols.copy()
        return self_copy
    
    def add_level(self, lvl, lvl_sol):
        assert lvl_sol.lvl == lvl
        assert lvl not in self.lvl_sols
        assert lvl == self.ell or lvl == min(self.lvl_sols)-1
        
        self.lvl_sols[lvl] = lvl_sol
        self.lambdas[lvl] = self.compute_lambda(lvl)
    
    def compute_lambda(self, lvl):
        assert lvl in self.lvl_sols
        
        if lvl == self.ell:
            return 1
        else:
            valid_region = self.lvl_sols[lvl].valid_region

            config_sets = {
                j: self.lvl_sols[j].kraw.get_config_set()
                for j in range(lvl, self.ell+1)
            }

            
            def prod(x,y):
                return x*y
            
            values = {}
            for config in valid_region:
                cur_val = 0
                for j in range(lvl+1,self.ell+1):
                    config_j = config + (0,) * (2**j - 2**lvl)
                    index_j = config_sets[j].index(config_j)
                    func_val = self.lvl_sols[j].sol[index_j]
                    coeff = functools.reduce(prod, [2**(lvl+i) - 1 for i in range(1,j-lvl+1)])
                    denom = functools.reduce(prod, [2**i - 1 for i in range(1,j-lvl+1)])
                    cur_val += Fraction(
                        coeff * self.lambdas[j] * func_val,
                        denom
                    )
                coeff = functools.reduce(prod, [2**(lvl+i) - 1 for i in range(1,self.ell-lvl+1)])
                denom = functools.reduce(prod, [2**i - 1 for i in range(1,self.ell-lvl+1)])
                cur_val += Fraction(coeff, denom)

                index_lvl = config_sets[lvl].index(config)
                cur_val *= Fraction(1, -self.lvl_sols[lvl].sol[index_lvl])

                values[config] = cur_val

            return max(values.values())
    
    def compute_value(self):
        return 1 + sum(self.lambdas[j] * self.lvl_sols[j].sol[0] for j in self.lambdas.keys())


%%writefile dual_sol_experiment.py

from collections import defaultdict

from solution_factory import SolutionFactory
from total_sol import TotalSol


class DualSolExperiment:
    
    def __init__(self, n, d, ell, m_vals):
        assert set(range(1,ell+1)) <= set(m_vals.keys())
        self.n = n
        self.d = d
        self.ell = ell
        self.m_vals = m_vals
        self.sol_factory = SolutionFactory(n,d)
        
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


%%writefile run_experiment.py

import sys
sys.path.append('/cs/labs/nati/eloyfer/projects/hierarchy-dual-solutions/ll_polynomial')
sys.path.append('/cs/labs/nati/eloyfer/projects/multivariate-krawchouks/')

from dual_sol_experiment import DualSolExperiment

DualSolExperiment(14, 8, 1, {1: [1,2,3,4]})

