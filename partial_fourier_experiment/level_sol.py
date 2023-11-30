
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
#         func_fourier = [x // norm_fac for x in func_fourier]
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
        gamma_sqr = [(sum(col), len(col)) for col in zip(*K_rows)]
        assert all(x % y == 0 for x,y in gamma_sqr)
        gamma_sqr = [(x//y)**2 for x,y in gamma_sqr]
#         gamma_sqr = [Fraction(sum(col), len(col))**2 for col in zip(*K_rows)]
        
        # symmetrize
        
        gamma_sqr_on_robit = {
            a0: (sum(gamma_sqr[self.kraw.config2index(a)] for a in orbit), len(orbit))
            for a0, orbit in self.kraw.orbit_map.items()
        }
        assert all(x % y == 0 for x,y in gamma_sqr_on_robit.values())
        gamma_sqr_on_robit = {
            a0: x//y for a0, (x,y) in gamma_sqr_on_robit.items()
        }
        
#         gamma_sqr_on_robit = {
#             a0: Fraction(sum(gamma_sqr[self.kraw.config2index(a)] for a in orbit), len(orbit))
#             for a0, orbit in self.kraw.orbit_map.items()
#         }
        
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
