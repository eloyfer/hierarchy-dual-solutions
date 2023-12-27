
import math
from fractions import Fraction
import itertools

class LevelSol:
    
    def __init__(self, lvl, n, d, configs, phi, K=None):
        
        assert lvl == int(math.log2(len(configs[0])))
        assert n == sum(configs[0])
        
        self.lvl = lvl
        self.n = n
        self.d = d
        self.configs = configs
        self.phi = phi

        if K is None:
            from krawtchouk_getter import get_krawtchouk
            K = get_krawchouk(n, lvl)

        self.init_sol(K)

    def get_valid_region(self):
        return self.phi.valid_region

    def get_config_set(self):
        return self.phi.config_set

    def init_sol(self, K):
        gamma_sqr = self.get_gamma_squared(K)
        raw_sol = [x * y for x,y in zip(self.phi, gamma_sqr)]
        self.raw_sol = raw_sol
        
        # compute the Fourier transform by multiplying with the Krawtchouk matrix
        func_fourier = K.matmul([[x] for x in raw_sol], transposed=True)

        # flatten
        func_fourier = list(itertools.chain(*func_fourier))

        # divide by 2^(ell*n)
        norm_fac = 2**(self.lvl * self.n)
        func_fourier = [Fraction(x, norm_fac) for x in func_fourier]
        self.raw_sol_fourier = func_fourier
        
        # keep useful quantities
        self.max_in_valid_region = max(self.raw_sol[K.config2index(a)] for a in self.get_valid_region())
        self.min_fourier_val = min(func_fourier)
        
        # compute the actual solution
        if self.raw_sol_fourier[0] > 0:
            denom = self.raw_sol_fourier[0]
            self.sol = [Fraction(x, denom) - 1 for x in self.raw_sol]
        
    
    def get_gamma_squared(self, K):
        K_rows = [K.get_row(a) for a in self.configs]
        gamma_sqr = [(sum(col), len(col)) for col in zip(*K_rows)]
        assert all(x % y == 0 for x,y in gamma_sqr)
        gamma_sqr = [(x//y)**2 for x,y in gamma_sqr]
#         gamma_sqr = [Fraction(sum(col), len(col))**2 for col in zip(*K_rows)]
        
        # symmetrize
        
        gamma_sqr_on_robit = {
            a0: (sum(gamma_sqr[K.config2index(a)] for a in orbit), len(orbit))
            for a0, orbit in K.orbit_map.items()
        }
        assert all(x % y == 0 for x,y in gamma_sqr_on_robit.values())
        gamma_sqr_on_robit = {
            a0: x//y for a0, (x,y) in gamma_sqr_on_robit.items()
        }
        
        gamma_sqr = [
            gamma_sqr_on_robit[K.config_to_orbit[a]] 
            for a in K.config_set
        ]
        
        return gamma_sqr
        
    
    def is_feasible(self):
        fourier_feasible = self.min_fourier_val >= 0
        valid_region_feasible = self.max_in_valid_region <= 0
        
        # this one is not strictly necessary but is convenient
        non_zero_fourier_at_0 = self.raw_sol_fourier[0] > 0
        
        return fourier_feasible and valid_region_feasible and non_zero_fourier_at_0
    
    def __repr__(self):
        return f'LevelSol(n={self.n},lvl={self.lvl},d={self.d},m={self.m},configs={self.configs})'
