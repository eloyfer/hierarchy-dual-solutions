
import functools
from fractions import Fraction

class TotalSol:
    """
    An object that stores the full dual solution, including the functions
    and the coefficients.
    """
    
    def __init__(self, n, max_lvl):
        self.n = n
        self.max_lvl = max_lvl
        self.lambdas = {}
        self.lvl_sols = {}
        self.data = []
    
    def get_copy(self):
        self_copy = TotalSol(self.n, self.max_lvl)
        self_copy.lambdas = self.lambdas.copy()
        self_copy.lvl_sols = self.lvl_sols.copy()
        return self_copy
    
    def add_level(self, lvl, lvl_sol):
        assert lvl_sol.lvl == lvl
        assert lvl not in self.lvl_sols
        assert lvl == self.max_lvl or lvl == min(self.lvl_sols)-1
        
        self.lvl_sols[lvl] = lvl_sol
        self.lambdas[lvl] = self.compute_lambda(lvl)

    @staticmethod
    def count_subspaces(dim1, dim2, dim3):
        """
        Count subspaces j
        """
        pass
    
    def compute_lambda(self, lvl):
        assert lvl in self.lvl_sols
        
        if lvl == self.max_lvl:
            return 1
        else:
            valid_region = self.lvl_sols[lvl].get_valid_region()

            config_sets = {
                j: self.lvl_sols[j].get_config_set()
                for j in range(lvl, self.max_lvl+1)
            }

            
            def prod(x,y):
                return x*y
            
            values = {}
            lvl_coeffs = {
                j: Fraction(
                    functools.reduce(prod, [2**(lvl+i) - 1 for i in range(1,j-lvl+1)]),
                    functools.reduce(prod, [2**i - 1 for i in range(1,j-lvl+1)])
                )
                for j in range(lvl+1,self.max_lvl+1)
            }
            for config in valid_region:
                cur_val = 0
                for j in range(lvl+1,self.max_lvl+1):
                    config_j = config + (0,) * (2**j - 2**lvl)
                    index_j = config_sets[j].index(config_j)
                    func_val = self.lvl_sols[j].sol[index_j]
                    cur_val += lvl_coeffs[j] * self.lambdas[j] * func_val
                coeff = functools.reduce(prod, [2**(lvl+i) - 1 for i in range(1,self.max_lvl-lvl+1)])
                denom = functools.reduce(prod, [2**i - 1 for i in range(1,self.max_lvl-lvl+1)])
                cur_val += Fraction(coeff, denom)

                index_lvl = config_sets[lvl].index(config)
                cur_val *= Fraction(1, -self.lvl_sols[lvl].sol[index_lvl])

                values[config] = cur_val

            return max(values.values())
    
    def compute_value(self):
        return 1 + sum(self.lambdas[j] * self.lvl_sols[j].sol[0] for j in self.lambdas.keys())

    def __repr__(self):
        return 'TotalSol(' + ','.join([repr(self.lvl_sols[lvl]) for lvl in range(1,self.max_lvl+1)]) + ')'
