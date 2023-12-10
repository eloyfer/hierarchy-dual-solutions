

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
