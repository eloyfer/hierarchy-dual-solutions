
from tqdm import tqdm
from level_sol import LevelSol

class SolutionFactory:
    """
    Create level-solutions (LevelSol objects) with parameters n,d,lvl,m
    """
    
    def __init__(self, n, d, lvl, m, kraw):
        self.n = n
        self.d = d
        self.lvl = lvl
        self.m = m
        self.kraw = kraw
        
        self.valid_region = self.compute_valid_region()
        self.phi = self.compute_phi()
    
    def compute_phi(self):
        K = self.kraw
        K1 = K.get_slice_rows(1)
        num_configs = len(K.get_config_set())
        def phi_func_eval(index):
            if self.m % 2 == 0:
                return sum((K1[i][index] + self.d)**self.m - (self.n - self.d)**self.m for i in range(len(K1)))
            else:
                return sum(K1[i][index]**self.m - (self.n - 2 * self.d)**self.m for i in range(len(K1)))
        phi = [phi_func_eval(j) for j in range(num_configs)]
        
        assert self.check_nonpos_constraint(phi)
        
        return phi
    
    def check_nonpos_constraint(self, func):
        """
        Verify the func is non-positive in the valid region
        """
        return all([func[self.kraw.config2index(a)] <= 0 for a in self.valid_region])

    def compute_valid_region(self):
        K = self.kraw
        K1 = K.get_slice_rows(1)
        
        def filter_func(idx):
            return all(K1[row][idx] <= self.n-2*self.d for row in range(len(K1)))
        
        config_set = K.get_config_set()
        valid_region = [config for idx,config in enumerate(config_set) if filter_func(idx)]
        return valid_region
    
    def get_sol(self, configs):
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
            n=self.n,
            d=self.d,
            lvl=self.lvl,
            m=self.m,
            configs=configs,
            phi=self.phi, 
            kraw=self.kraw, 
            valid_region=self.valid_region
        )
        return sol
    
    def get_all_level_sols(self):
        print(f'{self} creating all sols')
        K = self.kraw
        singles = [
            self.get_sol([config]) 
            for config in tqdm(K.get_orbit_list(), desc='singles')]
        orbits = [
            self.get_sol(K.orbit_map[config]) 
            for config in tqdm(K.get_orbit_list(), desc='orbits')]
        return singles + orbits
    
    def __repr__(self):
        return f'SolutionFactory(n={self.n}, d={self.d}, lvl={self.lvl}, m={self.m})'