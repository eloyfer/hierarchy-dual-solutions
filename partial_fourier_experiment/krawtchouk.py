
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
        

    
    @staticmethod
    def convolve(n, ell, func1, func2):
        config_set = list(index_set_generator(n,2**ell))
        assert len(func1) == len(func2)
        assert len(func1) == len(config_set)
        # TODO: complete this function
