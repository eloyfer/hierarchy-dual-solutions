from functools import partial


def get_phi(name, params):
    if name == 'even':
        return PhiEven(**params) 
    elif name == 'odd':
        return PhiOdd(**params)
    elif name == 'balanced':
        return PhiBalanced(**params)
    else:
        raise ValueError(f'Unknown phi function: {name}')

class PhiFuncBase:

    def __init__(self, n, d, lvl, K=None):
        self.n = n
        self.d = d
        self.lvl = lvl
        if K is None:
            from krawtchouk_getter import get_krawtchouk
            K = get_krawchouk(n, lvl)

        self.config_set = K.get_config_set()
        self.valid_region = self.get_valid_region(K)
        self.data = self.compute(K)

    def get_valid_region(self, K):
        K = self.get_krawtchouk()
        K1 = K.get_slice_rows(1)
        def filter_func(idx):
            return all(K1[row][idx] <= self.n-2*self.d for row in range(len(K1)))
        
        config_set = K.get_config_set()
        valid_region = [config for idx,config in enumerate(config_set) if filter_func(idx)]
        return valid_region

    def compute(self,K):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError
    
    def name(self):
        raise NotImplementedError

    def params_str(self):
        raise NotImplementedError

    def __getitem__(self,index):
        return self.data[index]

    def __call__(self,config):
        return self.data[self.config_set.index(config)]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


class PhiEven(PhiFuncBase):

    def __init__(self, m, *args, shift=0, **kwargs):
        assert m % 2 == 0
        self.m = m
        self.shift = shift
        super().__init__(*args, **kwargs)

    def compute(self, K):
        K1 = K.get_slice_rows(1)
        num_configs = len(K.get_config_set())
        def phi_func_eval(index):
            return sum((K1[i][index] + self.d + self.shift)**self.m - (self.n + self.shift - self.d)**self.m for i in range(len(K1)))
        phi = [phi_func_eval(j) for j in range(num_configs)]
        
        return phi

    def params_str(self):
        return f'm={m};shift={shift}'

    def __repr__(self):
        shift_str = f' + {self.shift}' if self.shift else ''
        func_str = f'$\\sum_{{u\\in (F_{{2}}^{self.n})^{self.lvl}}} (K_1(x) + {self.d}{shift_str})^{self.m} - ({self.n+self.shift} - {self.d})^{self.m}$'
        return func_str

    def name(self):
        return 'even'


class PhiOdd(PhiFuncBase):

    def __init__(self, m, *args, **kwargs):
        assert m % 2 == 1
        self.m = m
        super().__init__(*args, **kwargs)

    def compute(self, K):
        K1 = K.get_slice_rows(1)
        num_configs = len(K.get_config_set())
        def phi_func_eval(index):
            return sum(K1[i][index]**self.m - (self.n - 2 * self.d)**self.m for i in range(len(K1)))
        phi = [phi_func_eval(j) for j in range(num_configs)]
        
        return phi

    def params_str(self):
        return f'm={m}'

    def __repr__(self):
        func_str = f'$\\sum_{{u\\in (F_{{2}}^{self.n})^{self.lvl}}} (K_1(x))^{self.m} - ({self.n} - 2{self.d})^{self.m}$'
        return func_str

    def name(self):
        return 'odd'

class PhiBalanced(PhiOdd):
    """
    Phi for balanced codes.
    The computation is exactly the same as in PhiOdd, only with *even* m.
    Also the valid region is different.
    """

    def __init__(self, m, *args, **kwargs):
        assert m % 2 == 0
        self.m = m
        super(PhiOdd,self).__init__(*args, **kwargs)

    def get_valid_region(self, K):
        K1 = K.get_slice_rows(1)
        def filter_func(idx):
            weights = [(self.n - K1[row][idx])//2 for row in range(len(K1)) ]
            return all(w >= self.d and w <= (self.n - self.d) for w in weights)
        
        config_set = K.get_config_set()
        valid_region = [config for idx,config in enumerate(config_set) if filter_func(idx)]
        return valid_region

    def params_str(self):
        return f'm={m}'

    def name(self):
        return 'balanced'
