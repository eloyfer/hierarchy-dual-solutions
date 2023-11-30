

from krawtchouk import Krawtchouk
from solution_factory import SolutionFactory

class SolutionFactory2:
    
    def __init__(self, n, d):
        self.n = n
        self.d = d
        
        self.factories = {}
        self.kraw = {}
    
    def get_kraw(self, lvl):
        if lvl not in self.kraw:
            self.kraw[lvl] = Krawtchouk(self.n,lvl)
        return self.kraw[lvl]
    
    def get_factory(self, lvl, m):
        key = (lvl,m)
        if key not in self.factories:
            self.factories[key] = SolutionFactory(self.n, self.d, lvl, m, self.get_kraw(lvl))
        return self.factories[key]
    
    def get_all_level_sols(self, lvl, m):
        return self.get_factory(lvl,m).get_all_level_sols()