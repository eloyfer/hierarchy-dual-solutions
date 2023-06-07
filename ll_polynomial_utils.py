import numpy as np

def u2conf(u,n,ell):
    conf = [n-1] + [0]*(2**ell-1)
    conf[u] = 1
    return tuple(conf)

def get_Phi(n,ell,d,m,K_mat,config_set):
    """
    Return the function $\Phi = \prod_{v\neq 0} (\sum_{<u,v>=1} K_u^m - (n-2d)^m )$
    """
    V = np.arange(1,2**ell)
    U = {
        v: [u for u in range(1,2**ell) if bin(u&v).count('1') % 2 == 1]
        for v in V
    }
    U_indices = {
        v: [config_set.index(u2conf(u,n,ell)) for u in U[v]]
        for v in V
    }
    K_mat = K_mat.astype(np.float32)
    phi_parts = [((K_mat[U_indices[v]]+d)**m).sum(axis=0) - 2**(ell-1)*(n-d)**m for v in V]
    Phi = np.prod(phi_parts, axis=0)
    assert check_valid_regions(func,n,d,ell,config_set,K_mat)
    return Phi

def get_Phi_nonlinear(n,ell,d,m,K_mat,config_set):
    """
    Return the function $\Phi = \prod_{v\neq 0} (\sum_{<u,v>=1} K_u^m - (n-2d)^m )$
    """
    V = np.arange(1,2**ell)
    U = {
        v: [1<<u for u in range(ell) if ((v>>u) & 1) == 1]
        for v in V
    }
    U_indices = {
        v: [config_set.index(u2conf(u,n,ell)) for u in U[v]]
        for v in V
    }
    K_mat = K_mat.astype(np.float32)
    phi_parts = [((K_mat[U_indices[v]]+d)**m).sum(axis=0) - 2**(ell-1)*(n-d)**m for v in V]
    Phi = np.prod(phi_parts, axis=0)
    check_valid_regions_nonlinear(Phi,n,d,ell,config_set,K_mat)
    return Phi

def get_config_weights(config):
    n = sum(config)
    return (n - fwht(config))/2

def get_valid_regions(n,d,ell,config_set,K_mat):
    """
    get configurations which are valid, meaning if $X \in conf$ then the weights
    of the rowspan of X are all >=d or 0
    """
    K1s = K_mat[[config_set.index(u2conf(u,n,ell)) for u in range(1,2**ell)]]
    config_weights = (n-K1s)/2
    valid = (config_weights >= d) | (config_weights == 0)
    valid = valid.all(axis=0)
    valid_config_idx = np.nonzero(valid)[0][1:]
    return [config_set[i] for i in valid_config_idx]

def check_valid_regions(func, n,d,ell,config_set,K_mat):
    valid_regions = get_valid_regions(n,d,ell,config_set,K_mat)
    func_in_regions = func[[config_set.index(conf) for conf in valid_regions]]
    return ((func_in_regions <= 0) | np.isclose(func_in_regions,0)).all()

def get_valid_regions_nonlinear(n,d,ell,config_set,K_mat):
    """
    get configurations which are valid, meaning if $X \in conf$ then the weights
    of the rowspan of X are all >=d or 0
    """
    K1s = K_mat[[config_set.index(u2conf(1<<u,n,ell)) for u in range(ell)]]
    config_weights = (n-K1s)/2
    valid = (config_weights >= d) | (config_weights == 0)
    valid = valid.all(axis=0)
    valid_config_idx = np.nonzero(valid)[0][1:]
    return [config_set[i] for i in valid_config_idx]

def check_valid_regions_nonlinear(func, n,d,ell,config_set,K_mat):
    valid_regions = get_valid_regions_nonlinear(n,d,ell,config_set,K_mat)
    func_in_regions = func[[config_set.index(conf) for conf in valid_regions]]
    return ((func_in_regions <= 0) | np.isclose(func_in_regions,0)).all()
