# Dual Solutions for the LP Hierarchy for Linear Codes

This repo is dedicated to finding dual soltuions to the LP hierarchy DelsartLin($n,d,\ell$).
Currently, the approach we explore is the Loyfer-Linial polynomial, see here:
https://arxiv.org/abs/2211.12977

## Reachability
Here we explore paths on the graphs $\{A_u\}_{u\in F_2^\ell}$. Let $\alpha:F_2^\ell\to N$
is a configuration, namely $\sum_{u}\alpha(u) = n$. Let $U \in (F_2^\ell)^k$ for some
integer $1\leq k\leq 2^\ell-1$. Let $m$ be a positive integer. Then.
$$
n^{-m k}(\prod_{u\in U} A_u^m) L_{\alpha}
$$
is a distribution over the configuration space. We seek to understand this 
distribution. First we ask what is the support. We say that $\alpha'$
is $(U,m)$-reachable from $\alpha$ if it is in the support of the above
distribution. Given $\alpha, U, m$, which $\alpha'$ is $(U,m)$-reachable?

Let $\Delta = \alpha'-\alpha$. Then, $\alpha'$ is reachable if the following
system of equations has a solution:
```math
\begin{align}
m k_{u} &= \sum_{v\in\mathbb{F}_2^\ell} s_{u,v}  & \forall u \in U, k_u = \{1\leq i\leq k: U_i=u\}
\\
\Delta(v) &= \sum_{u\in U} s_{u,u+v} - s_{u,v} & \forall v\in\mathbb{F}_2^\ell \setminus\{0\}
\end{align}
```
$s_{u,v}$ is the number of times the column vector $u$ is added to $v$.


## LL-poly-solutions
Search for dual feasible solutions for the DelsarteLin hierarchy based 
on the LL polynomial. The LL polynomial is described here:
https://arxiv.org/abs/2211.12977

The function $\Phi$ is defined by

$$
\Phi = \prod_{v\in F_2^\ell \setminus\{0\}} 
\left(
\sum_{\langle u,v\rangle = 1} (K_{1_u} + d)^m - (n-d)^m
\right)
$$

where $n,d,\ell,m$ are parameters such that 
- $d$ can be taken even,
$1\leq d\leq n$.
- $m$ is even and $m \geq \frac{\ell-1}{\log_2 \frac{n+d}{n-d}}$. In practice
we want it large, $m=\omega_n(1)$.
- Due to computational constraints, we always take $\ell=2$.

We seek a non-zero function $f:F_2^{\ell\times n}\to\{0,1\}$ such that 
$\widehat{\Phi} * f \geq f$, and the support size of $f$ is minimal.
In practice, since $\Phi$ is symmetric we 
optimize of $f:Config_{n,\ell}\to\{0,1\}$. The support, in this case is

$$
\sum_{\alpha \in Config_{n,\ell}} \binom{n}{\alpha} f(\alpha)
$$

To conclude, we solve the following ILP:
```math
\begin{align*}
  & \min && \sum_{\alpha \in Config_{n,\ell}} \binom{n}{\alpha} f(\alpha)
  \\
  & \text{s.t.} && f:Config_{n,\ell}\to\{0,1\}
  \\
  &&& \sum_{\alpha \in Config_{n,\ell}} f(\alpha) \geq 1
  \\
  &&& (M - I) f \geq 0
\end{align*}
```
where $M = K^\top \cdot diag(\Phi) \cdot K^\top$. $K$ is the Krawtchouk matrix.
