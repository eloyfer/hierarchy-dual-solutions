# LL-poly-solutions
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
