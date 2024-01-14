# Classification of time series through LIFs with STDP


### TODO

- signal_length = 100 * 40
- fragment_length = 40 # 1e-1s
- n_samples ~ freq
- external loop (n_samples) <- by sampling points

### Signal definition

Given an array of phases $\Phi : \Phi \equiv \{ \omega_1, \omega_2, \omega_3, \omega_4 \}$   
Let $s$ be an array of signals, $s:s \equiv \{s_1, ..., s_{n_{signals}}\}$,  
 where each $s_p \in s$ is a combination of two time functions, with a pair $\omega_i, \omega_j \in \Phi$ as parameters.  
$s_p[n] \equiv (1 + Cos(t \omega_i) ) * (1 + Sin(t \omega_j))$, with $p \in \{1, n_{signals}\}$


We want to find a model $f(z; W)$ so that $f(z=z_q;W) = (\omega_i, \omega_j)$

$z_q \equiv s_z[n]; \space t_1 ≤ n ≤ t_2$
