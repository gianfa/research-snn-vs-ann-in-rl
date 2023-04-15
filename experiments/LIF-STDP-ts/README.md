# Classification of time series through LIFs with STDP


exp | file| . |
--  | --  | --|
--  | -     

### TODO
- [] explicit the sampling (and subsamplig) freq
- [] gridsearch?
- [] memory/cpu artifact in experimentkit

signal_length = 100 * 40
fragment_length = 40 # 1e-1s

n_samples ~ freq
ext loop (n_samples) <- by sampling points

subloop()


### Signal definition
Given an array of phases $\Phi : \Phi \equiv \{ \omega_1, \omega_2, \omega_3, \omega_4 \}$   
Let $s$ be an array of signals, $s:s \equiv \{s_1, ..., s_{n_{signals}}\}$,  
 where each $s_z \in s$ is a combination of two time functions, with a pair $\omega_i, \omega_j \in \Phi$ as parameters.  
$s_p[n] \equiv (1 + Cos(t \omega_i) ) * (1 + Sin(t \omega_j))$, with $p \in \{1, n_{signals}\}$


We want to find a model $f(z; W)$ so that $f(z;W)|z_q = (\omega_i, \omega_j)$

$z_q \equiv s_z[n]; t_1 ≤ n ≤ t_2$


TODELETE:
ISSUE:
attualmente:
- la rete restituisce solo il tracciato dell'ouput layer, spk_out: dt x output
- l'stdp si aspetta:
  - raster: n_neurons x dt
  - weights: matrice (ideale 2 layers)
    - ideal: map layers <-> weights