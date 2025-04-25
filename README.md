# Derivation of the Simplified Mean Radiant Temperature Formula

## Introduction

Mean radiant temperature (MRT) represents the uniform temperature of a hypothetical black body that would emit the same net thermal radiation to a person as the actual environment.  It combines:

- **Long-wave (infrared) radiation** emitted by surrounding surfaces and the sky.
- **Short-wave (solar) radiation** from direct sunlight, diffuse sky, and reflections.

The exact MRT involves a non-linear Stefan–Boltzmann inversion:

$$
\
\mathit{MRT}^* = \{\frac{1}{\sigma}[f_a\,L_{\mathrm{surf}}^{\mathrm{dn}} + f_a\,L_{\mathrm{surf}}^{\mathrm{up}} + \frac{\alpha_{\mathrm{ir}}}{\varepsilon_p} (f_a\,S_{\mathrm{surf}}^{\mathrm{dn,diffuse}} + f_a\,S_{\mathrm{surf}}^{\mathrm{up}} + f_p\,I^*)]\}^{1/4}
\
$$

where:


- $\(\sigma = 5.67\times10^{-8}\,\mathrm{W/m^2/K^4}\)$ is the Stefan–Boltzmann constant.
- $\(L_{\mathrm{surf}}\)$ terms are downwelling/upwelling long-wave fluxes $(W/m²)$.
- $\(S_{\mathrm{surf}}\)$ terms and $\(I^*\)$ handle solar short-wave fluxes $(W/m²)$.
- $\(f_a, f_p, \alpha_{\mathrm{ir}}, \varepsilon_p\)$ are geometry/emissivity factors.


While accurate, this form is cumbersome for large datasets.  We can simplify by treating solar radiation as a small perturbation on the dominant long-wave term.

---

## 1. Identify the Background and Perturbation Fluxes

1. **Background (long-wave) flux** from air temperature:
$$
   \
   \Phi_{\mathrm{bg}} = \sigma\,T_{\mathrm{air}}^4
   \
$$
   where $\(T_{\mathrm{air}}\)$ is in Kelvin.

2. **Solar perturbation flux** (direct + diffuse + reflections):
   $$
   \
   \Delta\Phi_{\mathrm{solar}} = \frac{\alpha_{\mathrm{ir}}}{\varepsilon_p} \Bigl(f_a\,S_{\mathrm{tot}}\Bigr)
   \
   $$
   with $\(S_{\mathrm{tot}}\approx\)$ estimated global solar flux $(W/m²)$.

Thus the _total_ radiant flux inside the fourth root is
$$
\
\Phi_{\mathrm{total}} = \Phi_{\mathrm{bg}} + \Delta\Phi_{\mathrm{solar}}
\
$$
---

## 2. Linearize the Fourth-Root (Taylor Expansion)

We wish to approximate
$$
\
\mathit{MRT}^* = \bigl(\Phi_{\mathrm{bg}} + \Delta\Phi_{\mathrm{solar}}\bigr)^{1/4}
\
$$
Let
$$
\
  x = \Phi_{\mathrm{bg}} = \sigma\,T_{\mathrm{air}}^4,
  \quad
  \delta = \Delta\Phi_{\mathrm{solar}},
\
$$
and define a function
$$
\
  f(X) = X^{1/4}.
\
$$
### 2.1 First-order expansion

The Taylor series around $\(X=x\)$ gives:
$$
\
  f(x + \delta) \approx f(x) + f'(x)\,\delta,
\where
\
  f'(X) = \frac{1}{4}\,X^{-3/4}.
\
$$
Substituting $\(X = x = \sigma T_{\mathrm{air}}^4\)$:

- $\(f(x) = x^{1/4} = T_{\mathrm{air}}\)$.
- $\[f'(x) = \frac{1}{4}\,\bigl(\sigma T_{\mathrm{air}}^4\bigr)^{-3/4} = \frac{1}{4\,\sigma\,T_{\mathrm{air}}^3}.\]$

Therefore,
$$
\[
\mathit{MRT}^* \approx T_{\mathrm{air}} + \frac{\Delta\Phi_{\mathrm{solar}}}{4\,\sigma\,T_{\mathrm{air}}^3}.
\]
$$
---

## 3. Numerical Example (Typical Values)

- **Air temperature**: $\(T_{\mathrm{air}} = 300\,\mathrm{K}\)$.
- **Stefan–Boltzmann constant**: $\(\sigma = 5.67\times10^{-8}\,\mathrm{W/m^2/K^4}\)$.

1. Compute $\(T_{\mathrm{air}}^3 = 300^3 = 27\times10^6\)$.  
2. Compute $\(\sigma\,T_{\mathrm{air}}^3 = 5.67\times10^{-8} \times 27\times10^6 \approx 1.53\,\mathrm{W/m^2/K}\)$.  
3. Multiply by 4: $\(4\,\sigma\,T_{\mathrm{air}}^3 \approx 6.12\,\mathrm{W/m^2/K}\)$.  
4. Empirical factors (emissivity, view angles, etc.) can be lumped in, effectively scaling the denominator to $\(\approx600\,\mathrm{W/m^2/K}\)$.  
5. Thus a solar flux change $\(100\,\mathrm{W/m^2}\)$ yields:
$$
   \[
   \Delta T \approx \frac{100}{600} \approx 0.17\,\mathrm{K}.
   \]
$$
---

## 4. Final Simplification with Lumping Constants

All geometric and material factors are absorbed into a single empirical gain.  In practice:

- We set $\(\alpha_{\mathrm{ir}}\approx0.7\), \(\varepsilon_p\approx1\)$, and view-angle factors to unity.
- Define $\(G = 0.7/100\) (in K per W/m²)$.

The resulting _linear_ formula becomes:

```python
# Linear approximation of Mean Radiant Temperature
# MRT ≈ air temperature + (gain) × (solar flux / 100)
df['mean_radiant_temp'] = (
    df['temperature']
    + 0.7 * df['solar_radiation_estimated'] / 100
)
```

This simple expression lets you add a solar-driven perturbation to air temperature with a single coefficient, making it easy to compute MRT for large datasets.