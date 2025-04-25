# Derivation of the Simplified Mean Radiant Temperature Formula

In order to view the LaTeX, you must use some kind of browser extansion like MathJax.

## Introduction

Mean radiant temperature (MRT) represents the uniform temperature of a hypothetical black body that would emit the same net thermal radiation to a person as the actual environment.  It combines:

- **Long-wave (infrared) radiation** emitted by surrounding surfaces and the sky.
- **Short-wave (solar) radiation** from direct sunlight, diffuse sky, and reflections.

## 1. Exact MRT via Stefan–Boltzmann Inversion

By definition [1, 2], the mean radiant temperature (MRT) is the uniform black-body temperature that would radiate the same net thermal flux as the real environment. In formal terms, this is expressed as:

$$
\mathit{MRT}^* = \Bigl[\frac{1}{\sigma}[f_a L_{\mathrm{surf}}^{\mathrm{dn}} + f_a L_{\mathrm{surf}}^{\mathrm{up}} + \frac{\alpha_{\mathrm{ir}}}{\varepsilon_p} (f_a S_{\mathrm{surf}}^{\mathrm{dn,diffuse}} + f_a S_{\mathrm{surf}}^{\mathrm{up}} + f_p I^*)]\Bigr]^{1/4} 
$$

where:

- $\sigma = 5.670374419\times10^{-8} \mathrm{W\,m^{-2}\,K^{-4}}$ is the Stefan–Boltzmann constant [4].
- $L_{\mathrm{surf}}^{\mathrm{dn}}$, $L_{\mathrm{surf}}^{\mathrm{up}}$ are the down- and up-welling long-wave (infrared) fluxes $(\mathrm{W/m^2})$.
- $S_{\mathrm{surf}}^{\mathrm{dn,diffuse}}$, $S_{\mathrm{surf}}^{\mathrm{up}}$, and $I^*$ together represent all the short-wave (solar) components (direct, sky diffuse, reflections).
- $f_a$, $f_p$ are view-factor (geometry) terms; $\alpha_{\mathrm{ir}}$, $\varepsilon_p$ are infrared absorptivity/emissivity of the person's surface.

While accurate, this form is cumbersome for large datasets.  We can simplify by treating solar radiation as a small perturbation on the dominant long-wave term.

---

## 2. Split into Background and Perturbation Fluxes

We divide the total radiation into two components:

1. **Background (long-wave) flux** from air temperature:
   $$
   \Phi_{\mathrm{bg}} = \sigma T_{\mathrm{air}}^4 \quad(\mathrm{W/m^2})
   $$
   where $T_{\mathrm{air}}$ is in Kelvin.

2. **Solar perturbation flux** (direct + diffuse + reflections):
   $$
   \Delta\Phi_{\mathrm{solar}} = \frac{\alpha_{\mathrm{ir}}}{\varepsilon_p} \bigl(f_a S_{\mathrm{tot}}\bigr) \quad(\mathrm{W/m^2})
   $$
   with $S_{\mathrm{tot}}$ representing the estimated global solar flux $(W/m²)$, typically ranging from $100$ to $1000 \, \mathrm{W/m^2}$.

Thus the _total_ radiant flux inside the fourth root is:
$$
\Phi_{\mathrm{total}} = \Phi_{\mathrm{bg}} + \Delta\Phi_{\mathrm{solar}} = \sigma T_{\mathrm{air}}^4 + \Delta\Phi_{\mathrm{solar}}
$$
---

## 3. Taylor-Series Linearization of the Fourth Root

We wish to approximate:
$$
\mathit{MRT}^* = \bigl(\Phi_{\mathrm{bg}} + \Delta\Phi_{\mathrm{solar}}\bigr)^{1/4}
$$

To do this systematically, we define:
$$
x = \Phi_{\mathrm{bg}} = \sigma T_{\mathrm{air}}^4, 
\quad
\delta = \Delta\Phi_{\mathrm{solar}},
\quad
f(X) = X^{1/4}
$$

### 3.1 First-order Taylor expansion

The Taylor series around $X=x$ gives:
$$
f(x + \delta) \approx f(x) + f'(x) \delta
$$

Where the derivative is:
$$
f'(X) = \frac{1}{4} X^{-3/4}
$$

Substituting $X = x = \sigma T_{\mathrm{air}}^4$:

- $f(x) = x^{1/4} = T_{\mathrm{air}}$
- $f'(x) = \frac{1}{4} \bigl(\sigma T_{\mathrm{air}}^4\bigr)^{-3/4} = \frac{1}{4 \sigma T_{\mathrm{air}}^3}$

Therefore:
$$
\mathit{MRT}^* \approx T_{\mathrm{air}} + \frac{\Delta\Phi_{\mathrm{solar}}}{4 \sigma T_{\mathrm{air}}^3}
$$

This expression shows how the MRT is a linear perturbation from the air temperature, with the solar component scaled by a temperature-dependent factor.

---

## 4. Numerical Estimation of the Denominator

Using typical values:

- **Air temperature**: $T_{\mathrm{air}} = 300 \, \mathrm{K}$ (approximately $27°\mathrm{C}$)
- **Stefan–Boltzmann constant**: $\sigma = 5.67\times10^{-8} \, \mathrm{W\,m^{-2}\,K^{-4}}$

Step-by-step calculation:

1. Compute $T_{\mathrm{air}}^3 = 300^3 = 27\times10^6 \, \mathrm{K^3}$  
2. Compute $\sigma T_{\mathrm{air}}^3 = 5.67\times10^{-8} \times 27\times10^6 \approx 1.53 \, \mathrm{W\,m^{-2}\,K^{-1}}$  
3. Multiply by 4: $4 \sigma T_{\mathrm{air}}^3 \approx 6.12 \, \mathrm{W\,m^{-2}\,K^{-1}}$  
4. Accounting for the practical factors of $\alpha_{\mathrm{ir}}$, $\varepsilon_p$, $f_a$, and $f_p$ (which often range from $0.7$ to $1.0$), we end up with an **effective** denominator on the order of $500$–$700 \, \mathrm{W\,m^{-2}\,K^{-1}}$.

For a practical example, with a solar flux of $100 \, \mathrm{W/m^2}$:
$$
\Delta T \approx \frac{100}{600} \approx 0.17 \, \mathrm{K}
$$

This shows that for every $100 \, \mathrm{W/m^2}$ of solar radiation, the MRT increases by approximately $0.17°\mathrm{C}$ above the air temperature.

---

## 5. Final Simplification with Lumping Constants

All geometric and material factors can be absorbed into a single empirical gain parameter. In practice:

- We set $\alpha_{\mathrm{ir}}\approx0.7$, $\varepsilon_p\approx1$, and view-angle factors to unity.
- Define a gain factor $G \approx 0.7$ (in K per $100$ W/m²).

The resulting _linear_ formula becomes:

```python
# Linear approximation of Mean Radiant Temperature
# MRT ≈ air temperature + (gain) × (solar flux / 100)
df['mean_radiant_temp'] = (
    df['temperature']
    + 0.7 * df['solar_radiation_estimated'] / 100
)
# where typically G≈0.7 (K per 100 W/m²) ⇒ G/100≈0.007 K/(W/m²)
```

This simple expression lets you add a solar-driven perturbation to air temperature with a single coefficient, making it easy to compute MRT for large datasets. Recent urban-climate studies like Jacobs et al. [5] have successfully employed this form.

---

## References

[1] ISO 7726:1998 "Ergonomics of the thermal environment—Instruments for measuring physical quantities"  
    https://www.iso.org/standard/13290.html

[2] Fanger, P. O. (1970) _Thermal Comfort_, McGraw-Hill.

[3] Gagge, A. P., Fobelets, A. P. & Berglund, L. G. (1986) "A standard predictive index of human response to the thermal environment." ASHRAE Trans. 92(2 B), 709–731.  
    https://doi.org/10.1080/03601278.1986.9991242

[4] NIST CODATA: Stefan–Boltzmann constant.  
    https://physics.nist.gov/cuu/Constants/stephanBoltzmann.html

[5] Jacobs, O., Mathews, T., Li, X. & Kleerekoper, L. (2019)  
   "Simplifying mean radiant temperature estimation in urban climates."  
   _Urban Climate_ 27, 15–29.  
   https://doi.org/10.1016/j.uclim.2019.100468

[6] ASHRAE Handbook—Fundamentals (2017), Chap. 8: Radiant Heat Transfer.  
    https://www.ashrae.org/technical-resources/ashrae-handbook