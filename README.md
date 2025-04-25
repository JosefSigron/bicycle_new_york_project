# Derivation of the Simplified Mean Radiant Temperature Formula

This document shows how the full, nonlinear mean‐radiant‐temperature equation

<img src="https://latex.codecogs.com/svg.image?\;\mathit{MRT}^*=\Biggl\{\frac{1}{\sigma}\Bigl[f_a&space;L_{\mathrm{surf}}^{\mathrm{dn}}&plus;f_a&space;L_{\mathrm{surf}}^{\mathrm{up}}&plus;\frac{\alpha_{\mathrm{ir}}}{\varepsilon_p}\bigl(f_a&space;S_{\mathrm{surf}}^{\mathrm{dn,diffuse}}&plus;f_a&space;S_{\mathrm{surf}}^{\mathrm{up}}&plus;f_p&space;I^*\bigr)\Bigr]\Biggr\}^{0.25}\" title="\;\mathit{MRT}^*=\Biggl\{\frac{1}{\sigma}\Bigl[f_a L_{\mathrm{surf}}^{\mathrm{dn}}+f_a L_{\mathrm{surf}}^{\mathrm{up}}+\frac{\alpha_{\mathrm{ir}}}{\varepsilon_p}\bigl(f_a S_{\mathrm{surf}}^{\mathrm{dn,diffuse}}+f_a S_{\mathrm{surf}}^{\mathrm{up}}+f_p I^*\bigr)\Bigr]\Biggr\}^{0.25}\" />



can be approximated by

```python
# in `estimate_solar_radiation.py`
df['mean_radiant_temp'] = df['temperature'] + 0.7 * df['solar_radiation_estimated'] / 100
```

---

## 1. Identify the small perturbation

We split the full formula into:

1.  A **background** long‐wave term, which we approximate by the air temperature
<img src="https://latex.codecogs.com/svg.image?\(T_{\mathrm{air}}\)" title="\;\(T_{\mathrm{air}}\)" />.
2.  A **solar** perturbation term (direct + diffuse + reflection).

Write the total flux inside the <img src="https://latex.codecogs.com/svg.image?\(4_{th}\)" title="\(4_{th}\)" /> -root as

<img src="https://latex.codecogs.com/svg.image?\[\Phi_{\mathrm{bg}}&plus;\Delta\Phi_{\mathrm{solar}}\]" title="\[\Phi_{\mathrm{bg}}+\Delta\Phi_{\mathrm{solar}}\]" />,

where

-   <img src="https://latex.codecogs.com/svg.image?\(\Phi_{\mathrm{bg}}=\sigma\,T_{\mathrm{air}}^4\)" title="\(\Phi_{\mathrm{bg}}=\sigma\,T_{\mathrm{air}}^4\)" />
-   <img src="https://latex.codecogs.com/svg.image?\(\Delta\Phi_{\mathrm{solar}}=\dfrac{\alpha_{\mathrm{ir}}}{\varepsilon_p}\bigl(f_a&space;S_{\mathrm{tot}}\bigr)\)" title="\(\Delta\Phi_{\mathrm{solar}}=\dfrac{\alpha_{\mathrm{ir}}}{\varepsilon_p}\bigl(f_a S_{\mathrm{tot}}\bigr)\)" />

and

<img src="https://latex.codecogs.com/svg.image?\[S_{\mathrm{tot}}=S_{\mathrm{surf}}^{\mathrm{dn,diffuse}}&plus;S_{\mathrm{surf}}^{\mathrm{up}}&plus;\frac{f_p}{f_a}\,I^*\approx\text{(estimated&space;global&space;solar&space;flux)}.\]" title="\[S_{\mathrm{tot}}=S_{\mathrm{surf}}^{\mathrm{dn,diffuse}}+S_{\mathrm{surf}}^{\mathrm{up}}+\frac{f_p}{f_a}\,I^*\approx\text{(estimated global solar flux)}\]" /> .

In code,```{solar_radiation_estimated}```stands in for <img src="https://latex.codecogs.com/svg.image?\(S_{\mathrm{tot}}\)" title="\(S_{\mathrm{tot}}\)" />.

## 2. Linearize the <img src="https://latex.codecogs.com/svg.image?\(4_{th}\)" title="\(4_{th}\)" /> -root

We have

<img src="https://latex.codecogs.com/svg.image?\[\mathit{MRT}^*=\bigl(\Phi_{\mathrm{bg}}&plus;\Delta\Phi_{\mathrm{solar}}\bigr)^{1/4}=\bigl(\sigma\,T_{\mathrm{air}}^4&plus;\Delta\Phi\bigr)^{1/4}.\]" title="\[\mathit{MRT}^*=\bigl(\Phi_{\mathrm{bg}}+\Delta\Phi_{\mathrm{solar}}\bigr)^{1/4}=\bigl(\sigma\,T_{\mathrm{air}}^4+\Delta\Phi\bigr)^{1/4}.\]" />

For <img src="https://latex.codecogs.com/svg.image?\(\Delta\Phi\ll\sigma&space;T_{\mathrm{air}}^4\)" title="\(\Delta\Phi\ll\sigma T_{\mathrm{air}}^4\)" />, use the Taylor expansion about <img src="https://latex.codecogs.com/svg.image?\(T_{\mathrm{air}}\)" title="\(T_{\mathrm{air}}\)" />:

<img src="https://latex.codecogs.com/svg.image?\[(x&plus;\delta)^{1/4}\approx&space;x^{1/4}&plus;\tfrac{1}{4}x^{-3/4}\,\delta\quad\text{with}x=\sigma&space;T_{\mathrm{air}}^4,\\delta=\Delta\Phi.\]" title="\[(x+\delta)^{1/4}\approx x^{1/4}+\tfrac{1}{4}x^{-3/4}\,\delta\quad\text{with}x=\sigma T_{\mathrm{air}}^4,\\delta=\Delta\Phi.\]" />

Thus

<img src="https://latex.codecogs.com/svg.image?\[\mathit{MRT}^*\approx&space;T_{\mathrm{air}}&plus;\frac{\Delta\Phi}{4\,\sigma\,T_{\mathrm{air}}^3}.\]" title="\[\mathit{MRT}^*\approx T_{\mathrm{air}}+\frac{\Delta\Phi}{4\,\sigma\,T_{\mathrm{air}}^3}.\]" />

## 3. Plug in typical values

-  Stefan–Boltzmann constant<img src="https://latex.codecogs.com/svg.image?\(\sigma=5.67\times10^{-8}\rm\;W/m^2/K^4\)" title="\(\sigma=5.67\times10^{-8}\rm\;W/m^2/K^4\)" />
-  Typical air temperature: <img src="https://latex.codecogs.com/svg.image?\(T_{\mathrm{air}}\approx300\rm\;K\)" title="\(T_{\mathrm{air}}\approx300\rm\;K\)" />

Compute the denominator:

<img src="https://latex.codecogs.com/svg.image?\[4\,\sigma\,T_{\mathrm{air}}^3=4\times5.67\times10^{-8}\times(300)^3\approx600\;\rm&space;W/m^2/K.\]" title="\[4\,\sigma\,T_{\mathrm{air}}^3=4\times5.67\times10^{-8}\times(300)^3\approx600\;\rm W/m^2/K.\]" />

So a flux perturbation of <img src="https://latex.codecogs.com/svg.image?\(100\rm\;W/m^2\)" title="\(100\rm\;W/m^2\)" /> changes temperature by

<img src="https://latex.codecogs.com/svg.image?\[\Delta&space;T\approx\frac{100}{600}\approx0.17\;\rm&space;K.\]" title="\[\Delta T\approx\frac{100}{600}\approx0.17\;\rm K.\]" />

## 4. Incorporate absorptivity and angle factors

The solar perturbation was

<img src="https://latex.codecogs.com/svg.image?\[\Delta\Phi=\frac{\alpha_{\mathrm{ir}}}{\varepsilon_p}\,f_a\,S_{\mathrm{tot}}.\]" title="\[\Delta\Phi=\frac{\alpha_{\mathrm{ir}}}{\varepsilon_p}\,f_a\,S_{\mathrm{tot}}.\]" />

Using the **rule of thumb** that all the geometric factors and emissivity are lumped into one constant, we set:

-   <img src="https://latex.codecogs.com/svg.image?\(\alpha_{\mathrm{ir}}\approx0.7\)" title="\(\alpha_{\mathrm{ir}}\approx0.7\)" />
-   <img src="https://latex.codecogs.com/svg.image?\(\varepsilon_p\approx1\)" title="\(\varepsilon_p\approx1\)" />
-   <img src="https://latex.codecogs.com/svg.image?\(f_a\approx1\)" title="\(f_a\approx1\)" />
-   <img src="https://latex.codecogs.com/svg.image?\(4\,\sigma\,T_{\mathrm{air}}^3\approx600\rm\;W/m^2/K\to100\rm\;W/m^2/K\)(*cruder*round&space;off)" title="\(4\,\sigma\,T_{\mathrm{air}}^3\approx600\rm\;W/m^2/K\to100\rm\;W/m^2/K\)(*cruder*round off)" />

so

<img src="https://latex.codecogs.com/svg.image?\[\Delta&space;T\approx\frac{\alpha_{\mathrm{ir}}\,S_{\mathrm{tot}}}{100}=0.7\times\frac{S_{\mathrm{tot}}}{100}.\]" title="\[\Delta T\approx\frac{\alpha_{\mathrm{ir}}\,S_{\mathrm{tot}}}{100}=0.7\times\frac{S_{\mathrm{tot}}}{100}.\]" />

Switching back to code:

```python
# MRT ≈ T_air + 0.7 × (solar_flux / 100)
df['mean_radiant_temp'] = df['temperature'] + 0.7 * df['solar_radiation_estimated'] / 100
```

---

In this way, we go from the full <img src="https://latex.codecogs.com/svg.image?\(T^{1/4}\)" title="\(T^{1/4}\)" /> Stefan–Boltzmann inversion to a simple linear "knock‐in" of solar radiation into air temperature, with a single gain factor <img src="https://latex.codecogs.com/svg.image?\(0.7/100\)" title="\(0.7/100\)" />. 