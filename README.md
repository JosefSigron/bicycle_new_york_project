# Derivation of the Simplified Mean Radiant Temperature Formula

This document shows how the full, nonlinear mean‐radiant‐temperature equation

<img src="https://latex.codecogs.com/svg.image?%5Ccolor%7Bwhite%7D%5C%3B%5Cmathit%7BMRT%7D%5E%2A%3D%5CBiggl%5C%7B%5Cfrac%7B1%7D%7B%5Csigma%7D%5CBigl%5Bf_a%20L_%7B%5Cmathrm%7Bsurf%7D%7D%5E%7B%5Cmathrm%7Bdn%7D%7D%2Bf_a%20L_%7B%5Cmathrm%7Bsurf%7D%7D%5E%7B%5Cmathrm%7Bup%7D%7D%2B%5Cfrac%7B%5Calpha_%7B%5Cmathrm%7Bir%7D%7D%7D%7B%5Cvarepsilon_p%7D%5Cbigl%28f_a%20S_%7B%5Cmathrm%7Bsurf%7D%7D%5E%7B%5Cmathrm%7Bdn%2Cdiffuse%7D%7D%2Bf_a%20S_%7B%5Cmathrm%7Bsurf%7D%7D%5E%7B%5Cmathrm%7Bup%7D%7D%2Bf_p%20I%5E%2A%5Cbigr%29%5CBigr%5D%5CBiggr%5C%7D%5E%7B0.25%7D%5C" title="\;\mathit{MRT}^*=\Biggl\{\frac{1}{\sigma}\Bigl[f_a L_{\mathrm{surf}}^{\mathrm{dn}}+f_a L_{\mathrm{surf}}^{\mathrm{up}}+\frac{\alpha_{\mathrm{ir}}}{\varepsilon_p}\bigl(f_a S_{\mathrm{surf}}^{\mathrm{dn,diffuse}}+f_a S_{\mathrm{surf}}^{\mathrm{up}}+f_p I^*\bigr)\Bigr]\Biggr\}^{0.25}\" />



can be approximated by

```python
# in `estimate_solar_radiation.py`
df['mean_radiant_temp'] = df['temperature'] + 0.7 * df['solar_radiation_estimated'] / 100
```

---

## 1. Identify the small perturbation

We split the full formula into:

1.  A **background** long‐wave term, which we approximate by the air temperature
<img src="https://latex.codecogs.com/svg.image?%5Ccolor%7Bwhite%7D%5C%28T_%7B%5Cmathrm%7Bair%7D%7D%5C%29" title="\;\(T_{\mathrm{air}}\)" />.
2.  A **solar** perturbation term (direct + diffuse + reflection).

Write the total flux inside the <img src="https://latex.codecogs.com/svg.image?%5Ccolor%7Bwhite%7D%5C%284_%7Bth%7D%5C%29" title="\(4_{th}\)" /> -root as

<img src="https://latex.codecogs.com/svg.image?%5Ccolor%7Bwhite%7D%5C%5B%5CPhi_%7B%5Cmathrm%7Bbg%7D%7D%2B%5CDelta%5CPhi_%7B%5Cmathrm%7Bsolar%7D%7D%5C%5D" title="\[\Phi_{\mathrm{bg}}+\Delta\Phi_{\mathrm{solar}}\]" />,

where

-   <img src="https://latex.codecogs.com/svg.image?%5Ccolor%7Bwhite%7D%5C%28%5CPhi_%7B%5Cmathrm%7Bbg%7D%7D%3D%5Csigma%5C%2CT_%7B%5Cmathrm%7Bair%7D%7D%5E4%5C%29" title="\(\Phi_{\mathrm{bg}}=\sigma\,T_{\mathrm{air}}^4\)" />
-   <img src="https://latex.codecogs.com/svg.image?%5Ccolor%7Bwhite%7D%5C%28%5CDelta%5CPhi_%7B%5Cmathrm%7Bsolar%7D%7D%3D%5Cdfrac%7B%5Calpha_%7B%5Cmathrm%7Bir%7D%7D%7D%7B%5Cvarepsilon_p%7D%5Cbigl%28f_a%20S_%7B%5Cmathrm%7Btot%7D%7D%5Cbigr%29%5C%29" title="\(\Delta\Phi_{\mathrm{solar}}=\dfrac{\alpha_{\mathrm{ir}}}{\varepsilon_p}\bigl(f_a S_{\mathrm{tot}}\bigr)\)" />

and

<img src="https://latex.codecogs.com/svg.image?%5Ccolor%7Bwhite%7D%5C%5BS_%7B%5Cmathrm%7Btot%7D%7D%3DS_%7B%5Cmathrm%7Bsurf%7D%7D%5E%7B%5Cmathrm%7Bdn%2Cdiffuse%7D%7D%2BS_%7B%5Cmathrm%7Bsurf%7D%7D%5E%7B%5Cmathrm%7Bup%7D%7D%2B%5Cfrac%7Bf_p%7D%7Bf_a%7D%5C%2CI%5E%2A%5Capprox%5Ctext%7B%28estimated%20global%20solar%20flux%29%7D.%5C%5D" title="\[S_{\mathrm{tot}}=S_{\mathrm{surf}}^{\mathrm{dn,diffuse}}+S_{\mathrm{surf}}^{\mathrm{up}}+\frac{f_p}{f_a}\,I^*\approx\text{(estimated global solar flux)}\]" /> .

In code,```{solar_radiation_estimated}```stands in for <img src="https://latex.codecogs.com/svg.image?%5Ccolor%7Bwhite%7D%5C%28S_%7B%5Cmathrm%7Btot%7D%7D%5C%29" title="\(S_{\mathrm{tot}}\)" />.

## 2. Linearize the <img src="https://latex.codecogs.com/svg.image?%5Ccolor%7Bwhite%7D%5C%284_%7Bth%7D%5C%29" title="\(4_{th}\)" /> -root

We have

<img src="https://latex.codecogs.com/svg.image?%5Ccolor%7Bwhite%7D%5C%5B%5Cmathit%7BMRT%7D%5E%2A%3D%5Cbigl%28%5CPhi_%7B%5Cmathrm%7Bbg%7D%7D%2B%5CDelta%5CPhi_%7B%5Cmathrm%7Bsolar%7D%7D%5Cbigr%29%5E%7B1/4%7D%3D%5Cbigl%28%5Csigma%5C%2CT_%7B%5Cmathrm%7Bair%7D%7D%5E4%2B%5CDelta%5CPhi%5Cbigr%29%5E%7B1/4%7D.%5C%5D" title="\[\mathit{MRT}^*=\bigl(\Phi_{\mathrm{bg}}+\Delta\Phi_{\mathrm{solar}}\bigr)^{1/4}=\bigl(\sigma\,T_{\mathrm{air}}^4+\Delta\Phi\bigr)^{1/4}.\]" />

For <img src="https://latex.codecogs.com/svg.image?%5Ccolor%7Bwhite%7D%5C%28%5CDelta%5CPhi%5Cll%5Csigma%20T_%7B%5Cmathrm%7Bair%7D%7D%5E4%5C%29" title="\(\Delta\Phi\ll\sigma T_{\mathrm{air}}^4\)" />, use the Taylor expansion about <img src="https://latex.codecogs.com/svg.image?%5Ccolor%7Bwhite%7D%5C%28T_%7B%5Cmathrm%7Bair%7D%7D%5C%29" title="\(T_{\mathrm{air}}\)" />:

<img src="https://latex.codecogs.com/svg.image?%5Ccolor%7Bwhite%7D%5C%5B%28x%2B%5Cdelta%29%5E%7B1/4%7D%5Capprox%20x%5E%7B1/4%7D%2B%5Ctfrac%7B1%7D%7B4%7Dx%5E%7B-3/4%7D%5C%2C%5Cdelta%5Cquad%5Ctext%7Bwith%7Dx%3D%5Csigma%20T_%7B%5Cmathrm%7Bair%7D%7D%5E4%2C%5C%5C%5Cdelta%3D%5CDelta%5CPhi.%5C%5D" title="\[(x+\delta)^{1/4}\approx x^{1/4}+\tfrac{1}{4}x^{-3/4}\,\delta\quad\text{with}x=\sigma T_{\mathrm{air}}^4,\\\delta=\Delta\Phi.\]" />

Thus

<img src="https://latex.codecogs.com/svg.image?%5Ccolor%7Bwhite%7D%5C%5B%5Cmathit%7BMRT%7D%5E%2A%5Capprox%20T_%7B%5Cmathrm%7Bair%7D%7D%2B%5Cfrac%7B%5CDelta%5CPhi%7D%7B4%5C%2C%5Csigma%5C%2CT_%7B%5Cmathrm%7Bair%7D%7D%5E3%7D.%5C%5D" title="\[\mathit{MRT}^*\approx T_{\mathrm{air}}+\frac{\Delta\Phi}{4\,\sigma\,T_{\mathrm{air}}^3}.\]" />

## 3. Plug in typical values

-  Stefan–Boltzmann constant<img src="https://latex.codecogs.com/svg.image?%5Ccolor%7Bwhite%7D%5C%28%5Csigma%3D5.67%5Ctimes10%5E%7B-8%7D%5Crm%5C%3BW/m%5E2/K%5E4%5C%29" title="\(\sigma=5.67\times10^{-8}\rm\;W/m^2/K^4\)" />
-  Typical air temperature: <img src="https://latex.codecogs.com/svg.image?%5Ccolor%7Bwhite%7D%5C%28T_%7B%5Cmathrm%7Bair%7D%7D%5Capprox300%5Crm%5C%3BK%5C%29" title="\(T_{\mathrm{air}}\approx300\rm\;K\)" />

Compute the denominator:

<img src="https://latex.codecogs.com/svg.image?%5Ccolor%7Bwhite%7D%5C%5B4%5C%2C%5Csigma%5C%2CT_%7B%5Cmathrm%7Bair%7D%7D%5E3%3D4%5Ctimes5.67%5Ctimes10%5E%7B-8%7D%5Ctimes%28300%29%5E3%5Capprox600%5C%3B%5Crm%20W/m%5E2/K.%5C%5D" title="\[4\,\sigma\,T_{\mathrm{air}}^3=4\times5.67\times10^{-8}\times(300)^3\approx600\;\rm W/m^2/K.\]" />

So a flux perturbation of <img src="https://latex.codecogs.com/svg.image?%5Ccolor%7Bwhite%7D%5C%28100%5Crm%5C%3BW/m%5E2%5C%29" title="\(100\rm\;W/m^2\)" /> changes temperature by

<img src="https://latex.codecogs.com/svg.image?%5Ccolor%7Bwhite%7D%5C%5B%5CDelta%20T%5Capprox%5Cfrac%7B100%7D%7B600%7D%5Capprox0.17%5C%3B%5Crm%20K.%5C%5D" title="\[\Delta T\approx\frac{100}{600}\approx0.17\;\rm K.\]" />

## 4. Incorporate absorptivity and angle factors

The solar perturbation was

<img src="https://latex.codecogs.com/svg.image?%5Ccolor%7Bwhite%7D%5C%5B%5CDelta%5CPhi%3D%5Cfrac%7B%5Calpha_%7B%5Cmathrm%7Bir%7D%7D%7D%7B%5Cvarepsilon_p%7D%5C%2Cf_a%5C%2CS_%7B%5Cmathrm%7Btot%7D%7D.%5C%5D" title="\[\Delta\Phi=\frac{\alpha_{\mathrm{ir}}}{\varepsilon_p}\,f_a\,S_{\mathrm{tot}}.\]" />

Using the **rule of thumb** that all the geometric factors and emissivity are lumped into one constant, we set:

-   <img src="https://latex.codecogs.com/svg.image?%5Ccolor%7Bwhite%7D%5C%28%5Calpha_%7B%5Cmathrm%7Bir%7D%7D%5Capprox0.7%5C%29" title="\(\alpha_{\mathrm{ir}}\approx0.7\)" />
-   <img src="https://latex.codecogs.com/svg.image?%5Ccolor%7Bwhite%7D%5C%28%5Cvarepsilon_p%5Capprox1%5C%29" title="\(\varepsilon_p\approx1\)" />
-   <img src="https://latex.codecogs.com/svg.image?%5Ccolor%7Bwhite%7D%5C%28f_a%5Capprox1%5C%29" title="\(f_a\approx1\)" />
-   <img src="https://latex.codecogs.com/svg.image?%5Ccolor%7Bwhite%7D%5C%284%5C%2C%5Csigma%5C%2CT_%7B%5Cmathrm%7Bair%7D%7D%5E3%5Capprox600%5Crm%5C%3BW/m%5E2/K%5Cto100%5Crm%5C%3BW/m%5E2/K%5C%29%28%2Acruder%2Around%20off%29" title="\(4\,\sigma\,T_{\mathrm{air}}^3\approx600\rm\;W/m^2/K\to100\rm\;W/m^2/K\)(*cruder*round off)" />

so

<img src="https://latex.codecogs.com/svg.image?%5Ccolor%7Bwhite%7D%5C%5B%5CDelta%20T%5Capprox%5Cfrac%7B%5Calpha_%7B%5Cmathrm%7Bir%7D%7D%5C%2CS_%7B%5Cmathrm%7Btot%7D%7D%7D%7B100%7D%3D0.7%5Ctimes%5Cfrac%7BS_%7B%5Cmathrm%7Btot%7D%7D%7D%7B100%7D.%5C%5D" title="\[\Delta T\approx\frac{\alpha_{\mathrm{ir}}\,S_{\mathrm{tot}}}{100}=0.7\times\frac{S_{\mathrm{tot}}}{100}.\]" />

Switching back to code:

```python
# MRT ≈ T_air + 0.7 × (solar_flux / 100)
df['mean_radiant_temp'] = df['temperature'] + 0.7 * df['solar_radiation_estimated'] / 100
```

---

In this way, we go from the full <img src="https://latex.codecogs.com/svg.image?%5Ccolor%7Bwhite%7D%5C%28T%5E%7B1/4%7D%5C%29" title="\(T^{1/4}\)" /> Stefan–Boltzmann inversion to a simple linear "knock‐in" of solar radiation into air temperature, with a single gain factor <img src="https://latex.codecogs.com/svg.image?%5Ccolor%7Bwhite%7D%5C%280.7/100%5C%29" title="\(0.7/100\)" />. 