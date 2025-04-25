# Simplified Mean Radiant Temperature Calculation

This document explains the simplified approach for calculating Mean Radiant Temperature (MRT) used in our weather analysis.

## Introduction

Mean radiant temperature (MRT) represents the uniform temperature of a hypothetical black body that would emit the same net thermal radiation to a person as the actual environment. It combines:

- **Long-wave (infrared) radiation** emitted by surrounding surfaces and the sky.
- **Short-wave (solar) radiation** from direct sunlight, diffuse sky, and reflections.

## The Simplified Approach

While traditional MRT calculations involve complex radiation models using the Stefan-Boltzmann law, we've implemented a simpler approach based on empirical research by Lindberg et al. (2008) and Thorsson et al. (2007).

### Key Formula

Our simplified approach uses a linear relationship:

$$
\text{MRT} = T_{\text{air}} + \text{Solar Adjustment}
$$

Where:
- $T_{\text{air}}$ is air temperature in °C
- Solar Adjustment varies based on cloud cover and time of day

### Solar Adjustment Calculation

The solar adjustment is calculated as:

$$
\text{Solar Adjustment} = 
\begin{cases}
(1 - 0.75 \times \text{Cloud Cover}) \times 4°\text{C} & \text{during day} \\
0°\text{C} & \text{during night}
\end{cases}
$$

Where:
- Cloud Cover is expressed as a fraction from 0 (clear sky) to 1 (complete overcast)
- Day is defined as hours between 6:00 and 18:00
- The maximum adjustment of 4°C on clear days is based on empirical studies

## Cloud Cover Data

We derive cloud cover from weather station data using the standard meteorological "oktas" coding system, representing eighths of sky coverage:

| Code | Description | Cloud Cover Fraction |
|------|-------------|---------------------|
| CLR:00 | Clear sky | 0.0 |
| FEW:01 | One okta (1/10 or less but not zero) | 0.1 |
| FEW:02 | Two oktas (2/10 - 3/10) | 0.25 |
| SCT:03 | Three oktas (4/10) | 0.4 |
| SCT:04 | Four oktas (5/10) | 0.5 |
| BKN:05 | Five oktas (6/10) | 0.6 |
| BKN:06 | Six oktas (7/10 - 8/10) | 0.75 |
| BKN:07 | Seven oktas (9/10 or more but not 10/10) | 0.9 |
| OVC:08 | Eight oktas (10/10, completely overcast) | 1.0 |
| VV:09 | Sky obscured | 1.0 |
| X:10 | Partial obscuration | 0.5 |

When multiple cloud layers are reported, we use the last layer's value to best represent the total sky state, as per meteorological conventions.

## Advantages of This Approach

1. **Simplicity**: Easy to implement and understand
2. **Efficiency**: Computationally light for large datasets
3. **Physical Interpretability**: Clear connection to observable weather conditions
4. **Empirical Basis**: Validated by research studies
5. **No Complex Parameters**: No need for radiation flux measurements or view factors

## Implementation

The implementation reads cloud cover data from weather station reports, converts oktas codes to fractional values, and applies the formula above to calculate MRT for each timestamp.

```python
# Simplified MRT calculation
df['mean_radiant_temp'] = df['temperature'] + np.where(
    df['datetime'].dt.hour.between(6, 18),  # If daytime
    (1 - 0.75 * df['cloud_cover']) * 4.0,   # Apply solar adjustment
    0.0                                     # No adjustment at night
)
```

## References

1. Lindberg, F., Holmer, B., & Thorsson, S. (2008). "SOLWEIG 1.0 – Modelling spatial variations of 3D radiant fluxes and mean radiant temperature in complex urban settings." International Journal of Biometeorology, 52(7), 697-713.
   https://doi.org/10.1007/s00484-008-0162-7

2. Thorsson, S., Lindberg, F., Eliasson, I., & Holmer, B. (2007). "Different methods for estimating the mean radiant temperature in an outdoor urban setting." International Journal of Climatology, 27(14), 1983-1993.
   https://doi.org/10.1002/joc.1537

3. Kántor, N., & Unger, J. (2011). "The most problematic variable in the course of human-biometeorological comfort assessment — the mean radiant temperature." Central European Journal of Geosciences, 3(1), 90-100.
   https://doi.org/10.2478/s13533-011-0010-x

4. World Meteorological Organization (2017). "International Cloud Atlas." WMO-No. 407.
   https://cloudatlas.wmo.int/en/home.html