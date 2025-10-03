import math

from uncertainties import ufloat

k_B = (1.380649) * 10**-23  # J/K
T = ufloat(20.9 + 273.15, 0.1)  # K
d = ufloat(0.51, 0.01) * 10**-6  # m
eta = ufloat(0.9775, 0.0241) * 10**-3  # Pa s
rho = ufloat(1.05, 0.005) * 10**3  # kg/m^3
rho_0 = ufloat(0.9982, 0.0001) * 10**3  # kg/m^3
g = ufloat(9.871, 0.001)  # m/s^2

D_expected = k_B * T / (3 * math.pi * d * eta)  # m^2/s
z_0_expected = 6 * k_B * T / (math.pi * d**3 * (rho - rho_0) * g)  # m
v_t = d**2 * (rho - rho_0) * g / (18 * eta)  # m/s
print(D_expected)
