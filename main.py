import os
from glob import glob  # Used only for instructive purposes

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pims
import scipy.odr as odr
import trackpy as tp
from pandas import DataFrame, Series  # for convenience\
from uncertainties import ufloat
from uncertainties.umath import log

hoogte_per_concentratie = [
    ufloat(565, 10),
    ufloat(590, 10),
    ufloat(570, 10),
    ufloat(545, 10),
]
begin_hoogte_concentratie = [
    ufloat(130, 2),
    ufloat(120, 2),
    ufloat(130, 2),
    ufloat(130, 2),
]
dikte_plaatje = ufloat(750, 60)

eta = ufloat(1.0016 * 10**-3, 0.0005 * 10**-3)  # sPa
rho = ufloat(1.05 * 10**3, 0.005 * 10**3)  # kg/m^3
rho_0 = ufloat(0.9982 * 10**3, 0.0001 * 10**3)  # kg/m^3
g = ufloat(9.871, 0.001)  # m/s^2
d = ufloat(0.51 * 10**-6, 0.01 * 10**-6)  # m
v_t = d**2 * (rho - rho_0) * g / (18 * eta)  # m/s


def kalibratie(concentratie, hoogte):
    x_0 = hoogte_per_concentratie[concentratie]
    begin = begin_hoogte_concentratie[concentratie]
    return -dikte_plaatje / (x_0 - begin) * hoogte + dikte_plaatje * x_0 / (x_0 - begin)


@pims.pipeline
def gray(image):
    return image[:, :, 1]  # Take just the blue channel


def deeltjes_tellen(filename, concentratie):
    frame = gray(pims.open(filename))[0]
    f = tp.locate(frame, 15, invert=True, minmass=275)
    # tp.annotate(f, frame)
    if concentratie == 1:
        return len(f) - 1
    return len(f)


# deeltjes_tellen("data/1% hoogte 190 mm/A0004-20250924_134311.jpg")


def gemiddelde_deeltjes_per_hoogte(concentratie, hoogte):
    namen = ["1%", "0.5%", "0.1%", "0.05%"]
    naam = namen[concentratie]
    files = glob("data/" + naam + "/" + naam + " hoogte " + str(hoogte) + " mm/*.jpg")
    list = []
    for file in files:
        list.append(deeltjes_tellen(file, concentratie))
    gemiddelde = np.mean(list)
    std = np.std(list)
    return ufloat(gemiddelde, std)
    # return ufloat(deeltjes_tellen(files[1]), 10)


# gemiddelde_deeltjes_per_hoogte(2, 230)


def hoogtes_per_concentratie(concentratie):
    namen = ["1%", "0.5%", "0.1%", "0.05%"]
    naam = namen[concentratie]
    mapjes = glob("data/" + naam + "/*/", recursive=False)
    hoogtes = []
    for i in mapjes:
        hoogtes.append(int(i[14 + 2 * len(naam) : 17 + 2 * len(naam)]))
    return hoogtes


# hoogtes_per_concentratie(1)


def Plot_hoogte_aantal_deeltjes(concentratie):
    hoogtes = hoogtes_per_concentratie(concentratie)
    echte_hoogtes = []
    echte_hoogtes_std = []
    for a in hoogtes:
        echte_hoogtes.append(kalibratie(concentratie, a).n)
        echte_hoogtes_std.append(kalibratie(concentratie, a).s)

    aantal_deeltjes = []
    for hoogte in hoogtes:
        aantal_deeltjes.append(gemiddelde_deeltjes_per_hoogte(concentratie, hoogte))
    N = []
    N_std = []
    for i in aantal_deeltjes:
        N.append(i.n)
        N_std.append(i.s)

    plt.errorbar(
        echte_hoogtes, N, xerr=echte_hoogtes_std, yerr=N_std, fmt="o", markersize=5
    )
    plt.show()


# Plot_hoogte_aantal_deeltjes(0)
# Plot_hoogte_aantal_deeltjes(1)
# Plot_hoogte_aantal_deeltjes(2)
# Plot_hoogte_aantal_deeltjes(3)


def f(B, ln_N):
    return B[0] + B[1] * ln_N


B_start = [5.0, 0.25]


def plot_z0_bepalen(concentratie):
    hoogtes = hoogtes_per_concentratie(concentratie)
    echte_hoogtes = []
    echte_hoogtes_std = []
    for a in hoogtes:
        echte_hoogtes.append((kalibratie(concentratie, a) * 10**-6).n)
        echte_hoogtes_std.append((kalibratie(concentratie, a) * 10**-6).s)

    aantal_deeltjes = []
    for hoogte in hoogtes:
        aantal_deeltjes.append(gemiddelde_deeltjes_per_hoogte(concentratie, hoogte))
    ln_N = []
    ln_N_std = []
    for i in aantal_deeltjes:
        ln_N.append(log(i).n)
        ln_N_std.append(log(i).s)

    odr_model = odr.Model(f)
    odr_data = odr.RealData(echte_hoogtes, ln_N, sx=echte_hoogtes_std, sy=ln_N_std)
    odr_obj = odr.ODR(odr_data, odr_model, beta0=B_start)
    odr_res = odr_obj.run()
    par_best = odr_res.beta
    b = ufloat(par_best[1], odr_res.sd_beta[1])
    z_0 = -1 / b

    # par_sig_ext = odr_res.sd_beta
    # par_cov = odr_res.cov_beta
    # print("De (INTERNE!) covariantiematrix  = \n", par_cov)
    # chi2 = odr_res.sum_square
    # print("\nChi-squared =", chi2)
    # chi2red = odr_res.res_var
    # print("Reduced chi-squared =", chi2red, "\n")
    # odr_res.pprint()

    # xplot = np.linspace(np.min(echte_hoogtes), np.max(echte_hoogtes), num=100)
    # plt.plot(xplot, f(par_best, xplot), "r")

    # plt.errorbar(
    #     echte_hoogtes,
    #     ln_N,
    #     xerr=echte_hoogtes_std,
    #     yerr=ln_N_std,
    #     fmt="o",
    #     markersize=5,
    # )
    # plt.show()
    return z_0


# Plot_hoogte_aantal_deeltjes(0)
# Plot_hoogte_aantal_deeltjes(0)
# plot_z0_bepalen(3)

# x = ufloat(1000, 3)
# print(log(x))


def D_bepaling(concentratie):
    D = v_t * plot_z0_bepalen(concentratie)  # m^2/s
    return D


def plot_D_C():
    concentraties = [0, 1, 2, 3]
    D = []
    D_std = []
    for i in concentraties:
        D.append(D_bepaling(i).n)
        D_std.append(D_bepaling(i).s)

    gemiddelde = np.mean(D)

    plt.axhline(8.57 * 10**-13)
    plt.axhline(gemiddelde)
    plt.errorbar(concentraties, D, yerr=D_std, fmt="o")
    plt.show()


plot_D_C()
