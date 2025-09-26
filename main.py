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


def kalibratie(concentratie, hoogte):
    x_0 = hoogte_per_concentratie[concentratie]
    begin = begin_hoogte_concentratie[concentratie]
    return -dikte_plaatje / (x_0 - begin) * hoogte + dikte_plaatje * x_0 / (x_0 - begin)


@pims.pipeline
def gray(image):
    return image[:, :, 1]  # Take just the blue channel


def deeltjes_tellen(filename):
    frame = gray(pims.open(filename))[0]
    f = tp.locate(frame, 15, invert=True, minmass=250)
    # tp.annotate(f, frame)
    return len(f)


# deeltjes_tellen("data/1% hoogte 190 mm/A0004-20250924_134311.jpg")


def gemiddelde_deeltjes_per_hoogte(concentratie, hoogte):
    namen = ["1%", "0.5%", "0.1%", "0.05%"]
    naam = namen[concentratie]
    files = glob("data/" + naam + "/" + naam + " hoogte " + str(hoogte) + " mm/*.jpg")
    list = []
    for file in files:
        list.append(deeltjes_tellen(file))
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


def f(B, ln_N):
    return B[0] + B[1] * ln_N


B_start = [5.0, 0.25]


def plot_z0_bepalen(concentratie):
    hoogtes = hoogtes_per_concentratie(concentratie)
    echte_hoogtes = []
    echte_hoogtes_std = []
    for a in hoogtes:
        echte_hoogtes.append(kalibratie(concentratie, a).n)
        echte_hoogtes_std.append(kalibratie(concentratie, a).s)

    aantal_deeltjes = []
    for hoogte in hoogtes:
        aantal_deeltjes.append(gemiddelde_deeltjes_per_hoogte(concentratie, hoogte))
    ln_N = []
    ln_N_std = []
    for i in aantal_deeltjes:
        print(i)
        ln_N.append(log(i).n)
        ln_N_std.append(log(i).s)

    odr_model = odr.Model(f)
    odr_data = odr.RealData(echte_hoogtes, ln_N, sx=echte_hoogtes_std, sy=ln_N_std)
    odr_obj = odr.ODR(odr_data, odr_model, beta0=B_start)
    odr_res = odr_obj.run()
    par_best = odr_res.beta

    par_sig_ext = odr_res.sd_beta
    par_cov = odr_res.cov_beta
    print("De (INTERNE!) covariantiematrix  = \n", par_cov)
    chi2 = odr_res.sum_square
    print("\nChi-squared =", chi2)
    chi2red = odr_res.res_var
    print("Reduced chi-squared =", chi2red, "\n")
    odr_res.pprint()

    xplot = np.linspace(np.min(echte_hoogtes), np.max(echte_hoogtes), num=100)
    plt.plot(xplot, f(par_best, xplot), "r")

    plt.errorbar(
        echte_hoogtes,
        ln_N,
        xerr=echte_hoogtes_std,
        yerr=ln_N_std,
        fmt="o",
        markersize=5,
    )
    plt.show()


# Plot_hoogte_aantal_deeltjes(2)
# Plot_hoogte_aantal_deeltjes(0)
plot_z0_bepalen(3)

# x = ufloat(1000, 3)
# print(log(x))
