import os
from glob import glob  # Used only for instructive purposes

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pims
import scipy.odr as odr
import trackpy as tp
from pandas import DataFrame, Series  # for convenience
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


concentratie_0_hoogtes = [
    490,
    390,
    130,
    170,
    290,
    150,
    310,
    250,
    190,
    330,
    270,
    230,
    370,
    210,
    350,
    470,
    530,
    450,
    510,
    410,
    430,
]
concentratie_0_aantal_deeltjes = [
    ufloat(540.94, 19.625911443803062),
    ufloat(165.14, 9.816333327673831),
    ufloat(9.5, 2.3),
    ufloat(16.94, 2.9217118269945788),
    ufloat(65.86, 7.158240007152596),
    ufloat(11.58, 2.20081802973349),
    ufloat(85.32, 7.925755484494837),
    ufloat(39.64, 3.8562157615984094),
    ufloat(18.1, 2.7073972741361767),
    ufloat(104.28, 8.3977139746481),
    ufloat(47.26, 5.245226401214727),
    ufloat(36.4, 4.741307836451879),
    ufloat(146.62, 8.037138794372037),
    ufloat(32.72, 3.4701008630874117),
    ufloat(125.36, 7.271203476729282),
    ufloat(431.68, 14.107359781333997),
    ufloat(809.96, 18.863679386588398),
    ufloat(360.16, 11.293112945507984),
    ufloat(673.98, 17.16914674641696),
    ufloat(229.3, 11.103603018840326),
    ufloat(277.38, 10.94511763299052),
]
concentratie_0_echte_hoogtes = list(
    map(lambda x: kalibratie(0, x), concentratie_0_hoogtes)
)

concentratie_1_hoogtes = [
    480,
    280,
    140,
    160,
    120,
    380,
    200,
    340,
    220,
    360,
    320,
    260,
    300,
    240,
    180,
    420,
    400,
    440,
    500,
    460,
]
concentratie_1_aantal_deeltjes = [
    ufloat(352.3, 14.906709898565811),
    ufloat(31.68, 5.330816072610272),
    ufloat(5.92, 1.8421726303471129),
    ufloat(9.54, 2.0900717691026784),
    ufloat(5.78, 1.952331938989884),
    ufloat(111.66, 11.172484056824606),
    ufloat(13.26, 3.04506157573209),
    ufloat(72.1, 6.487680633323438),
    ufloat(19.0, 3.1622776601683795),
    ufloat(92.56, 6.447200943044974),
    ufloat(52.9, 4.9325449820554095),
    ufloat(28.0, 3.3105890714493698),
    ufloat(42.04, 4.0593595553978705),
    ufloat(22.34, 4.366279881088706),
    ufloat(9.22, 2.879513847856961),
    ufloat(177.46, 8.29267146340671),
    ufloat(143.66, 10.136291234963604),
    ufloat(214.92, 12.827844713746734),
    ufloat(454.06, 13.767221941989606),
    ufloat(289.64, 12.849529174253817),
]
concentratie_1_echte_hoogtes = list(
    map(lambda x: kalibratie(1, x), concentratie_1_hoogtes)
)

concentratie_2_hoogtes = [
    170,
    150,
    290,
    390,
    130,
    490,
    410,
    430,
    530,
    470,
    510,
    450,
    370,
    230,
    350,
    210,
    190,
    250,
    310,
    270,
    330,
]
concentratie_2_aantal_deeltjes = [
    ufloat(2.54, 1.1173182178770735),
    ufloat(0.18, 0.38418745424597095),
    ufloat(4.58, 1.778651174345324),
    ufloat(14.4, 3.358571124749333),
    ufloat(0.62, 0.66),
    ufloat(54.52, 4.875407675261629),
    ufloat(18.14, 3.4000588230205664),
    ufloat(24.2, 2.4166091947189146),
    ufloat(110.86, 7.647247870966392),
    ufloat(43.94, 4.305391968218458),
    ufloat(75.3, 6.989277502002621),
    ufloat(24.24, 2.634843448859913),
    ufloat(9.92, 3.8201570648338534),
    ufloat(3.4, 1.4422205101855958),
    ufloat(8.88, 2.285957129956728),
    ufloat(2.1, 0.854400374531753),
    ufloat(3.9, 1.374772708486752),
    ufloat(1.44, 0.9830564581955606),
    ufloat(6.96, 2.2087100307645637),
    ufloat(4.68, 2.18577217477028),
    ufloat(7.16, 1.803995565404749),
]
concentratie_2_echte_hoogtes = list(
    map(lambda x: kalibratie(2, x), concentratie_2_hoogtes)
)

concentratie_3_hoogtes = [
    430,
    410,
    510,
    450,
    470,
    350,
    210,
    370,
    230,
    270,
    330,
    250,
    310,
    190,
    290,
    150,
    170,
    130,
    390,
    490,
]
concentratie_3_aantal_deeltjes = [
    ufloat(10.56, 3.163289427162807),
    ufloat(7.36, 2.2068982758613953),
    ufloat(26.3, 4.57493169347915),
    ufloat(12.9, 2.41039415863879),
    ufloat(14.56, 2.786826151736057),
    ufloat(3.52, 0.964157663455516),
    ufloat(0.36, 0.5571355310873648),
    ufloat(4.42, 1.4709180806557516),
    ufloat(0.34, 0.5517245689653488),
    ufloat(1.54, 0.876584280032445),
    ufloat(2.5, 1.4730919862656235),
    ufloat(0.4, 0.5291502622129182),
    ufloat(2.0, 1.0392304845413265),
    ufloat(0.66, 0.6514598989960932),
    ufloat(1.18, 0.9314504817756014),
    ufloat(0.7, 0.7810249675906653),
    ufloat(1.02, 0.6779380502671317),
    ufloat(0.34, 0.5517245689653488),
    ufloat(7.56, 1.4022838514366482),
    ufloat(17.46, 2.070845238061019),
]
concentratie_3_echte_hoogtes = list(
    map(lambda x: kalibratie(3, x), concentratie_3_hoogtes)
)

fig, axs = plt.subplots(2, 2, constrained_layout=True)
box = dict(facecolor="yellow", pad=5, alpha=0.2)
# fig.align_ylabels(axs[:, 1])
# fig.subplots_adjust(left=200, wspace=600)
# labelx = -0.3  # axes coords

# for j in range(2):
#     axs[j, 1].yaxis.set_label_coords(labelx, 0.5)

axs[0, 0].errorbar(
    list(map(lambda x: x.n, concentratie_3_echte_hoogtes)),
    list(map(lambda x: x.n, concentratie_3_aantal_deeltjes)),
    xerr=list(map(lambda x: x.s, concentratie_3_echte_hoogtes)),
    yerr=list(map(lambda x: x.s, concentratie_3_aantal_deeltjes)),
    fmt="o",
    color="red",
)
axs[0, 1].errorbar(
    list(map(lambda x: x.n, concentratie_2_echte_hoogtes)),
    list(map(lambda x: x.n, concentratie_2_aantal_deeltjes)),
    xerr=list(map(lambda x: x.s, concentratie_2_echte_hoogtes)),
    yerr=list(map(lambda x: x.s, concentratie_2_aantal_deeltjes)),
    fmt="o",
    color="orange",
)
axs[1, 0].errorbar(
    list(map(lambda x: x.n, concentratie_1_echte_hoogtes)),
    list(map(lambda x: x.n, concentratie_1_aantal_deeltjes)),
    xerr=list(map(lambda x: x.s, concentratie_1_echte_hoogtes)),
    yerr=list(map(lambda x: x.s, concentratie_1_aantal_deeltjes)),
    fmt="o",
    color="forestgreen",
)
axs[1, 1].errorbar(
    list(map(lambda x: x.n, concentratie_0_echte_hoogtes)),
    list(map(lambda x: x.n, concentratie_0_aantal_deeltjes)),
    xerr=list(map(lambda x: x.s, concentratie_0_echte_hoogtes)),
    yerr=list(map(lambda x: x.s, concentratie_0_aantal_deeltjes)),
    fmt="o",
    color="royalblue",
)

for ax in axs.flat:
    ax.set_ylim(bottom=0)

axs[0, 0].set_title("(a) Concentration 0.05%", fontsize=10)
axs[0, 0].set_ylabel("Number of particles $N$ (AU)")
axs[0, 1].set_title("(b) Concentration 0.1%", fontsize=10)
axs[1, 0].set_title("(c) Concentration 0.5%", fontsize=10)
axs[1, 0].set(xlabel="Hoogte $z$ (µm)")
axs[1, 0].set_ylabel("Number of particles $N$ (AU)")
axs[1, 1].set_title("(d) Concentration 1%", fontsize=10)
axs[1, 1].set(xlabel="Hoogte $z$ (µm)")
fig.align_ylabels(axs)

plt.show()
