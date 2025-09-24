from glob import glob  # Used only for instructive purposes

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pims
import trackpy as tp
from pandas import DataFrame, Series  # for convenience\
from uncertainties import ufloat

hoogte_per_concentratie = [ufloat(565, 10), ufloat(590, 10), ufloat(570, 10)]
dikte_plaatje = ufloat(750, 60)


def kalibratie(concentratie, hoogte):
    x_0 = hoogte_per_concentratie[concentratie]
    return -dikte_plaatje / (x_0 - 130) * hoogte + dikte_plaatje * x_0 / (x_0 - 130)


print(kalibratie(0, 565))


@pims.pipeline
def gray(image):
    return image[:, :, 1]  # Take just the blue channel


def deeltjes_tellen(filename):
    frame = gray(pims.open(filename))[0]
    f = tp.locate(frame, 15, invert=True, minmass=250)
    # tp.annotate(f, frame)
    return len(f)


# deeltjes_tellen("data/1% hoogte 190 mm/A0004-20250924_134311.jpg")


def gemiddelde_deeltjes_per_hoogte(hoogte):
    files = glob(f"data/{hoogte}/*.jpg")
    list = []
    for file in files:
        list.append(deeltjes_tellen(file))

    gemiddelde = np.mean(list)
    std = np.std(list)
    print(gemiddelde)
    print(std)


# gemiddelde_deeltjes_per_hoogte("1% hoogte 530 mm")
# gemiddelde_deeltjes_per_hoogte("1% hoogte 210 mm")
# frame = gray(pims.open("foto.jpg"))[0]
# # print(len(frame))
# # plt.imshow(frame)
# # plt.show()
# # print(list(frame)[0])
# f = tp.locate(frame, 15, invert=True, minmass=300)
# print(f)
# tp.annotate(f, frame)
# # fig, ax = plt.subplots()
# # ax.hist(f["mass"], bins=20)

# # # Optionally, label the axes.
# # ax.set(xlabel="mass", ylabel="count")
# # # ax.show()
# # plt.show()
