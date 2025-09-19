from glob import glob  # Used only for instructive purposes

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pims
import trackpy as tp
from pandas import DataFrame, Series  # for convenience


@pims.pipeline
def gray(image):
    return image[:, :, 1]  # Take just the green channel


frame = gray(pims.open("foto.jpg"))[0]
# print(len(frame))
# plt.imshow(frame)
# plt.show()
# print(list(frame)[0])
f = tp.locate(frame, 15, invert=True, minmass=300)
print(f)
tp.annotate(f, frame)
# fig, ax = plt.subplots()
# ax.hist(f["mass"], bins=20)

# # Optionally, label the axes.
# ax.set(xlabel="mass", ylabel="count")
# # ax.show()
# plt.show()
