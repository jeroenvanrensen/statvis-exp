from uncertainties import ufloat

plaatje1 = ufloat(150, 15)
plaatje2 = ufloat(600, 60)
hoogte_plaatje = plaatje1 + plaatje2
aantal_draaiingen = ufloat(10, 0.1)

dikte_per_draaiing = hoogte_plaatje / aantal_draaiingen
print(dikte_per_draaiing)
