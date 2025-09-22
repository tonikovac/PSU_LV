import pandas as pd

cars = pd.read_csv("mtcars.csv")

najveca_potrosnja = cars.sort_values("mpg").head(5)[["car", "mpg"]]
print(najveca_potrosnja, "\n")

najmanja_potrosnja_8 = cars[cars["cyl"] == 8].sort_values("mpg").head(3)[["car", "mpg"]]
print(najmanja_potrosnja_8, "\n")

srednja_potrosnja_6 = cars[cars["cyl"] == 6]["mpg"].mean()
print(round(srednja_potrosnja_6, 2), "\n")

srednja_potrosnja_4 = cars[(cars["cyl"] == 4) & (cars["wt"]*1000 >= 2000) & (cars["wt"]*1000 <= 2200)]["mpg"].mean()
print(round(srednja_potrosnja_4, 2), "\n")

broj_mjenjaca = cars["am"].value_counts()
print("automatski:", broj_mjenjaca[0], ", ručni :", broj_mjenjaca[1], "\n")

auto_preko100 = cars[(cars["am"] == 0) & (cars["hp"] > 100)].shape[0]
print(auto_preko100, "\n")

cars["masa_kg"] = cars["wt"] *  453.59
print(cars[["car", "masa_kg"]])