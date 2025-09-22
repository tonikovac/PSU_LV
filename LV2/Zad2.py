import numpy as np
import matplotlib .pyplot as plt

data = np.loadtxt(open("mtcars.csv", "rb"),
                  delimiter=",", skiprows=1,
                  usecols=(1, 2, 3, 4, 5, 6))

mpg = data[:, 0]
hp = data[:, 3]

plt.scatter(hp, mpg)
plt.xlabel('Horsepower (hp)')
plt.ylabel('Miles per Gallon (mpg)')
plt.title('Potrošnja / Snaga')
plt.show()
wt = data[:, 5]

plt.scatter(hp, mpg, s=wt*100, alpha=0.5)
plt.xlabel('Horsepower (hp)')
plt.ylabel('Miles per Gallon (mpg)')
plt.title('Potrošnja / Snaga')
plt.show()
min_mpg = np.min(mpg)
max_mpg = np.max(mpg)
mean_mpg = np.mean(mpg)

print(f"Minimalna potrošnja: {min_mpg}")
print(f"Maksimalna potrošnja: {max_mpg}")
print(f"Srednja potrošnja: {mean_mpg:.2f}")
cyl = data[:, 1]
mpg_6cyl = mpg[cyl == 6]

min_mpg_6 = np.min(mpg_6cyl)
max_mpg_6 = np.max(mpg_6cyl)
mean_mpg_6 = np.mean(mpg_6cyl)

print("Za automobile sa 6 cilindara:")
print(f"Minimalna potrošnja: {min_mpg_6}")
print(f"Maksimalna potrošnja: {max_mpg_6}")
print(f"Srednja potrošnja: {mean_mpg_6:.2f}")
