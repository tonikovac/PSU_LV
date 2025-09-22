import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

cars = pd.read_csv("mtcars.csv")

plt.figure(figsize=(8,5))
sns.barplot(x="cyl", y="mpg", data=cars, ci=None)
plt.title("Potrošnja automobila po broju cilindara")
plt.xlabel("Broj cilindara")
plt.ylabel("MPG")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x="cyl", y="wt", data=cars)
plt.title("Težina automobila")
plt.xlabel("Broj cilindara")
plt.ylabel("Težina")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x="am", y="mpg", data=cars)
plt.title("Potrošnja automobila po tipu mjenjača")
plt.xlabel("Tip mjenjača")
plt.ylabel("MPG")
plt.show()


plt.figure(figsize=(8,5))
sns.scatterplot(x="hp", y="qsec", hue="am", data=cars, palette={0:"red", 1:"blue"}, s=100)
plt.title("Ubrzanje i snaga po tipu mjenjača")
plt.xlabel("Snaga")
plt.ylabel("Vrijeme ubrzanja")
plt.legend(title="Mjenjač", labels=["Automatski","Ručni"])
plt.show()
