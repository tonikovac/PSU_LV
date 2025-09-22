import numpy as np
import matplotlib.pyplot as plt
x = np.array([2.0, 3.0, 3.0, 1.0, 2.0])
y= np.array([2.0, 2.0, 1.0, 1.0, 2.0])
plt.plot(x, y, 'b', linewidth=1, marker=".", markersize=5)
plt.axis([0,4,0,4])
plt.xlabel('x os')
plt.ylabel('y os')
plt.title('Primjer')
plt.show()