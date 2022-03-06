import numpy as np
import json
import matplotlib.pyplot as plt
x = json.load(open("./results/losses.json"))
print(x)
#x = np.array([5, 4, 1, 4, 5])
y = np.linspace(0, len(x), len(x))

plt.title("Line graph")
plt.plot(y, x, color="red")
plt.ylim([-100000, 10000000])
plt.show()
