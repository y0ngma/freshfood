import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
x = np.arange(100)
y1 = np.random.normal(0, 1, 100)
y2 = np.random.normal(0, 1, 100).cumsum()

## 1. Automatic detection of elements to be shown in the legend
## 1-1. label
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)

## labels: A list of labels to show next to the artists.
ax.plot(x, y1, 'b-', label='company 1')
ax.plot(x, y2, 'r--', label='Company 2')
ax.legend()
plt.show()

