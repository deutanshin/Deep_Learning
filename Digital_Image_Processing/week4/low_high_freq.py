import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import cv2

img = cv2.imread('img/pandas.jpg', cv2.IMREAD_GRAYSCALE)

h, w = img.shape
size = 100

high_region = img[h // 2-size//2:h//2+size//2, w//2-size//2:w//2+size//2]

low_region = img[0:size, w-size:w]

x = np.arange(size)
y = np.arange(size)

x, y = np.meshgrid(x, y)

fig = plt.figure()
ax0 = fig.add_subplot(121, projection='3d')
ax1 = fig.add_subplot(122, projection='3d')
ax0.plot_surface(x, y, high_region.astype(float), cmap=cm.viridis)
ax1.plot_surface(x, y, low_region.astype(float), cmap=cm.viridis)
plt.show()


# ascasc