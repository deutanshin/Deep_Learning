import cv2
import numpy as numpy
from matplotlib import pyplot as pyplot

image = cv2.imread('../img/cats.jpg', 0)

plt.hist(image.ravel(), 256, [0, 256])
plt.show()