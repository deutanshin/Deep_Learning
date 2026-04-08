import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../img/img32.jpg', 0)

M, N = img.shape

P, Q = 2*M, 2*N
padded_img = np.zeros((P, Q))
padded_img[:M, :N] = img

cv2.imwrite('../img/padded_img.jpg', padded_img)