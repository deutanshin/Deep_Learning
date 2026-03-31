import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('../img/bean.jpg', cv2.IMREAD_GRAYSCALE)
H, W = img.shape

array = np.zeros(256)
for i in range(H):
    for j in range(W):
        array[img[i, j]] += 1

cdf = np.zeros(256)
sum = 0
for i in range(256):
    sum += array[i]
    cdf[i] = sum

plt.figure()
plt.hist(img.ravel(), 256, [0, 256])

plt.savefig('./hist1.jpg')
plt.clf()


cdf_min = cdf[cdf > 0][0]
cdf_max = cdf[255]
L = 256

result = np.zeros((H, W), dtype=np.uint8)

for i in range(H):
    for j in range(W):
        v = img[i, j]
        numer = cdf[v] - cdf_min
        denomi= cdf_max - cdf_min
        
        if denomi != 0:
            h_v = round((numer/ denomi) * (L - 1))
        else:
            h_v = v
            
        result[i, j] = h_v

plt.figure()
plt.hist(result.ravel(), 256, [0, 256])

plt.savefig('./hist2.jpg')
plt.close()

cv2.imwrite('./result.jpg', result)