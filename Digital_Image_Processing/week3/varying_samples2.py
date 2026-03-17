import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('img/chest.jpg', cv2.IMREAD_GRAYSCALE)

quant_factor = 4

factor = 256 // quant_factor
img_quantized = (img // factor) * (255 // (quant_factor - 1))
img_quantized = img_quantized.astype(np.uint8)

fig,axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(img, cmap='gray')
axes[0].set_title(f'Original')
axes[0].axis('on')

axes[1].imshow(img_quantized, cmap='gray')
axes[1].set_title(f'Quantized ({quant_factor} Levels)')
axes[1].axis('on')

plt.show()

cv2.imwrite('img/img_quantized.jpg', img_quantized)