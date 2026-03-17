import cv2
import numpy as np

img = cv2.imread('img/xray.jpg', cv2.IMREAD_GRAYSCALE)

levels = [32, 8, 4, 2]

for i in levels:
    factor = 256 // i

    img_quantized = (img // factor) * (255 // (i - 1))
    img_quantized = img_quantized.astype(np.uint8)

    cv2.imwrite(f'xray_q{i}.jpg', img_quantized)