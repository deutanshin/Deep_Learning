import numpy as np
import cv2

def generate_zone_plate(size=512):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    zone_plate = np.sin(100 * np.pi * R**2)

    zone_plate = ((zone_plate + 1) / 2 * 255).astype(np.uint8)
    return zone_plate


img = generate_zone_plate(512)
h, w = img.shape

stride = 4
aliased_sub = img[::stride, ::stride]

aliase_restored = cv2.resize(aliased_sub, (w, h), interpolation=cv2.INTER_NEAREST)

blurred_img = cv2.GaussianBlur(img, (15, 15), 5)
anti_aliased_sub = blurred_img[::stride, ::stride]
anti_aliased_restored = cv2.resize(anti_aliased_sub, (w, h), interpolation=cv2.INTER_NEAREST)

cv2.imwrite('img/01_original_zone_plate.png', img)

cv2.imwrite('img/02_aliased_stride4.png', aliase_restored)

cv2.imwrite('img/03_anti_aliased_stride4.png', anti_aliased_restored)