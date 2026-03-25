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

strides = [2, 4, 8]

for s in strides:
    sub = img[::s, ::s]
    
    restored = cv2.resize(sub, (w, h), interpolation=cv2.INTER_NEAREST)
    
    filename = f'assignment1_restored_stride_{s}.png'
    cv2.imwrite(f'submit_img/{filename}', restored)
