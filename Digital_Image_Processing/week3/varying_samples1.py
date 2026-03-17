import cv2
import matplotlib.pyplot as plt

img_bgr = cv2.imread('img/plane.jpg', cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h, w, c = img_rgb.shape

img_subsampled2 = cv2.resize(img_rgb, (w // 2, h // 2))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(img_rgb)
axes[0].set_title(f'Original ({w}x{h})')
axes[0].axis('on')

axes[1].imshow(img_subsampled2)
axes[1].set_title(f'Subsampled ({w//2}x{h//2})')
axes[1].axis('on')

plt.show()

img_save_bgr = cv2.cvtColor(img_subsampled2, cv2.COLOR_RGB2BGR)
cv2.imwrite('img/subsampled_2.jpg', img_save_bgr)


