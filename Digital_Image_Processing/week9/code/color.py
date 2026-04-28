import cv2
import matplotlib.pyplot as plt
import numpy as np


img_bgr = cv2.imread('../img/birds.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

R, G, B = cv2.split(img_rgb)

zeros = np.zeros_like(R)
img_R = cv2.merge([R, zeros, zeros])
img_G = cv2.merge([zeros, G, zeros])
img_B = cv2.merge([zeros, zeros, B])

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
axes[0].imshow(img_rgb); axes[0].set_title('Original'); axes[0].axis('off')
axes[1].imshow(img_R); axes[0].set_title('R'); axes[1].axis('off')
axes[2].imshow(img_G); axes[0].set_title('G'); axes[2].axis('off')
axes[3].imshow(img_B); axes[0].set_title('B'); axes[3].axis('off')
plt.tight_layout
plt.show()

cv2.imwrite('../img/channel_R.jpg', cv2.cvtColor(img_R, cv2.COLOR_RGB2BGR))
cv2.imwrite('../img/channel_G.jpg', cv2.cvtColor(img_G, cv2.COLOR_RGB2BGR))
cv2.imwrite('../img/channel_B.jpg', cv2.cvtColor(img_B, cv2.COLOR_RGB2BGR))