import cv2
import numpy as np
import matplotlib.pyplot as plt

img_original = cv2.imread('birds.jpg')
img_hsv = cv2.cvtColor(img_original, cv2.COLOR_BGR2HSV)

h, s ,v = cv2.split(img_hsv)

v_tmp = np.clip(v.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)

img_result = cv2.merge([h, s, v_tmp])
img_result = cv2.cvtColor(img_result, cv2.COLOR_HSV2BGR)

cv2.imwrite('img_modified.jpg', img_result)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
axes[1].set_title('Modified (V channel x 1.5)')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('comparison_result.jpg')

# V 값을 1.5배하여 원본이미지와 대조해 보았을 때
# 이미지 고유의 색상과 선명도는 변하지 않은 채 전체적인 밝기가 높아진 것을 확인하였다.
# 따라서 색상의 왜곡 없이 밝기만을 제어하는 상황에서는 HSV 모델이 효과적임을 알 수 있다.