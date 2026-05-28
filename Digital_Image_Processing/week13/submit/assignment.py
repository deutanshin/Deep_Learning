import cv2
import numpy as np
import matplotlib.pyplot as plt


# 커널은 3x3의 모두 1로 채운 형태를 사용하였으며 4회 반복시에 모든 노이즈가 제거됨을 확인할 수 있었음
# 오프닝 1회 시에는 육안으로 차이를 식별할 수 없었으나 반복하는 과정을 시각화 하여 오프닝 연산의 효과를 볼 수 있었음

img = cv2.imread("../img/test.png", cv2.IMREAD_GRAYSCALE)

kernel = np.ones((3, 3), np.uint8)


erosion = cv2.erode(img, kernel, iterations=1)
dilation = cv2.dilate(img, kernel, iterations=1)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(15, 10))

titles = ['Original', 'Erosion', 'Dilation', 'Opening', 'Closing']
images = [img, erosion, dilation, opening, closing]

for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i], fontsize=14)
    plt.axis('off')

plt.tight_layout()
plt.savefig("result_image1.png", bbox_inches='tight')




opening_it1 = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
opening_it2 = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
opening_it3 = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=3)
opening_it4 = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=4)

plt.figure(figsize=(15, 4))

op_titles = ['Opening (it=1)', 'Opening (it=2)', 'Opening (it=3)', 'Opening (it=4)']
op_images = [opening_it1, opening_it2, opening_it3, opening_it4]

for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(op_images[i], cmap='gray')
    plt.title(op_titles[i], fontsize=14)
    plt.axis('off')

plt.tight_layout()
plt.suptitle("Result", fontsize=16, y=1.08)
plt.savefig("result_image2.png", bbox_inches='tight') 


