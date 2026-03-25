import numpy as np
import cv2
import os


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

# 가우시안 블러를 적용하여 샘플링 전 고주파 성분 제거 

k_size = (17, 17) 
sigma = 4
blurred_img = cv2.GaussianBlur(img, k_size, sigma)

# 3. 서브샘플링 진행 
anti_aliased_sub = blurred_img[::stride, ::stride]

# 4. 관찰을 위해 원래 크기로 복원 (Nearest Neighbor) 
anti_aliased_restored = cv2.resize(anti_aliased_sub, (w, h), interpolation=cv2.INTER_NEAREST)

# 5. 결과 저장 
cv2.imwrite(f'submit_img/assignment2_best_result4.png', anti_aliased_restored)


"""
filter strength가 클 수록 aliasing 발생이 줄어들지만 이미지가 전반적으로 흐려져

이미지의 왜곡을 최대한 줄이면서도 선명함을 유지할 수 있는 지점을 찾는 것이 중요해 보인다.
"""