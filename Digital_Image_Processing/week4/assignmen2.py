import numpy as np
import cv2
import os

# 저장 폴더 생성
save_path = 'submit_img'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def generate_zone_plate(size=512):
    """중심에서 멀어질수록 주파수가 증가하는 Zone Plate 이미지 생성"""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    # 신호 생성: sin(100 * pi * R^2)
    zone_plate = np.sin(100 * np.pi * R**2)
    # OpenCV 처리를 위해 0~255 uint8 스케일로 변환
    zone_plate = ((zone_plate + 1) / 2 * 255).astype(np.uint8)
    return zone_plate

# 1. 원본 이미지 생성 및 설정
img = generate_zone_plate(512)
h, w = img.shape
stride = 4

# 2. [Experiment B] Anti-aliasing (Sampling after applying Lowpass Filter)
# 가우시안 블러를 적용하여 샘플링 전 고주파 성분 제거
# Tip: (15, 15), sigma=5는 실습 자료의 예시이며, 값을 조절해 최적의 결과를 찾아보세요.
k_size = (15, 15)
sigma = 5
blurred_img = cv2.GaussianBlur(img, k_size, sigma)

# 3. 서브샘플링 진행
anti_aliased_sub = blurred_img[::stride, ::stride]

# 4. 관찰을 위해 원래 크기로 복원 (Nearest Neighbor)
anti_aliased_restored = cv2.resize(anti_aliased_sub, (w, h), interpolation=cv2