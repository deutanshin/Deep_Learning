import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_entropy(data):

    hist, _ = np.histogram(data.flatten(), bins=511, range=(-255, 255))
    
    p = hist / np.sum(hist)
    p = p[p > 0]  

    entropy = -np.sum(p * np.log2(p))
    return entropy

def to_display(error_img):
    return np.clip(error_img + 128, 0, 255).astype(np.uint8)

image_path = "../img/test.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


s = image.astype(np.float32)
rows, cols = s.shape

u_D = np.zeros_like(s)


u_D[1:, 1:] = s[1:, 1:] - 0.5 * (s[:-1, 1:] + s[1:, :-1])

u_D[0, :] = s[0, :]
u_D[:, 0] = s[:, 0]

entropy_orig = calculate_entropy(s) 
std_orig = np.std(s)
    
entropy_D = calculate_entropy(u_D) 
std_D = np.std(u_D) 
print(f"Original Image -> Entropy: {entropy_orig:.4f}, Std: {std_orig:.4f}")
print(f"Diagonal Pred  -> Entropy: {entropy_D:.4f}, Std: {std_D:.4f}")

plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(to_display(u_D), cmap='gray')
plt.title('Diagonal Prediction Error')
plt.axis('off')


plt.subplot(2, 2, 3)
plt.hist(s.flatten(), bins=256, range=(0, 255), color='blue', alpha=0.7)
plt.title('Original Histogram')

plt.subplot(2, 2, 4)
plt.hist(u_D[1:, 1:].flatten(), bins=255, range=(-127, 128), color='purple', alpha=0.7)
plt.title('Diagonal Prediction Error Histogram')

plt.tight_layout()
plt.savefig('diagonal_prediction_result.png')

"""
대각선 예측을 통해 인접 픽셀 간의 공간적 중복성을 제거한 결과, 원본 대비 엔트로피와 표준편차가 크게 감소하며 
히스토그램이 0 근처로 좁게 집중되는 정량적 변화를 확인했습니다. 
시각적으로는 예측 모델이 대응하기 어려운 급격한 밝기 변화 구간인 에지(Edge) 성분들이 오차 이미지에서 윤곽선 형태로 도드라지게 나타납니다.
"""