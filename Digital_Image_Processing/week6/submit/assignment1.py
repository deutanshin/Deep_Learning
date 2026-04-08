import cv2
import numpy as np

def get_dist_matrix(M, N):
    u = np.arange(M)
    v = np.arange(N)
    u, v = np.meshgrid(u - M//2, v - N//2, indexing='ij')
    return np.sqrt(u**2 + v**2)

def apply_blpf(dft_img, d0, n):
    M, N = dft_img.shape
    D = get_dist_matrix(M, N)
    H = 1 / (1 + (D / d0)**(2 * n))
    return dft_img * H, H

def apply_glpf(dft_img, d0):
    M, N = dft_img.shape
    D = get_dist_matrix(M, N)
    H = np.exp(-(D**2) / (2 * (d0**2)))
    return dft_img * H, H

img = cv2.imread('../img/sample.jpg', 0)
M, N = img.shape
P, Q = 2*M, 2*N
padded = np.zeros((P, Q), dtype=np.float32)
padded[:M, :N] = img

for x in range(P):
    for y in range(Q):
        padded[x, y] *= ((-1)**(x+y))

dft_img = np.fft.fft2(padded)

low_pass_blpf, h_blpf = apply_blpf(dft_img, 30, 2)
low_pass_glpf, h_glpf = apply_glpf(dft_img, 30)

def restore_image(filtered_dft, P, Q, M, N):
    idft_img = np.fft.ifft2(filtered_dft).real
    for x in range(P):
        for y in range(Q):
            idft_img[x, y] *= ((-1)**(x+y))
    return np.clip(idft_img[:M, :N], 0, 255).astype(np.uint8)

result_blpf = restore_image(low_pass_blpf, P, Q, M, N)
result_glpf = restore_image(low_pass_glpf, P, Q, M, N)

cv2.imwrite('result_BLPF.jpg', result_blpf)
cv2.imwrite('result_GLPF.jpg', result_glpf)

# Butterworth 필터는 차단 주파수 근처에서 완만한 전이를 보이고 엣지를 부드럽게 처리하지만
# 미세한 링잉 현상이 관찰될 수 있다. 반면 가우시안 필터는 링잉현상이 전혀 발생하지 않으며
# 더 자연스럽고 부드러운 블러 효과를 만들었다.