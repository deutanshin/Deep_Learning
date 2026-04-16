import numpy as np
import cv2
import matplotlib.pyplot as plt


def write_img(filename, img_data):
    img = (img_data * 255).astype(np.uint8)
    cv2.imwrite(filename, img)

def get_atmospheric_turbulence_filter(shape, k=0.0025):
    rows, cols = shape
    
    U, V = np.meshgrid(np.arange(-cols//2, cols//2), np.arange(-rows//2, rows//2))

    D_squared = U**2 + V**2
    H = np.exp(-k * (D_squared) ** (5 / 6))
    
    return H


import numpy as np

def get_laplacian_freq(rows, cols):
    p = np.array([[ 0, -1,  0],
                  [-1,  4, -1],
                  [ 0, -1,  0]])
    
    p_padded = np.zeros((rows, cols))
    p_padded[0:3, 0:3] = p
    
    p_shifted = np.roll(p_padded, shift=(-1, -1), axis=(0, 1))
    
    P_ = np.fft.fft2(p_shifted)
    P = np.fft.fftshift(P_)
    
    return P

def cls_filter(G, H, gamma=0.01):
    rows, cols = G.shape
    
    H_conj = np.conj(H)
    
    H_mag_sq = np.abs(H)**2
    
    P = get_laplacian_freq(rows, cols)
    P_mag_sq = np.abs(P)**2
    
    denominator = H_mag_sq + (gamma * P_mag_sq)
    
    epsilon = 1e-8
    denominator = np.where(denominator == 0, epsilon, denominator)
    
    F_hat = (H_conj / denominator) * G
    
    return F_hat


def inverse_filter(G, H, epsilon=1e-3):
    H_inv = np.copy(H)
    H_inv[np.abs(H_inv) < epsilon] = epsilon

    F_hat = G / H_inv
    return F_hat


img = cv2.imread('../img/test.jpg', cv2.IMREAD_GRAYSCALE)
img = img / 255.0
rows, cols = img.shape

F = np.fft.fftshift(np.fft.fft2(img))

H = get_atmospheric_turbulence_filter((rows, cols), k=0.0025)
G = F * H

noise = np.random.normal(0, 0.01, (rows, cols))
noise_fft = np.fft.fftshift(np.fft.fft2(noise))
G_noisy = G + noise_fft

F_inv = inverse_filter(G_noisy, H)

F_cls_001 = cls_filter(G_noisy, H, gamma=0.001)
F_cls_01 = cls_filter(G_noisy, H, gamma=0.01)
F_cls_1 = cls_filter(G_noisy, H, gamma=0.1)

def to_spatial(freq_img):
    spatial_img = np.abs(np.fft.ifft2(np.fft.ifftshift(freq_img)))
    return np.clip(spatial_img, 0, 1)

img_blurred = to_spatial(G_noisy)
img_inv = to_spatial(F_inv)
img_cls_001 = to_spatial(F_cls_001)
img_cls_01 = to_spatial(F_cls_01)
img_cls_1 = to_spatial(F_cls_1)

write_img('degraded_img.jpg', img_blurred)
write_img('invers_filter_result.jpg', img_inv)
write_img('cls_filter_g_dot1.jpg', img_cls_01)

'''
감마값이 낮으면 엣지는 날카롭지만 노이즈가 심해지고 
감마값이 높아지면 노이즈는 줄어들지만 이미지가 흐려지는 트레이드 오프를 확인할 수 있습니다
'''
