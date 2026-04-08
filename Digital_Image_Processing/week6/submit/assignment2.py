import cv2
import numpy as np

def apply_laplacian(dft_img):
    M, N = dft_img.shape
    u = np.arange(M)
    v = np.arange(N)
    u, v = np.meshgrid(u - M//2, v - N//2, indexing='ij')
    
    H = -(u**2 + v**2)
    return dft_img * H, H

f_xy = cv2.imread('../img/sample.jpg', 0)
M, N = f_xy.shape
P, Q = 2*M, 2*N

padded = np.zeros((P, Q), dtype=np.float32)
padded[:M, :N] = f_xy
for x in range(P):
    for y in range(Q):
        padded[x, y] *= ((-1)**(x+y))

F_uv = np.fft.fft2(padded)

G_uv, H_lap = apply_laplacian(F_uv)

g_xy_full = np.fft.ifft2(G_uv).real
for x in range(P):
    for y in range(Q):
        g_xy_full[x, y] *= ((-1)**(x+y))

g_xy = g_xy_full[:M, :N]


g_normalized = cv2.normalize(g_xy, None, 0, 255, cv2.NORM_MINMAX)
g_for_subtraction = g_normalized - 127

sharpened = f_xy.astype(np.float32) - g_for_subtraction
sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

cv2.imwrite('task2_filtered_g.jpg', g_normalized.astype(np.uint8))
cv2.imwrite('task2_sharpened.jpg', sharpened)  