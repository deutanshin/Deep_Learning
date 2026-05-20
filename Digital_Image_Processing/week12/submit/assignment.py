import cv2
import numpy as np
import matplotlib.pyplot as plt

f1 = cv2.imread("../img/frame1.png", cv2.IMREAD_GRAYSCALE)
f2 = cv2.imread("../img/frame2.png", cv2.IMREAD_GRAYSCALE)

h, w = f1.shape

b_size = 16
search = 15

s_x = w // 2
s_y = h //2

t_block = f2[s_y:s_y + b_size, s_x:s_x + b_size]

min_sse = float('inf')
best_dy, best_dx = 0, 0

for dy in range(-search, search + 1):
    for dx in range(-search, search + 1):
        ref_y = s_y + dy
        ref_x = s_x + dx


        if ref_y >= 0 and ref_y + b_size <= h and ref_x >= 0 and ref_x + b_size <= w:
            candidate = f1[ref_y:ref_y + b_size, ref_x:ref_x + b_size]

            diff = t_block.astype(np.int32) - candidate.astype(np.int32)
            sse = np.sum(np.square(diff))

            if sse < min_sse:
                min_sse = sse
                best_dy = dy
                best_dx = dx


vis_img = cv2.cvtColor(f1, cv2.COLOR_GRAY2BGR)

print(best_dy, best_dx)
print(min_sse)

cv2.rectangle(vis_img, (s_x, s_y), (s_x + b_size, s_y + b_size), (0, 0, 255), 1)

matched_y, matched_x = s_y + best_dy, s_x + best_dx
cv2.rectangle(vis_img, (matched_x, matched_y), (matched_x + b_size, matched_y + b_size), (0, 255, 0), 1)

plt.figure(figsize=(8, 8))
plt.title("Motion Estimation: Red(Original) -> Green(Matched)")
plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.savefig("motion_estimation_result.png", bbox_inches='tight')


# 22261041 신동주 과제

# (dx, dy) : -6 -15
# min_SSE : 39286
