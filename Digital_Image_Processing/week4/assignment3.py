import cv2
import numpy as np

def manual_bilinear_2x(img):
    h, w, c = img.shape
    
    h_expanded = np.zeros((h, w * 2, c), dtype=np.uint8)
    h_expanded[:, ::2] = img
    
    for j in range(w - 1):
        v1 = h_expanded[:, 2*j].astype(int)
        v2 = h_expanded[:, 2*j + 2].astype(int)
        h_expanded[:, 2*j + 1] = ((v1 + v2) / 2).astype(np.uint8)
        
    result = np.zeros((h * 2, w * 2, c), dtype=np.uint8)
    result[::2, :] = h_expanded
    
    for i in range(h - 1):
        v1 = result[2*i, :].astype(int)
        v2 = result[2*i + 2, :].astype(int)
        result[2*i + 1, :] = ((v1 + v2) / 2).astype(np.uint8)
        
    return result

src = cv2.imread('img/pandas.jpg')

manual_res = manual_bilinear_2x(src)
opencv_res = cv2.resize(src, (src.shape[1]*2, src.shape[0]*2), interpolation=cv2.INTER_LINEAR)
diff = cv2.absdiff(manual_res, opencv_res)

cv2.imwrite('submit_img/assignment3_manual.png', manual_res)
cv2.imwrite('submit_img/assignment3_opencv.png', opencv_res)
cv2.imwrite('submit_img/assignment3_diff.png', diff)