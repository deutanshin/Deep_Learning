import cv2

img1 = cv2.imread('img/color_img.jpg')
img2 = cv2.imread('img/color_img.jpg', cv2.IMREAD_GRAYSCALE)

print("color img shape")
print(img1.shape)

print("gray scaled img shape")
print(img2.shape)



cv2.imshow('window1', img2)
cv2.imwrite('img/gray.jpg', img2)
cv2.waitKey()

cv2. destroyAllWindows()

