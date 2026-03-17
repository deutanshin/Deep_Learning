import cv2

img = cv2.imread('img/color_img.jpg')

b, g, r = cv2.split(img)

cv2.imshow("Blue: ", b)
cv2.imshow("Green: ", g)
cv2.imshow("Red: ", r)


cv2.waitKey(0)
cv2.destroyAllWindows()