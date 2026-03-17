import cv2

print(cv2.__version__)

img = cv2.imread('color_img.jpg')

cv2.imshow('window1', img)

cv2.waitKey()

cv2.destroyAllWindows()