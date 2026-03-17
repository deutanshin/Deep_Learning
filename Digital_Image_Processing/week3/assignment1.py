import cv2

img = cv2.imread('img/dogs.jpg')

factor = 2

h, w, c = img.shape

for i in range(1, 4):
    resized = cv2.resize(img, (w // (factor ** i), h // (factor ** i)))

    cv2.imwrite(f'dogs_resized{factor ** i}.jpg', resized)
