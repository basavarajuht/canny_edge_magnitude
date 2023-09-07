import cv2
import numpy as np
import matplotlib.pyplot as plt
img = plt.imread('C:\\Users\\Lenovo\\Desktop\\16.png')

kernel = cv2.getGaussianKernel(11, 2)

kernel = kernel.dot(kernel.T)

img_gaussian = cv2.filter2D(img, -1, kernel)

plt.figure(1)
plt.subplot(121)
plt.imshow(img)
plt.title("Original Image")

plt.subplot(122)
plt.imshow(img_gaussian)
plt.title("Gaussian Filter Image")
plt.show()

img_gaussian = np.float64(img_gaussian)

mask_x = np.zeros((2, 1))
mask_x[0] = -1
mask_x[1] = 1

I_x = cv2.filter2D(img_gaussian, -1, mask_x)
mask_y = mask_x.T
I_y = cv2.filter2D(img_gaussian, -1, mask_y)

Gm = (I_x ** 2 + I_y ** 2) ** 0.5
Gd = np.rad2deg(np.arctan2(I_y, I_x))
cv2.imshow('magnitude',Gm)

labels=Gm.shape
result = np.zeros((labels), np.uint8)
res = np.where(Gm >= 0.006)
result[res[0], res[1]] = 255
cv2.imshow('result',result)
cv2.waitKey(0)
cv2.destroyAllWindows()

