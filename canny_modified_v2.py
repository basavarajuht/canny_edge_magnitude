import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage import color
def cannyEdge(img, sigma):
    size = int(2*(np.ceil(3*sigma))+1)
    x, y = np.meshgrid(np.arange(-size/2+1, size/2+1),
                       np.arange(-size/2+1, size/2+1))
    normal = 1 / (2.0 * np.pi * sigma**2)
    kernel = np.exp(-(x**2+y**2) / (2.0*sigma**2)) / \
        normal  # calculating gaussian filter

    kern_size, gauss = kernel.shape[0], np.zeros_like(img, dtype=float)

    for i in range(img.shape[0]-(kern_size-1)):
        for j in range(img.shape[1]-(kern_size-1)):
            window = img[i:i+kern_size, j:j+kern_size] * kernel
            gauss[i, j] = np.sum(window)

    kernel, kern_size = np.array(
        [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]), 3  # edge detection
    gx, gy = np.zeros_like(
        gauss, dtype=float), np.zeros_like(gauss, dtype=float)

    for i in range(gauss.shape[0]-(kern_size-1)):
        for j in range(gauss.shape[1]-(kern_size-1)):
            window = gauss[i:i+kern_size, j:j+kern_size]
            gx[i, j], gy[i, j] = np.sum(
                window * kernel.T), np.sum(window * kernel)

    magnitude = np.sqrt(gx**2 + gy**2)
    theta = ((np.arctan(gy/gx))/np.pi) * 180  # radian to degree conversion
    nms = np.copy(magnitude)

    theta[theta < 0] += 180

    # non maximum suppression; quantization and suppression done in same step
    for i in range(theta.shape[0]-(kern_size-1)):
        for j in range(theta.shape[1]-(kern_size-1)):
            if (theta[i, j] <= 22.5 or theta[i, j] > 157.5):
                if(magnitude[i, j] <= magnitude[i-1, j]) and (magnitude[i, j] <= magnitude[i+1, j]):
                    nms[i, j] = 0
            if (theta[i, j] > 22.5 and theta[i, j] <= 67.5):
                if(magnitude[i, j] <= magnitude[i-1, j-1]) and (magnitude[i, j] <= magnitude[i+1, j+1]):
                    nms[i, j] = 0
            if (theta[i, j] > 67.5 and theta[i, j] <= 112.5):
                if(magnitude[i, j] <= magnitude[i+1, j+1]) and (magnitude[i, j] <= magnitude[i-1, j-1]):
                    nms[i, j] = 0
            if (theta[i, j] > 112.5 and theta[i, j] <= 157.5):
                if(magnitude[i, j] <= magnitude[i+1, j-1]) and (magnitude[i, j] <= magnitude[i-1, j+1]):
                    nms[i, j] = 0

    return gauss, magnitude

img = io.imread('C:\\Users\\Lenovo\\Desktop\\16.png')
img = color.rgb2gray(img)
labels=img.shape
sigma=0.5
lower_thresh=50
upper_thresh=100
gauss, magnitude= cannyEdge(
        img, sigma)
plt.imshow(magnitude, cmap='gray')
plt.show()

result = np.zeros((labels), np.uint8)
res = np.where(magnitude >= 0.5)
result[res[0], res[1]] = 255
plt.imshow(result, cmap='gray')
plt.show()
io.imsave('canny_5_5.jpg', result)



