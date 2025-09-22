import numpy as np
import matplotlib.pyplot as plt


img = plt.imread("tiger.png")
img = img[:,:,0].copy()
print(img.shape)
print(img.dtype)
plt.figure()
plt.imshow(img, cmap="gray")
plt.axis('off')
plt.show()

imga = np.clip(img * 3 ,0, 1) 
plt.imshow(imga, cmap="gray")
plt.axis('off')
plt.show()

imgb = np.fliplr(img)
plt.imshow(imgb, cmap="gray")
plt.axis('off')
plt.show()

imgc = img[::20,::20]
plt.imshow(imgc, cmap="gray")
plt.axis('off')
plt.show()

imgd = np.zeros_like(img)
height, width = img.shape
start = width // 4
end = width // 2
imgd[:, start:end] = img[:, start:end]
plt.imshow(imgd)
plt.axis('off')
plt.imshow(imgd, cmap="gray")
plt.show()

