import cv2
import imageio

img = cv2.imread("individualimage.png")

images = []
for i in range(7):
    for j in range(6):
        images.append(img[512 * i:512 * (i + 1), 512 * j:512 * (j + 1)])
images = images[:-6]
imageio.mimsave('gif.gif',  images+images[::-1] ,fps=100)