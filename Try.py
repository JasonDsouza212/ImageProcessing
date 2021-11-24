import cv2
import matplotlib.pyplot as plt
img = cv2.imread('rick.jpg') 
#adding noise to the image
#removing noise from a image
#converting color image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#converting image into binary image
thresh=127
imgbw=cv2.threshold(gray_image,thresh,255,cv2.THRESH_BINARY)[1]
imd=cv2.imread('sp_noise.jpg') 
st = cv2.fastNlMeansDenoisingColored(imd,None,10,10,7,21)
denoise_1 = cv2.fastNlMeansDenoisingColored(imd,None,3,3,7,21) 


#displaying all output
plt.subplot(221),plt.imshow(image),plt.title('Original Image')
plt.subplot(222),plt.imshow(denoise_1),plt.title('Original')

# plt.subplot(222),plt.imshow(gray_image,cmap='gray'),plt.title('Grayscale Image')
# plt.subplot(223),plt.imshow(imgbw,cmap='gray'),plt.title('Binary Image')
# plt.subplot(224),plt.hist(img.ravel(),256,[0,256]),plt.title('Histogram Image')
plt.show()