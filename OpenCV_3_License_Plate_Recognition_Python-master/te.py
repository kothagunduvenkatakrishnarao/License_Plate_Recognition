
# testing module - te.py


#from pytesseract import image_to_string
#from PIL import Image

#import pytesseract
#pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract'
#img = Image.open('C:/Users/sriha/Desktop/sri.png').convert('L')
#text=pytesseract.image_to_string(img)
#print(img)
#print(text)

#from pytesseract import image_to_string
#from PIL import Image
import cv2
import numpy as np
import math
#im = Image.open(r'C:\Users\<user>\Downloads\dashboard-test.jpeg')
#print(im)
import os
#print(image_to_string(im))


#import pytesseract
#from PIL import Image

#pytesseract.pytesseract.tesseract_cmd="C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"
#im=Image.open("C:\\Users\\<user>\\Desktop\\ro\\capt.png")
#print(pytesseract.image_to_string(im,lang='eng'))

from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract'
#image = Image.open(r'C:\Users\sriha\Desktop\demo\OpenCV_3_License_Plate_Recognition_Python-master\my.png').convert('L')
#height, width, numChannels = img.shape
#cv2.imshow('kkk',img)
#ndrr= np.zeros((height, width, 3), np.uint8)
#ndarr=extractValue1(img)
#img11 = Image.fromarray(ndarr, 'RGB')
#img11.save('my.png')
#img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
image = cv2.imread('my.png')

#col = Image.open('my.png')
#gray = col.convert('L')
#bw = gray.point(lambda x: 0 if x<150 else 250, '1')
#cleaned_image_name =  'col'+ '_cleaned.jpg'
#bw.save('cleaned_image_name.jpg')

#print(type(cleaned_image_name))
#from matplotlib import pyplot as plt

#dst = cv2.fastNlMeansDenoisingColored('cleaned_image_name.jpg',None,10,10,7,21)

#plt.subplot(121),plt.imshow(image)
#plt.subplot(122),plt.imshow(dst)
#plt.show()




#gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)


#img = cv.bilateralFilter(img,9,75,75)


#gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#gray = cv2.medianBlur(gray, 3)
#kernel = np.ones((5,5),np.uint8)
#gray = cv2.dilate(gray,kernel,iterations = 1)
#filename = "{}.png".format(os.getpid())
#cv2.imwrite(filename, gray)

#grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#(thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY) #gray-grayImage
#cv2.imshow('Black white image', blackAndWhiteImage)
#img2 = Image.fromarray(blackAndWhiteImage, 'RGB')
#img2.save("bandw.png")



#def get_string(img_path):
    # Read image using opencv
#img = cv2.imread(r'C:\Users\sriha\Desktop\demo\OpenCV_3_License_Plate_Recognition_Python-master\my.png')

    # Extract the file name without the file extension
##file_name = os.path.basename(img_path).split('.')[0]
#file_name = file_name.split()[0]

    # Create a directory for outputs
#output_path = os.path.join(output_dir, file_name)
 #   if not os.path.exists(output_path):
  #      os.makedirs(output_path)


img = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
 # Convert to gray
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
kernel = np.ones((1, 1), np.uint8)
img = cv2.dilate(img, kernel, iterations=1)
img = cv2.erode(img, kernel, iterations=1)
    # Apply blur to smooth out the edges
img = cv2.GaussianBlur(img, (5, 5), 0)
img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # Save the filtered image in the output directory
#save_path = os.path.join(output_path, file_name + "_filter_" + str(method) + ".jpg")
#cv2.imwrite('sri'.png', img)

    # Recognize text with tesseract for python
#result = pytesseract.image_to_string(img, lang="eng")
#return result


text = pytesseract.image_to_string(img)

print(text)




#text=pytesseract.image_to_string(img)
#print(text)
#print(len(text))

#import numpy as np

#w, h = 512, 512
#data = np.zeros((h, w, 3), dtype=np.uint8)
#data[256, 256] = [255, 0, 0]
#img = Image.fromarray(data, 'RGB')
#img.save('my.png')
#img.show() 