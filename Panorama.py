import cv2
import os

path = 'Resources/Images' ## directory path where images are stored
images = [] ## list of images
myList = os.listdir(path) ## create list of images
print(f'Total no of images detected {len(myList)}')
## loop through images
for imgN in myList:
    curImg = cv2.imread(f'{path}/{imgN}') ## read image from file
    curImg = cv2.resize(curImg, (0,0), None, 0.2, 0.2) ## resize image
    images.append(curImg) ## add image to list

stitcher = cv2.Stitcher.create() ## create stritcher object
(status,result) = stitcher.stitch(images) ## stitch images together into panorama
if (status == cv2.STITCHER_OK):
    print('Panorama Generated')
    cv2.imshow(path, result) ## display panorama
    cv2.waitKey(1)
else:
    print('Panorama Generation Unsuccessful')

cv2.waitKey(1)
