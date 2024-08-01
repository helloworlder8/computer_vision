import cv2


old_image = cv2.imread('000000000001_labeled.jpg')
old_image = cv2.cvtColor(old_image, cv2.COLOR_BGR2RGB)
print(old_image.shape)