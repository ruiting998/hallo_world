import cv2
from classifal_moblie_image import get_image_classifal
from classifal_moblie_image import init_img_classify
#初始化模型
init_img_classify()


print("开始")
image = cv2.imread("F:\\train\\nocover\\000166.jpg")
flag = get_image_classifal(image)
print(flag)
image = cv2.imread("F:\\train\\nocover\\000166.jpg")
flag = get_image_classifal(image)
print(flag)
image = cv2.imread("F:\\train\\nocover\\000166.jpg")
flag = get_image_classifal(image)
print(flag)
image = cv2.imread("F:\\train\\nocover\\000166.jpg")
flag = get_image_classifal(image)
print(flag)