import cv2
import matplotlib.pyplot as plt

video =  cv2.VideoCapture('./IMG_2809.MP4')
num = 0
save_step = 10
while True:
    ret, frame = video.read()
    if not ret:
        break
    if num % save_step == 0:
        cv2.imwrite('frameg%d.jpg' % num, frame)
    num += 1
