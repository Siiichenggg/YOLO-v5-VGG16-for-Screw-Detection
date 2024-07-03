from PIL import Image
import os

import matplotlib.pyplot as plt

# 指定的目录
directory = '/Users/lusicheng/Desktop/yolov5-7.0/runs/detect/exp4'

# 获取目录下的所有文件
files = os.listdir(directory)

# 过滤出.jpg和.png文件
image_files = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]

# 打印.jpg和.png文件的数量
imgnum = len(image_files)

# 保存文件夹的初始名称
save_dir_name = 'exp1'

for i in range(imgnum):
    # 构造图像文件的路径
    i = i+1
    img_path = f'/Users/lusicheng/Desktop/yolov5-7.0/runs/detect/exp/{i}.jpg'

    # 检查图像文件是否存在
    if not os.path.exists(img_path):
        continue

    # 打开图像文件
    img = Image.open(img_path)

    # 获取图像的宽度和高度
    img_width, img_height = img.size

    # 构造标签文件的路径
    label_path = f'/Users/lusicheng/Desktop/yolov5-7.0/runs/detect/exp/labels/{i}.txt'

    # 检查标签文件是否存在
    if not os.path.exists(label_path):
        continue

    # 打开并读取文本文件
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # 分割行以获取坐标
        values = line.split()
        class_id = int(values[0])
        x_center, y_center, width, height = map(float, values[1:])

        # 将归一化的坐标转换为像素值
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height

        # 计算边界框的左上角和右下角坐标
        left = x_center - width / 2
        top = y_center - height / 2
        right = x_center + width / 2
        bottom = y_center + height / 2

        img_cropped = img.crop((left, top, right, bottom))



        # 创建保存文件夹的路径
        save_dir = f'/Users/lusicheng/Desktop/yolov5-7.0/saveshots/{save_dir_name}/{class_id}'
        os.makedirs(save_dir, exist_ok=True)
        img_cropped.save(f'{save_dir}/cropped_image_{class_id}.jpg')

    # 递增保存文件夹的名称
    save_dir_name = f'exp{i+1}'
