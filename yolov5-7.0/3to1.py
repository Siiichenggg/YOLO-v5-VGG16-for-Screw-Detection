import os
import cv2

folder_path = "/Users/lusicheng/Desktop/xunlianji/huasi"
save_path = "/Users/lusicheng/Desktop/xunlianji3to1"

# 确保保存路径存在
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # 只处理图片文件
        # 构建完整的文件路径
        file_path = os.path.join(folder_path, filename)
        
        # 读取彩色图像
        img = cv2.imread(file_path)
        
        # 转换为灰度图
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 构建保存文件的路径
        save_file_path = os.path.join(save_path, filename)
        
        # 保存灰度图
        cv2.imwrite(save_file_path, gray_img)

print("所有图片已成功转换为灰度图并保存。")