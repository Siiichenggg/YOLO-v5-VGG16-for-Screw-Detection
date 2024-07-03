import os
import cv2

def process_files():
    folder_path = "/Users/lusicheng/Desktop/summerresearch2024/Visiondetect/yolov5-7.0/runs/detect/exp"
    save_folder_path = "/Users/lusicheng/Desktop/summerresearch2024/Visiondetect/VGG16/predictdata"

    # 确保保存文件夹存在
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    # 获取exp文件夹内的文件列表
    file_list = os.listdir(folder_path)

    # 读取文件夹内的txt文件
    txt_files = [file for file in file_list if file.endswith(".txt")]
    crops = []  # 用于保存裁剪信息

    for txt_file in txt_files:
        file_path = os.path.join(folder_path, txt_file)
        with open(file_path, "r") as f:
            lines = f.readlines()  # 逐行读取txt文件内容
            for line in lines:
                line = line.strip()  # 去除行尾的换行符和空格
                parts = line.split()
                if len(parts) == 5:
                    try:
                        class_num = int(parts[0])
                        coordinates = list(map(float, parts[1:]))
                        crops.append((class_num, coordinates))
                    except ValueError:
                        print(f"Error: Line '{line}' in file '{txt_file}' contains invalid numbers")
                else:
                    print(f"Error: Line '{line}' in file '{txt_file}' does not contain exactly 5 values")

    # 读取图片
    img_files = [file for file in file_list if file.endswith((".jpg", ".png", ".jpeg"))]
    if not img_files:
        print("No image files found.")
    else:
        for img_file in img_files:
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Error: Unable to read image {img_path}")
                continue

            img_height, img_width = img.shape[:2]

            for i, (class_num, coordinates) in enumerate(crops):
                x_center, y_center, width, height = map(float, coordinates)
                
                # Convert YOLO format (center x, center y, width, height) to (x1, y1, x2, y2)
                x1 = int((x_center - width / 2) * img_width)
                y1 = int((y_center - height / 2) * img_height)
                x2 = int((x_center + width / 2) * img_width)
                y2 = int((y_center + height / 2) * img_height)
                
                # Ensure x1 < x2 and y1 < y2
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # Check if coordinates are within the image boundaries
                x1 = max(0, min(img_width - 1, x1))
                y1 = max(0, min(img_height - 1, y1))
                x2 = max(0, min(img_width, x2))
                y2 = max(0, min(img_height, y2))

                # Check if the coordinates result in a valid crop
                if x1 >= x2 or y1 >= y2:
                    print(f"Invalid crop coordinates for {img_file}: ({x1}, {y1}, {x2}, {y2})")
                    continue

                cropped_img = img[y1:y2, x1:x2]
                
                if cropped_img.size == 0:
                    print(f"Error: Cropped image is empty for coordinates ({x1}, {y1}, {x2}, {y2}) in {img_file}")
                    continue

                # 保存裁剪后的图片到新文件
                cropped_img_path = os.path.join(save_folder_path, f"cropped_{i + 1}_{img_file}")
                cv2.imwrite(cropped_img_path, cropped_img)
                print(f"Saved cropped image to {cropped_img_path}")

if __name__ == "__main__":
    process_files()
