import os
import cv2

def process_files():
    folder_path = "/Users/lusicheng/Desktop/summerresearch2024/Visiondetect/yolov5-7.0/runs/detect/exp"
    save_folder_path = "/Users/lusicheng/Desktop/xunlianji"

    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    file_list = os.listdir(folder_path)
    img_files = [file for file in file_list if file.endswith((".jpg", ".png", ".jpeg"))]

    if not img_files:
        print("No image files found.")
        return

    for img_file in img_files:
        img_path = os.path.join(folder_path, img_file)
        txt_file = img_file.rsplit('.', 1)[0] + '.txt'
        txt_path = os.path.join(folder_path, txt_file)

        if not os.path.exists(txt_path):
            print(f"No corresponding txt file found for {img_file}")
            continue

        with open(txt_path, "r") as f:
            lines = f.readlines()

        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Unable to read image {img_path}")
            continue

        img_height, img_width = img.shape[:2]

        for line in lines:
            line = line.strip()
            parts = line.split()
            if len(parts) == 5:
                try:
                    class_num = int(parts[0])
                    coordinates = list(map(float, parts[1:]))

                    x_center, y_center, width, height = coordinates
                    x1 = int((x_center - width / 2) * img_width)
                    y1 = int((y_center - height / 2) * img_height)
                    x2 = int((x_center + width / 2) * img_width)
                    y2 = int((y_center + height / 2) * img_height)

                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)

                    x1 = max(0, min(img_width - 1, x1))
                    y1 = max(0, min(img_height - 1, y1))
                    x2 = max(0, min(img_width, x2))
                    y2 = max(0, min(img_height, y2))

                    if x1 >= x2 or y1 >= y2:
                        print(f"Invalid crop coordinates for {img_file}: ({x1}, {y1}, {x2}, {y2})")
                        continue

                    cropped_img = img[y1:y2, x1:x2]
                    if cropped_img.size == 0:
                        print(f"Error: Cropped image is empty for coordinates ({x1}, {y1}, {x2}, {y2}) in {img_file}")
                        continue

                    cropped_img_path = os.path.join(save_folder_path, f"cropped_{img_file}")
                    cv2.imwrite(cropped_img_path, cropped_img)
                    print(f"Saved cropped image to {cropped_img_path}")

                except ValueError:
                    print(f"Error: Line '{line}' in file '{txt_file}' contains invalid numbers")

if __name__ == "__main__":
    process_files()