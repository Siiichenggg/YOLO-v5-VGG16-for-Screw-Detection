import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
import cv2
import os
import subprocess
import threading

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("GUI Example")
        self.cap = cv2.VideoCapture(0)
        
        # 创建图像显示区
        self.image_label = tk.Label(root, text="图像预览页面", bg="white", width=800, height=400)
        self.image_label.grid(row=0, column=0, padx=10, pady=10)

        # 创建拍照按钮
        self.photo_button = tk.Button(root, text="拍照按钮", command=self.take_photo)
        self.photo_button.grid(row=0, column=1, padx=10, pady=10)

        # 创建运行第二个脚本的按钮
        self.script2_button = tk.Button(root, text="2", command=lambda: self.run_script_in_thread(2))
        self.script2_button.grid(row=1, column=0, padx=10, pady=10)

        # 创建运行第三个脚本的按钮
        self.script3_button = tk.Button(root, text="3", command=lambda: self.run_script_in_thread(3))
        self.script3_button.grid(row=1, column=1, padx=10, pady=10)

        # 创建带滚动条的文本框
        self.text_box = ScrolledText(root, width=80, height=20)
        self.text_box.grid(row=2, column=0, columnspan=2, padx=10, pady=10)
        
        # 开始摄像头实时预览
        self.update_frame()
        
        # 开始监控目标目录
        self.target_dir = "/Users/lusicheng/Desktop/summerresearch2024/Visiondetect/VGG16/predictdata"
        self.detected_files = set()  # 已检测文件的集合
        self.monitor_directory()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            # 将图像从 BGR 转换为 RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转换为 Image 对象
            img = Image.fromarray(frame)
            # 调整图像大小以适应标签
            img = img.resize((640, 480), Image.LANCZOS)
            # 转换为 ImageTk 对象
            imgtk = ImageTk.PhotoImage(image=img)
            # 更新标签中的图像
            self.image_label.config(image=imgtk)
            self.image_label.image = imgtk
        self.root.after(10, self.update_frame)

    def take_photo(self):
        if hasattr(self, 'current_frame'):
            # 保存图像到指定路径
            save_dir = "/Users/lusicheng/Desktop/summerresearch2024/Visiondetect/yolov5-7.0/data/images"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, "photo.jpg")
            cv2.imwrite(save_path, self.current_frame)
            self.display_image(save_path)
            self.text_box.insert(tk.END, f"拍照完成：{save_path}\n")
            self.run_script_in_thread(1)

    def display_image(self, file_path):
        # 打开图片并调整大小以适应标签
        img = Image.open(file_path)
        img = img.resize((640, 480), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)

        # 更新标签中的图像
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo

    def run_script_in_thread(self, script_number):
        thread = threading.Thread(target=self.run_script, args=(script_number,))
        thread.start()

    def run_script(self, script_number):
        scripts = [
            "/Users/lusicheng/Desktop/summerresearch2024/Visiondetect/yolov5-7.0/detect.py",
            "/Users/lusicheng/Desktop/summerresearch2024/Visiondetect/VGG16/crop.py",
            "/Users/lusicheng/Desktop/summerresearch2024/Visiondetect/VGG16/VGG16_train/VGG16Predict.py"
        ]
        
        script = scripts[script_number - 1]
        script_dir = os.path.dirname(script)  # 获取脚本所在目录
        try:
            result = subprocess.run(
                ['python3', script],  # 使用 python3 以确保兼容性
                check=True, 
                capture_output=True, 
                text=True,
                cwd=script_dir  # 设置工作目录为脚本所在目录
            )
            self.text_box.insert(tk.END, f"运行 {script} 完成:\n{result.stdout}\n")
            self.text_box.see(tk.END)
        except subprocess.CalledProcessError as e:
            self.text_box.insert(tk.END, f"运行 {script} 出错:\n{e.stderr}\n")

    def monitor_directory(self):
        if os.path.exists(self.target_dir):
            files = os.listdir(self.target_dir)
            new_files = [file for file in files if file.endswith(('.png', '.jpg', '.jpeg')) and file not in self.detected_files]
            if new_files:
                for file in new_files:
                    self.detected_files.add(file)
                    self.text_box.insert(tk.END, f"检测到新图片文件: {file}\n")
                    self.text_box.see(tk.END)
        self.root.after(1000, self.monitor_directory)

    def on_closing(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
