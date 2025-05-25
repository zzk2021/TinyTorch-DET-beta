import os
import random

# 设置路径
image_dir = r"E:\data\coco128\coco128\images\train2017"
label_dir = r"E:\data\coco128\coco128\labels\train2017"
output_dir = r"E:\data\coco128\coco128"

# 获取所有图像文件名（支持常见扩展）
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(image_files)  # 打乱顺序以确保随机划分

# 划分训练/测试集（80% / 20%）
split_idx = int(len(image_files) * 0.8)
train_images = image_files[:split_idx]
test_images = image_files[split_idx:]

# 计算 max_length 和类别集合
max_length = 0
classes_set = set()
for image_file in image_files:
    base_name = os.path.splitext(image_file)[0]
    label_path = os.path.join(label_dir, base_name + ".txt")
    if not os.path.exists(label_path):
        continue
    with open(label_path, 'r') as label_f:
        lines = label_f.readlines()
        line_count = len(lines)
        max_length = max(max_length, line_count)
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:  # 确保至少有5个部分
                class_id = parts[0]
                classes_set.add(int(class_id))

num_classes = max(classes_set) + 1

# 写入标注文件函数
def write_annotation(output_path, image_list):
    with open(output_path, 'w') as out_f:
        out_f.write(f"{max_length} {num_classes}\n")  # 第一行写入 max_length 和 num_classes
        for image_file in image_list:
            base_name = os.path.splitext(image_file)[0]
            label_path = os.path.join(label_dir, base_name + ".txt")
            if not os.path.exists(label_path):
                print(f"警告：未找到标注文件 {label_path}")
                continue
            abs_image_path = os.path.abspath(os.path.join(image_dir, image_file))
            with open(label_path, 'r') as label_f:
                lines = label_f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id, xc, yc, w, h = parts
                    out_line = f"{abs_image_path} {class_id} {xc} {yc} {w} {h}\n"
                    out_f.write(out_line)

# 输出文件路径
train_anno_file = os.path.join(output_dir, "train_annotation.txt")
test_anno_file = os.path.join(output_dir, "test_annotation.txt")

write_annotation(train_anno_file, train_images)
write_annotation(test_anno_file, test_images)

print(f"训练集标注文件已保存至: {train_anno_file}")
print(f"测试集标注文件已保存至: {test_anno_file}")
print(f"最大标注框数量: {max_length}")
print(f"类别数量: {num_classes}")