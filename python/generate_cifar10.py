import os
from torchvision import datasets, transforms
from PIL import Image
from sklearn.model_selection import train_test_split

os.makedirs('E:\data\cifar10', exist_ok=True)
# Step 1: 加载 CIFAR-10 数据集
dataset = datasets.CIFAR10(root='E:\data\cifar10', train=True, download=True)

# 类别列表
classes = dataset.classes
num_classes = len(classes)  # 获取类别数量
# Step 2: 创建 images 文件夹
os.makedirs('E:\data\cifar10\images', exist_ok=True)
for c in classes:
    os.makedirs(f'E:\data\cifar10\images/{c}', exist_ok=True)

# 图像转换函数
transform = transforms.ToPILImage()

# 存储所有样本的路径和标签
all_samples = []

# Step 3: 遍历数据集，保存图像，并记录路径和标签
for i, (img, label) in enumerate(dataset):
    class_name = classes[label]
    img_filename = f'{i}.png'
    img_save_path = f'E:\data\cifar10/images/{class_name}/{img_filename}'
    img.save(img_save_path)
    all_samples.append((img_save_path, label))

# Step 4: 划分训练集和验证集（比如 9:1）
train_samples, val_samples = train_test_split(all_samples, test_size=0.1, random_state=42)
# Step 5: 写入 anno 文件，并在第一行写入类别数量
with open('E:\data\cifar10/train_annotations.txt', 'w') as f_train, \
        open('E:\data\cifar10/test_annotations.txt', 'w') as f_val:

    # 写入类别数量作为第一行
    f_train.write(f'{num_classes}\n')
    f_val.write(f'{num_classes}\n')

    # 写入训练集
    for path, label in train_samples:
        f_train.write(f'{path} {label}\n')

    # 写入验证集
    for path, label in val_samples:
        f_val.write(f'{path} {label}\n')

print("✅ 训练集和验证集注解文件已生成，并在第一行列出类别数量：")
print(" - E:\data\cifar10/train_annotations.txt")
print(" - E:\data\cifar10/test_annotations.txt")