import torch
from torchvision import transforms
from PIL import Image
import os
from train import ResNetModel

# 加载训练好的模型
model = ResNetModel()  # 请确保ResNetModel类定义与之前训练模型时相同
model.load_state_dict(torch.load('model/mnist_resnet_model.pth'))
model.eval()

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 读取文件夹中的所有PNG图像并进行预测
def predict_images_in_folder(folder_path):
    correct_count = 0
    total_count = 0

    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            file_path = os.path.join(folder_path, filename)
            image = Image.open(file_path).convert("L")  # 转为灰度图像
            image = transforms.functional.resize(image, (224, 224))  # 调整大小
            image = transform(image).unsqueeze(0)  # 数据预处理
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            
            # 获取预测结果和文件名的第一个字符（假设文件名的第一个字符是标签）
            prediction = int(filename[0])
            actual_label = predicted.item()

            total_count += 1
            if prediction == actual_label:
                correct_count += 1

            print(f"File: {filename}, Predicted: {actual_label}, Actual: {prediction}")

    accuracy = correct_count / total_count * 100 if total_count > 0 else 0
    print(f"\nTotal images: {total_count}, Correct predictions: {correct_count}, Accuracy: {accuracy:.2f}%")

# 示例用法：替换为你的文件夹路径
folder_path = 'new_images'
predict_images_in_folder(folder_path)
