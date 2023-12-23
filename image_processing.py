from PIL import Image, ImageOps
import os
import numpy as np

def add_gaussian_noise(image, mean=0, std=25):
    """
    添加高斯噪声到图像中
    """
    noise = np.random.normal(mean, std, image.size)
    noise = noise.reshape(image.size[1], image.size[0]).astype('uint8')
    noisy_image = Image.fromarray(np.array(image.convert('L')) + noise)
    return noisy_image

def preprocess_images(input_path, output_path, target_size=(224, 224), num_generated=10):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for file in os.listdir(input_path):
        if file.endswith(".png"):
            file_path = os.path.join(input_path, file)
            image = Image.open(file_path)
            
            # 调整图像大小
            image = image.resize(target_size, Image.LANCZOS)
            
            # 添加噪声并进行数据增强
            for i in range(num_generated):
                transformed_image = image.copy()
                
                # 随机添加高斯噪声
                noisy_image = add_gaussian_noise(transformed_image)
                
                # 生成新文件名
                new_filename = f"{os.path.splitext(file)[0]}_enhanced_noise_{i}.png"
                new_file_path = os.path.join(output_path, new_filename)
                
                # 保存增强后的带噪声的图像
                noisy_image.save(new_file_path)

# 示例用法
input_directory = 'origin_images'
output_directory = 'new_images'
preprocess_images(input_directory, output_directory)
