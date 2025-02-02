import os
import torch
from PIL import Image
import numpy as np
import multiprocessing
import json


# Function to convert tensor to image and save it
def tensor_to_image(tensor_path, output_dir, channel):

    tensor = torch.load(tensor_path)
    selected_tensor = tensor[..., channel].squeeze(-1) 
    print('select',selected_tensor.shape)
    class_colors_stpls3d = {
    0: (0, 0, 0),  # Ground - Черный
    1: (128, 128, 128),  # Building - Зеленый
    2: (255, 255, 0),  # LowVegetation - Желтый
    3: (0, 0, 255),  # MediumVegetation - Синий
    4: (255, 0, 0),  # HighVegetation - Красный
    5: (0, 255, 255),  # Vehicle - Бирюзовый
    6: (255, 0, 255),  # Truck - Магента
    7: (255, 128, 0),  # Aircraft - Оранжевый
    8: (255, 20, 147),  # Bike - Deep Pink
}

    M = selected_tensor.shape[0]
    print('Image size:', M)
    
    # Создание RGB тензора с нулями
    rgb_tensor = torch.zeros(M, M, 3, dtype=torch.uint8)
    print(selected_tensor)
    # Заполнение RGB тензора на основе классов
    for class_id in class_colors_stpls3d.keys():
        color = torch.tensor(class_colors_stpls3d[class_id], dtype=torch.uint8) 
        # Используем булеву маску для выбора пикселей с текущим классом и присваиваем соответствующий цвет
        rgb_tensor[selected_tensor == class_id] = torch.tensor(color)
     
    # Преобразование tензора RGB в NumPy массив и создание изображения
    #print(rgb_tensor)
    image_array = rgb_tensor.numpy()
    image = Image.fromarray(image_array)

    # Формирование имени файла для сохранения изображения
    filename = os.path.basename(tensor_path).replace('.pt', '.png')
    output_file_path = os.path.join(output_dir, filename)
    
    # Сохранение изображения
    image.save(output_file_path)
    print(f"Saved image {output_file_path}")

# Main function to process all .pt files in a directory
def process_tensors(input_dir, output_dir, channels, n_workers):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all .pt files in the input directory
    tensor_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.pt')]

    # Use multiprocessing.Pool for parallel processing of files
    with multiprocessing.Pool(processes=n_workers) as pool:
        pool.starmap(tensor_to_image, [(file, output_dir, channels) for file in tensor_files])


if __name__ == "__main__":


    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    # Путь к файлу с информацией о файлах

    # Задайте пороговое значение
    n_workers = config['to_rgb']['n_workers']  # Установите желаемый порог
    input_directory = config['to_rgb']['input_directory']  # Change this to your input directory path
    output_directory = config['to_rgb']['output_directory']  # Change this to your output directory path
    name_channels = config['to_rgb']['channels']

    meta_file = config['to_rgb']['meta_file']

    # Открытие файла JSON
#    with open(meta_file, 'r') as file:
#        data = json.load(file)  # Загрузка содержимого в переменную data

    # Получение значения mean_distance
    desired_channels = [6]
    process_tensors(input_directory, output_directory, desired_channels, n_workers)
