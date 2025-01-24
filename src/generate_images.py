import os
import torch
from PIL import Image
import numpy as np
import multiprocessing
import json


# Function to convert tensor to image and save it
def tensor_to_image(tensor_path, output_dir, channels):

    # Load the tensor from the .pt file
    tensor = torch.load(tensor_path)


    # Check if the tensor has at least one channel
    if tensor.ndim != 3 or tensor.shape[2] < max(channels):
        print(f"Tensor in {tensor_path} does not have enough channels.")
        return

    # Select the desired channels (0-indexed)
    selected_tensor = tensor[..., channels]/torch.max(tensor[..., channels])*255
    #selected_tensor = tensor[..., channels]
    # Convert the tensor to a numpy array and scale to [0, 255]
    image_array = (selected_tensor.numpy()).astype(np.uint8)

    # Create an image from the numpy array
    image = Image.fromarray(image_array)

    # Create a filename based on the original tensor filename
    filename = os.path.basename(tensor_path).replace('.pt', '.png')
    output_file_path = os.path.join(output_dir, filename)

    # Save the image
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
    file_filter = config['filtering']['filter_info']

    # Задайте пороговое значение
    n_workers = config['visualization']['n_workers']  # Установите желаемый порог
    input_directory = config['visualization']['input_directory']  # Change this to your input directory path
    output_directory = config['visualization']['output_directory']  # Change this to your output directory path
    name_channels = config['visualization']['channels']

    meta_file = config['visualization']['meta_file']

    # Открытие файла JSON
#    with open(meta_file, 'r') as file:
#        data = json.load(file)  # Загрузка содержимого в переменную data

    # Получение значения mean_distance
    desired_channels = []
    #data = {'rgb':[1,2,3]}
    data = {"mean_z_minus_z_min":0,"std_z":4, "mean_distance":5}

    desired_channels.extend([0,4,5])
    #for name_channel in name_channels:
    #    desired_channels.extend(data[name_channel])

    # Specify which channels you want to keep (0-indexed)
    # print('desired_channels', desired_channels)
    #desired_channels = [0,0,0]
    process_tensors(input_directory, output_directory, desired_channels, n_workers)
