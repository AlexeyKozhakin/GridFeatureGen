import os
import json
import torch
from tqdm import tqdm

# Чтение конфигурационного файла
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Пример использования
input_folder = config['check_tensor']['tensor_pathfile']  # Укажите путь к вашей папке с тензорами


# Функция для загрузки тензоров и вывода информации о размерности
def load_and_display_tensor_info(file_path):
        # Загружаем тензор
        tensor = torch.load(file_path)
        # Выводим информацию о размерности
        print(f"Файл: {file_path}, Размерность: {tensor.shape}")


# Вызов функции с указанным путем
load_and_display_tensor_info(input_folder)


