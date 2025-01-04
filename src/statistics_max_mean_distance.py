import os
import json
import torch
from torch.utils.data import DataLoader
from utils import TensorDatasetMaxMeanDist
import pandas as pd


# Read the configuration file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Directory containing .las files
file_dir = config['statistics']['input_directory']
batch_size = config['statistics']['batch_size']
output_file = config['statistics']['output_csv']
meta_file = config['statistics']['meta_file']
stat_file = config['statistics']['stat_dist']

# Открытие файла JSON
with open(meta_file, 'r') as file:
    data = json.load(file)  # Загрузка содержимого в переменную data

# Получение значения mean_distance
mean_distance_column = data['mean_distance'][0]
print(mean_distance_column)  # Вывод: [5]

# Get list of .pt files
file_paths = [os.path.join(file_dir, f) for f in os.listdir(file_dir) if f.endswith('.pt')]
base_dataset = TensorDatasetMaxMeanDist(file_paths, mean_distance_column)
data_loader = DataLoader(base_dataset, batch_size=batch_size, shuffle=False)

# Список для хранения тензоров с количеством точек и информации о файлах
file_info = {'paths': [], 'max_mean_dists': []}
files_processed = 0
total_files = len(file_paths)
# Использование DataLoader
for batch_max_mean_dist, batch_paths in data_loader:
    # Сохраняем информацию о файлах и количестве точек
    for path, dist in zip(batch_paths, batch_max_mean_dist):
        file_info['paths'].append(path)
        file_info['max_mean_dists'].append(dist.item())

    # Вывод текущего состояния с учетом размера батча
    files_processed += len(batch_max_mean_dist)  # Количество обработанных файлов в текущем батче
    files_remaining = total_files - files_processed  # Осталось файлов

    if files_remaining < 0:
        print(f"Обработано файлов: {total_files}/{total_files}, Осталось: {0}")
    else:
        print(f"Обработано файлов: {files_processed}/{total_files}, Осталось: {files_remaining}")

results_df = pd.DataFrame(
file_info)
results_df.to_csv(output_file, index=False)

# Вычисление статистик
mean_value = int(results_df['max_mean_dists'].mean())
min_value = int(results_df['max_mean_dists'].min())
max_value = int(results_df['max_mean_dists'].max())
std_value = int(results_df['max_mean_dists'].std())

# Вывод значений на экран
print('mean_max_mean_dists', mean_value)
print('min_max_mean_dists', min_value)
print('max_max_mean_dists', max_value)
print('std_max_mean_dists', std_value)

# Создание словаря для записи в JSON
stats = {
    'mean_max_mean_dists': mean_value,
    'min_max_mean_dists': min_value,
    'max_max_mean_dists': max_value,
    'std_max_mean_dists': std_value
}

# Запись статистик в JSON файл
with open(stat_file, 'w') as json_file:
    json.dump(stats, json_file, indent=4)

print(f"Статистики записаны в файл {stat_file}")