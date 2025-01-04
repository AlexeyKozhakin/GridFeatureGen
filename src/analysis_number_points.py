import os
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from utils import LASDatasetNumPoints

# Чтение конфигурационного файла
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Каталог с файлами .las
file_dir = config['analysis']['input_directory']
batch_size = config['analysis']['batch_size']

# Получение списка файлов
file_paths = [os.path.join(file_dir, f) for f in os.listdir(file_dir) if f.endswith('.las')]
total_files = len(file_paths)
print(f"Найдено файлов: {total_files}")

# Инициализация датасета
points_dataset = LASDatasetNumPoints(file_paths)

# DataLoader для обработки батчей
data_loader = DataLoader(points_dataset, batch_size=batch_size, shuffle=False)

# Список для хранения тензоров с количеством точек и информации о файлах
all_point_counts = []
file_info = []  # Список для хранения информации о файлах

files_processed = 0

# Использование DataLoader
for num_points, paths in data_loader:
    # Добавляем количество точек в список
    all_point_counts.append(num_points)  # Добавляем текущий батч в список

    # Сохраняем информацию о файлах и количестве точек
    for path, count in zip(paths, num_points):
        file_info.append({'Путь к файлу': path, 'Количество точек': count.item()})

    # Вывод текущего состояния с учетом размера батча
    files_processed += len(paths)  # Количество обработанных файлов в текущем батче
    files_remaining = total_files - files_processed  # Осталось файлов

    if files_remaining < 0:
        print(f"Обработано файлов: {total_files}/{total_files}, Осталось: {0}")
    else:
        print(f"Обработано файлов: {files_processed}/{total_files}, Осталось: {files_remaining}")

# Объединение всех тензоров в один
all_point_counts_tensor = torch.cat(all_point_counts)
print('Количество данных в тензоре:', all_point_counts_tensor.shape)

# Вычисление среднего и стандартного отклонения по всем точкам
mean_points = all_point_counts_tensor.float().mean().item()  # Приводим к float для вычислений
std_points = all_point_counts_tensor.float().std().item()  # Используем std вместо var
max_points = all_point_counts_tensor.float().max().item()  # Максимальное количество точек
min_points = all_point_counts_tensor.float().min().item()  # Минимальное количество точек

# Вычисление среднего минус стандартное отклонение и округление до целого числа
mean_minus_std = round(mean_points - std_points)

# Вывод результатов на экран
print(f"Макс количество точек: {max_points}")
print(f"Мин количество точек: {min_points}")
print(f"Среднее количество точек: {round(mean_points)}")
print(f"Стандартное отклонение количества точек: {round(std_points)}")
print(f"Среднее минус стандартное отклонение (округленное): {mean_minus_std}")

# Запись данных в CSV файл с использованием pandas для статистики
results_df = pd.DataFrame({
    'Среднее количество точек': [round(mean_points)],
    'Стандартное отклонение': [round(std_points)],
    'Среднее минус стандартное отклонение': [mean_minus_std]
})

output_file_stats = 'sta_points.csv'
results_df.to_csv(output_file_stats, index=False)

print(f"Результаты статистики записаны в файл: {output_file_stats}")

# Запись информации о файлах в отдельный CSV файл
file_info_df = pd.DataFrame(file_info)
output_file_info = 'file_info.csv'
file_info_df.to_csv(output_file_info, index=False)

print(f"Информация о файлах записана в файл: {output_file_info}")
