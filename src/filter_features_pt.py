import os
import pandas as pd
import json
import shutil
from concurrent.futures import ThreadPoolExecutor

# Чтение конфигурационного файла
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Путь к файлу с информацией о файлах
file_filter = config['feature_filtering']['data_csv']
# Задайте пороговое значение
threshold = config['feature_filtering']['max_distance_threshold']
dir_above = config['feature_filtering']['above_threshold_directory']
dir_below = config['feature_filtering']['below_threshold_directory']
n_threads = config['feature_filtering']['threads']

# Чтение файла file_info.csv
file_info_df = pd.read_csv(file_filter)

# Убедитесь, что каталоги существуют, если нет, создайте их
os.makedirs(dir_above, exist_ok=True)
os.makedirs(dir_below, exist_ok=True)

# Функция для копирования файла
def copy_file(file_path, target_dir):
    try:
        shutil.copy(file_path, target_dir)
        print(f"Файл {file_path} скопирован в {target_dir}")
    except Exception as e:
        print(f"Ошибка при копировании файла {file_path}: {e}")

# Использование ThreadPoolExecutor для управления потоками
with ThreadPoolExecutor(max_workers=n_threads) as executor:  # Задайте желаемое количество потоков
    futures = []
    for index, row in file_info_df.iterrows():
        file_path = row['paths']
        max_mean_dist = row['max_mean_dists']

        # Определение целевого каталога в зависимости от max_mean_dists
        if max_mean_dist > threshold:
            target_dir = dir_above
        else:
            target_dir = dir_below

        # Отправка задачи на выполнение в пул потоков
        futures.append(executor.submit(copy_file, file_path, target_dir))

# Ожидание завершения всех задач
for future in futures:
    future.result()

print("Все файлы скопированы.")
