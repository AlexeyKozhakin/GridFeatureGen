import os
import pandas as pd
import json

# Чтение конфигурационного файла
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Путь к файлу с информацией о файлах
file_filter = config['filtering']['filter_info']
# Задайте пороговое значение
threshold = config['filtering']['threshold_points']  # Установите желаемый порог

# Чтение файла file_info.csv
file_info_df = pd.read_csv(file_filter)

# Выводим исходные данные для проверки
print("Исходные данные:")
print(file_info_df)

# Удаление файлов с количеством точек меньше порога
files_to_delete = file_info_df[file_info_df['Количество точек'] < threshold]['Путь к файлу']

# Удаляем файлы с диска
for file_path in files_to_delete:
    try:
        os.remove(file_path)
        print(f"Удален файл: {file_path}")
    except Exception as e:
        print(f"Не удалось удалить файл {file_path}: {e}")

# Выводим оставшиеся данные для проверки
filtered_df = file_info_df[file_info_df['Количество точек'] >= threshold]
print("\nОставшиеся файлы:")
print(filtered_df)

# Сохранение оставшихся данных в новый CSV файл
filtered_file = 'filtered_file_info.csv'
filtered_df.to_csv(filtered_file, index=False)

print(f"\nФайлы с количеством точек ниже {threshold} удалены. Результаты сохранены в файл: {filtered_file}")
