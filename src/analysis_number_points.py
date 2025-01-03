import os
from torch.utils.data import DataLoader
from .utils import LASDatasetNumPoints

# Загрузка пути из конфигурации
import json

# Чтение конфигурационного файла
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Каталог с файлами .las
file_dir = config['general']['input_directory']

# Получение списка файлов
file_paths = [os.path.join(file_dir, f) for f in os.listdir(file_dir) if f.endswith('.las')]
print(file_paths)
# Инициализация датасета
points_dataset = LASDatasetNumPoints(file_paths)

# DataLoader для обработки батчей
data_loader = DataLoader(points_dataset, batch_size=2, shuffle=False)

# Использование DataLoader
for batch in data_loader:
    print(batch.shape)  # Размер (B, N, 7) или другой в зависимости от transform_fn
