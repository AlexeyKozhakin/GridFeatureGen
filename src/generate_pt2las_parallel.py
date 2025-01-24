import json
import os
import sys
import laspy
import torch
import numpy as np
from scipy.spatial import cKDTree
from map_label2rgb import class_colors_stpls3d 
from multiprocessing import Pool, cpu_count


# ----------------------------------------
# Чтение конфигурационного файла config.json
# ----------------------------------------

def read_config(config_path):
    """
    Читает конфигурационный файл config.json и извлекает параметры из раздела generate_pt2las.
    
    :param config_path: Путь к config.json
    :return: Словарь с параметрами из раздела generate_pt2las
    """
    if not os.path.exists(config_path):
        print(f"Ошибка: Файл конфигурации {config_path} не найден.")
        sys.exit(1)

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Ошибка: Некорректный формат JSON в {config_path}: {e}")
        sys.exit(1)

    # Проверяем, есть ли раздел generate_pt2las в конфигурации
    if 'generate_pt2las' not in config:
        print("Ошибка: Раздел 'generate_pt2las' отсутствует в config.json.")
        sys.exit(1)

    params = config['generate_pt2las']

    # Проверяем наличие всех необходимых ключей
    required_keys = [
        'input_las_dir', 
        'input_pt_dir', 
        'is_gen_rgb', 
        'is_gen_label', 
        'output_las_label_dir', 
        'output_las_rgb_dir', 
        'map_label2rgb'
    ]

    missing_keys = [key for key in required_keys if key not in params]
    if missing_keys:
        print(f"Ошибка: В разделе 'generate_pt2las' отсутствуют ключи: {', '.join(missing_keys)}.")
        sys.exit(1)

    return params

# ----------------------------------------
# Пример использования
# ----------------------------------------

if __name__ == "__main__":
    config_path = "config.json"  # Укажите путь к вашему config.json
    config_params = read_config(config_path)

    # Вывод параметров для проверки
    print("Параметры из config.json:")
    for key, value in config_params.items():
        print(f"{key}: {value}")


# ----------------------------------------
# Проверка входных параметров is_gen_rgb и is_gen_label
# ----------------------------------------

def check_generation_params(is_gen_rgb, is_gen_label):
    """
    Проверяет параметры is_gen_rgb и is_gen_label. 
    Завершает выполнение, если обе переменные равны False.
    
    :param is_gen_rgb: Логическая переменная для генерации RGB в LAS файлах
    :param is_gen_label: Логическая переменная для генерации классов в LAS файлах
    """
    if not is_gen_rgb and not is_gen_label:
        print("Генерация LAS файлов с классами и/или RGB отключена.")
        print("Проверьте настройки is_gen_rgb и is_gen_label в config.json.")
        sys.exit(0)

# ----------------------------------------
# Проверка существования и соответствия LAS и PT файлов
# ----------------------------------------

def check_file_correspondence(input_las_dir, input_pt_dir):
    """
    Проверяет существование LAS и PT файлов и их соответствие по базовым именам.
    
    :param input_las_dir: Путь к директории с LAS файлами
    :param input_pt_dir: Путь к директории с PT файлами
    """
    # Считываем все файлы с расширениями .las и .pt
    las_files = [f for f in os.listdir(input_las_dir) if f.endswith('.las')]
    pt_files = [f for f in os.listdir(input_pt_dir) if f.endswith('.pt')]

    # Создаем множества базовых имен файлов (без расширений)
    las_basenames = {os.path.splitext(f)[0] for f in las_files}
    pt_basenames = {os.path.splitext(f)[0] for f in pt_files}

    # Ищем несоответствия
    missing_pt_files = las_basenames - pt_basenames
    missing_las_files = pt_basenames - las_basenames

    # Выводим предупреждения, если есть несоответствия
    if missing_pt_files:
        print("Следующие LAS файлы не имеют соответствующих PT файлов:")
        for name in missing_pt_files:
            print(f"  {name}.las")
    if missing_las_files:
        print("Следующие PT файлы не имеют соответствующих LAS файлов:")
        for name in missing_las_files:
            print(f"  {name}.pt")

    # Если есть несоответствия, завершить выполнение
    if missing_pt_files or missing_las_files:
        print("Устраните несоответствия перед продолжением.")
        sys.exit(1)


def load_las_to_numpy(file_path):
    """
    Обрабатывает один LAS файл и возвращает выборку точек и классов в виде тензоров.
    file_path: Путь к LAS файлу.
    num_points_lim: Количество точек для выборки.
    """
    try:
        las = laspy.read(file_path)

        # Извлечение координат и цветовых данных
        points = np.vstack((las.x, las.y, las.z, las.red, las.green, las.blue)).T
        return points
    except Exception as e:
        print(f"Ошибка при обработке файла {file_path}: {e}")
        return None


# ----------------------------------------
# Пример использования
# ----------------------------------------

def process_file(args):
    las_file, input_pt_dir, output_dir, class_colors_stpls3d = args

    # Извлечение базового имени файла
    las_basename = os.path.splitext(os.path.basename(las_file))[0]
    pt_file = os.path.join(input_pt_dir, las_basename + '.pt')

    # Загрузка данных
    points = load_las_to_numpy(las_file)
    pt = torch.load(pt_file).to('cpu').numpy()

    M = pt.shape[0]

    # Сетка (M x M): Генерация квадратной сетки по x и y
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    x_grid, y_grid = np.linspace(x_min, x_max, M), np.linspace(y_min, y_max, M)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Создание дерева для поиска ближайших соседей
    tree = cKDTree(grid_points)

    # Поиск ближайших соседей для каждой точки
    distances, indices = tree.query(points[:, :2])

    # Присвоение классов точкам
    classes = pt[:, :, 6].T.reshape(-1)[indices]

    # Преобразование классов в RGB
    rgb_colors = np.array([class_colors_stpls3d[c] for c in classes])

    # Добавление классов к точкам
    points_with_classes = np.hstack([points, classes[:, None]])

    # Замена RGB-значений в points[:, 3:6] на цвета
    points[:, 3:6] = rgb_colors

    # Чтение оригинального LAS файла
    las = laspy.read(las_file)

    # Создание нового LAS файла
    new_las = laspy.create(point_format=las.point_format, file_version=las.header.version)
    new_las.x = points[:, 0]
    new_las.y = points[:, 1]
    new_las.z = points[:, 2]
    new_las.red = points[:, 3]
    new_las.green = points[:, 4]
    new_las.blue = points[:, 5]

    # Сохранение нового файла
    output_las_path = os.path.join(output_dir, las_basename + '.las')
    new_las.write(output_las_path)
    print(f"Processed {las_file} -> {output_las_path}")

def main(config_params, class_colors_stpls3d, num_processes=None):
    input_las_dir = config_params["input_las_dir"]
    input_pt_dir = config_params["input_pt_dir"]
    output_dir = config_params["output_las_rgb_dir"]

    os.makedirs(output_dir, exist_ok=True)

    las_files = [os.path.join(input_las_dir, f) for f in os.listdir(input_las_dir) if f.endswith('.las')]

    # Определение количества процессов
    if num_processes is None:
        num_processes = cpu_count()

    # Создание пула процессов
    with Pool(processes=num_processes) as pool:
        pool.map(process_file, [(las_file, input_pt_dir, output_dir, class_colors_stpls3d) for las_file in las_files])

    print("Processing complete.")

# Использование функции main
if __name__ == "__main__":
    config_path = "config.json"  # Укажите путь к вашему config.json
    config_params = read_config(config_path)

    # Проверка настроек генерации
    check_generation_params(config_params["is_gen_rgb"], config_params["is_gen_label"])

    # Проверка файлов на соответствие
    check_file_correspondence(config_params["input_las_dir"], config_params["input_pt_dir"])


    main(config_params, class_colors_stpls3d, num_processes=4)
