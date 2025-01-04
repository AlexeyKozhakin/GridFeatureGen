import torch
import laspy
import numpy as np
from collections import defaultdict
from pathlib import Path


class LASDatasetNumPoints(torch.utils.data.Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        las = laspy.read(file_path)
        # Извлечение координат и цветовых данных
        points = las.x

        num_points = points.shape[0]  # (N, 7)

        return num_points, file_path


# Основной класс для модульного добавления признаков
class GridTransformator:
    def __init__(self):
        self.features = defaultdict(list)

    def compute(self, data, M, K):
        """Вычисляет признаки для всех точек сетки."""
        result, grid = get_knn_data(data, M, K)
        return result, grid

# Функции для вычисления признаков
def get_mesh_grid(data, M):
    B, N, _ = data.shape

    # Extract x and y coordinates
    x_coords = data[:, :, 0]  # Shape (B, N)
    y_coords = data[:, :, 1]  # Shape (B, N)

    # Find boundary points for each batch
    x_min = x_coords.min(dim=1)[0]  # Shape (B,)
    x_max = x_coords.max(dim=1)[0]  # Shape (B,)
    y_min = y_coords.min(dim=1)[0]  # Shape (B,)
    y_max = y_coords.max(dim=1)[0]  # Shape (B,)

    # Create base grids
    base_x = torch.linspace(0, 1, M, device=data.device)  # Shape (M,)
    base_y = torch.linspace(0, 1, M, device=data.device)  # Shape (M,)

    # Scale and shift base grids for each batch
    x_grid = x_min[:, None] + (x_max[:, None] - x_min[:, None]) * base_x[None, :]  # Shape (B, M)
    y_grid = y_min[:, None] + (y_max[:, None] - y_min[:, None]) * base_y[None, :]  # Shape (B, M)

    # Create mesh grids using matrix operations
    x_mesh = x_grid.unsqueeze(2).expand(B, M, M)  # Shape (B, M, M)
    y_mesh = y_grid.unsqueeze(1).expand(B, M, M)  # Shape (B, M, M)

    # Stack the mesh grids along the last dimension
    grid = torch.stack((x_mesh, y_mesh), dim=-1)  # Shape (B, M, M, 2)

    return grid



def get_knn_data(data, M, K):
    """
    Поиск K ближайших соседей для каждого батча на равномерной сетке.
    Args:
        data (torch.Tensor): Входной тензор (B, N, 7).
        M (int): Размер сетки (MxM).
        K (int): Количество ближайших соседей.
    Returns:
        torch.Tensor: Тензор ближайших соседей (B, M, M, 7, K).
    """
    B, N, _ = data.shape
    grid = get_mesh_grid(data, M)  # (B, M, M, 2)

    # Расчет расстояний от точек сетки до каждой точки облака по координатам x, y
    grid_coords = grid.reshape(B, -1, 2).float()  # (B, M*M, 2)
    data_coords = data[:, :, :2].float()  # (B, N, 2)

    # Вычисление квадратов расстояний
    dists = torch.cdist(grid_coords, data_coords)  # (B, M*M, N)

    # Находим индексы K ближайших соседей
    _, knn_indices = torch.topk(dists, K, dim=2, largest=False)  # (B, M*M, K)

    # Извлечение данных ближайших соседей
    # Убедимся, что индексы соответствуют батчам
    batch_indices = torch.arange(B).view(B, 1, 1).expand(-1, M * M, K)
    knn_data = data[batch_indices, knn_indices]  # (B, M*M, K, 7)

    # Преобразование в итоговый тензор (B, M, M, 7, K)
    knn_data = knn_data.reshape(B, M, M, K, 7).permute(0, 1, 2, 4, 3)  # (B, M, M, 7, K)
    return knn_data, grid

def random_point_sampling(points, num_samples):
    """
    Выполняет случайную выборку точек.
    points: Tensor с размерностью (N, D), где N - количество точек, D - размерность данных.
    num_samples: Количество точек для выборки.
    """
    num_points = points.shape[0]
    indices = torch.randperm(num_points)[:num_samples]
    return points[indices], indices

def load_las_to_tensor(file_path, num_points_lim=4096):
    """
    Обрабатывает один LAS файл и возвращает выборку точек и классов в виде тензоров.
    file_path: Путь к LAS файлу.
    num_points_lim: Количество точек для выборки.
    """
    try:
        las = laspy.read(file_path)

        # Извлечение координат и цветовых данных
        points = torch.tensor(np.vstack((las.x, las.y, las.z, las.red, las.green, las.blue)).T,
            dtype=torch.float32)
        # Тензор с размерностью (N, 6)

        # Извлечение классов
        classes = torch.tensor(las.classification, dtype=torch.int64)  # Тензор с классами (N,)

        # Проверка количества точек
        num_points = points.shape[0]
        if num_points > num_points_lim:
            # Случайная выборка точек
            sampled_points, sampled_indices = random_point_sampling(points, num_points_lim)

            # Получаем классы для отобранных точек
            sampled_classes = classes[sampled_indices]

            # Объединяем координаты и классы
            return torch.cat((sampled_points, sampled_classes.unsqueeze(-1)), dim=1)  # (num_points_lim, 7)
        else:
            print(f"Количество точек в файле меньше лимита {num_points_lim}, пропуск файла.")
            return None
    except Exception as e:
        print(f"Ошибка при обработке файла {file_path}: {e}")
        return None


class LASDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, num_points_lim=4096):
        self.file_paths = file_paths
        self.num_points_lim = num_points_lim

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        file_name = Path(file_path).stem
        num_points_lim = self.num_points_lim
        points = load_las_to_tensor(file_path, num_points_lim)  # (N, 7)
        return file_name, points

# Основной класс для модульного добавления признаков
class FeatureExtractor:
    def __init__(self):
        self.features = defaultdict(list)

    def add_feature(self, name, func):
        """Добавляет вычисление признака."""
        self.features[name].append(func)

    def compute(self, knn_data, grid, data):
        """Вычисляет признаки для всех точек сетки."""
        results = {}
        for name, funcs in self.features.items():
            for func in funcs:
                results[name] = func(knn_data, grid, data)
        return results

# Функции для вычисления признаков
def feature_1_mean_z(knn_data, _, data):
    """
    1. Найти минимальное значение z по всему облаку (из (1, N, 7)).
    2. Отнять это минимальное значение из структурированного тензора (1, M, M, 7, K) - третей координаты.
    3. Найти среднее значение по соседям K mean(z - z_min).
    """
    z_min = data[:, :, 2].min()  # Минимальное z по всему облаку
    z_values = knn_data[:, :, :, 2, :]  # Координаты z из (1, M, M, 7, K)
    mean_z = (z_values - z_min).mean(dim=-1)  # Среднее по соседям K
    return mean_z.unsqueeze(-1)  # Добавляем измерение для согласованности

def feature_2_mode_color(knn_data, _, __):
    """
    Найти моду цвета среди соседей K.
    Это координаты цвета с индексами 3, 4, 5 в (1, M, M, 7, K).
    """
    colors = knn_data[:, :, :, 3:6, :]  # Достаем цвета (r, g, b)
    modes = torch.mode(colors, dim=-1).values  # Находим моду по соседям K
    return modes  # Размерность (1, M, M, 3)

def feature_3_std_z(knn_data, _, __):
    """
    Найти стандартное отклонение z (третья координата) для соседей K.
    """
    z_values = knn_data[:, :, :, 2, :]  # Координаты z из (1, M, M, 7, K)
    std_z = z_values.std(dim=-1)  # Стандартное отклонение по соседям K
    return std_z.unsqueeze(-1)  # Добавляем измерение для согласованности

def feature_4_mean_distance(knn_data, grid, _):
    """
    Найти среднее расстояние от узла сетки (grid) до соседей (knn_data).
    """
    # Координаты узлов сетки (1, M, M, 2) -> (1, M, M, 2, 1) для совместимости с knn_data
    grid_coords = grid.unsqueeze(-1)  # Размерность становится (1, M, M, 2, 1)

    # Координаты соседей (1, M, M, 2, K)
    neighbor_coords = knn_data[:, :, :, :2, :]  # Берем x и y

    # Вычисляем расстояния (евклидово расстояние)
    distances = torch.norm(grid_coords - neighbor_coords, dim=3)  # (1, M, M, K)

    # Находим среднее расстояние
    mean_distance = distances.mean(dim=-1)  # (1, M, M)
    return mean_distance.unsqueeze(-1)  # Добавляем измерение для согласованности

class TensorDatasetMaxMeanDist(torch.utils.data.Dataset):
    def __init__(self, file_paths, column_dist):
        self.file_paths = file_paths
        self.column_dist = column_dist
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        column_dist = self.column_dist
        file_path = self.file_paths[idx]
        tensor = torch.load(file_path)
        # Извлечение координат и цветовых данных
        max_mean_dist = int(torch.max(tensor[:,:,column_dist]))
        return max_mean_dist, file_path