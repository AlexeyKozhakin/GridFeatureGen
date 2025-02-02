import os
import json
import torch
from tqdm import tqdm


def parse_coordinates(filename):
    """
    Извлекает координаты из имени файла, например: '453000_3974000.pt'.
    """
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    # Разделяем по `_` и берем последние два числа
    parts = name.split('_')
    y, x = map(int, parts[-2:])
#    y, x = map(int, name.split('_'))
    return x, y


def stitch_tensors(input_folder, output_file):
    """
    Склеивает тензоры из папки на основе их нумерации.
    """
    files = [f for f in os.listdir(input_folder) if f.endswith('.pt')]
    coordinates = {}

    for file in files:
        x, y = parse_coordinates(file)
        coordinates[(x, y)] = file

    if not coordinates:
        raise ValueError("В папке нет тензоров для склейки")

    x_coords = sorted({x for x, y in coordinates.keys()})
    y_coords = sorted({y for x, y in coordinates.keys()})

    if len(x_coords) < 1 or len(y_coords) < 1:
        raise ValueError("Недостаточно координат для определения сетки")

    step_x = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 0
    step_y = y_coords[1] - y_coords[0] if len(y_coords) > 1 else 0

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # Определяем количество тайлов
    num_x = ((max_x - min_x) // step_x) + 1 if step_x != 0 else 1
    num_y = ((max_y - min_y) // step_y) + 1 if step_y != 0 else 1

    # Загружаем первый тензор для определения формы
    first_tensor = torch.load(
        os.path.join(input_folder, coordinates[(min_x, min_y)]),
        map_location=torch.device('cpu')
    )

    tile_shape = first_tensor.shape

    final_height = num_y * tile_shape[0]
    final_width = num_x * tile_shape[1]

    final_tensor = torch.zeros((final_height, final_width, tile_shape[2]))

    default_tensor = torch.zeros(tile_shape)

    for y in tqdm(range(num_y), desc="Склеивание строк"):
        for x in range(num_x):
            coord_x = min_x + x * step_x
            coord_y = min_y + y * step_y

            start_x = x * tile_shape[1]
            start_y = y * tile_shape[0]

            if (coord_x, coord_y) in coordinates:
                tile = torch.load(
                    os.path.join(input_folder, coordinates[(coord_x, coord_y)]),
                    map_location=torch.device('cpu')
                )
            else:
                tile = default_tensor

            final_tensor[start_y:start_y + tile_shape[0],
            start_x:start_x + tile_shape[1],
            :] = tile

    torch.save(final_tensor, output_file)
    print(f"Склеенный тензор сохранен как '{output_file}'")


# Чтение конфигурационного файла
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

input_folder = config['join_tensors']['input_splited_tensors_path']
output_file = config['join_tensors']['output_joined_tensor_pathfile']
stitch_tensors(input_folder, output_file)
