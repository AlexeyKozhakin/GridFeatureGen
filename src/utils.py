import torch
import laspy

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
        return num_points