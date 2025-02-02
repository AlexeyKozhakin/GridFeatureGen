import os
import json
import torch
from torch.utils.data import DataLoader
from src.utils import LASDataset, GridTransformator, FeatureExtractor
from src.utils import feature_1_mean_z, feature_2_mode_color, feature_3_std_z, feature_4_mean_distance, feature_5_class


# Dictionary mapping feature names to their respective functions
dict_feature_fun = {
    "mean_z_minus_z_min": feature_1_mean_z,
    "rgb": feature_2_mode_color,
    "std_z": feature_3_std_z,
    "mean_distance": feature_4_mean_distance,
    "class": feature_5_class,
}

# Read the configuration file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Directory containing .las files
file_dir = config['features']['input_directory']
batch_size = config['features']['batch_size']
num_points_lim = config['features']['num_points']
M = config['features']['grid_size']
K = config['features']['num_neighbors']
name_features = config['features']['features_to_generate']
output_dir = config['features']['features_output_directory']

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get list of .las files
file_paths = [os.path.join(file_dir, f) for f in os.listdir(file_dir) if f.endswith('.las')]

# Initialize Dataset and DataLoader
base_dataset = LASDataset(file_paths, num_points_lim)
data_loader = DataLoader(base_dataset, batch_size=batch_size, shuffle=False)

# Create instances of GridTransformator and FeatureExtractor
grid_transformator = GridTransformator()
extractor = FeatureExtractor()
iter = 0
# Process each batch in the DataLoader
for batch_file_name, batch in data_loader:
    iter+=1
    print(f"Processing batch {iter}/{len(data_loader)}")
    print(batch.shape)  # Size (B, N, 7) or other depending on transform_fn

    # Compute KNN data and grid
    batch_knn_data, batch_grid = grid_transformator.compute(batch, M, K)
    print(batch_knn_data.shape)

    # Add features to the extractor
    for name_feature in name_features:
        extractor.add_feature(name_feature, dict_feature_fun[name_feature])

    # Compute features
    features = extractor.compute(batch_knn_data, batch_grid, batch)

    for name_feature in name_features:
        print(name_feature, features[name_feature].shape)

    # Concatenate features into a final tensor
    final_features = torch.cat(
        [features[name_feature] for name_feature in name_features],
        dim=-1
    )

    for name_feature in name_features:
        print(name_feature, features[name_feature].shape)

    print('final_features', final_features.shape)

    # Save the final features tensor to a file
    for file_name, one_features_tensor in zip(batch_file_name, final_features):
        output_file_path = os.path.join(output_dir, f'{file_name}.pt')
        torch.save(one_features_tensor, output_file_path)
        print(f'Saved final features for batch {file_name} to {output_file_path}')

# Создание мета-данных
meta_data = {}
index_start = 0

for name_feature in name_features:
    # Получаем количество индексов для текущего признака
    num_indices = features[name_feature].shape[3]  # Предполагается, что размерность 0 - это количество индексов
    for index in range(index_start, index_start+num_indices):
        if name_feature in meta_data:
            meta_data[name_feature].append(index)
        else:
            meta_data[name_feature] = [index]
    index_start += num_indices

# Запись данных в JSON файл
with open('meta_data.json', 'w', encoding='utf-8') as file:
    json.dump(meta_data, file, ensure_ascii=False, indent=4)

print("Метаданные записаны в файл meta_data.json")
