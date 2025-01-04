# GridFeatureGen

Последовательность запуска скриптов
1) Анализируем количество точек

aalysis_number_points.py
2) Определяем нижний порог точек и удаляем не нужные файлы

filter_dataset.py
3) Затем генерируем датасет с фичами

generate_features.py
4) Потом анализуем расстояния

statistics_max_mean_distance.py
5) Разделяем на сильно испорченные

filter_geatures_pt.py

6) Строим изображения

generate_images.py