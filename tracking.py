import os
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import json
import matplotlib.path as mplPath

# Указание исходной директории и имени видео
SOURCE_DIR = "data"
VIDEO_NAME = 'KRA-7-26-2023-08-10-evening'
FRAME_STEP = 3

# Путь до модели YOLO
MODEL_PATH = 'best.pt'

# Опция для отображения пользовательского интерфейса
SHOW_UI = True

# Чтение данных из JSON-файла
json_file = open(os.path.join(SOURCE_DIR, 'markup', 'jsons', VIDEO_NAME + '.json'))
json_data = json.load(json_file)
areas = json_data['areas']

# Инициализация модели YOLO
model = YOLO(MODEL_PATH)
model.to('cuda')

# Загрузка видео
cap = cv2.VideoCapture(os.path.join(SOURCE_DIR, 'video', VIDEO_NAME + '.mp4'))
fps = cap.get(cv2.CAP_PROP_FPS)

# Инициализация структур для хранения истории движения объектов
track_history = defaultdict(lambda: [])

storage = []
class_names = ['tractor', 'bike', 'bus', 'car', 'motorcycle', 'trolleybus', 'truck']

frame_number = 1

frame_size = ()

def convert_to_real_coordinates(coordinates, resolution):
    new_coordinates = []

    for coord in coordinates:
        x = int(coord[0] * resolution[0])
        y = int(coord[1] * resolution[1])
        new_coordinates.append([x, y])

    return new_coordinates

# Цикл обработки каждого кадра видео
while cap.isOpened():
    success, frame = cap.read()

    if frame_number == 1:
        frame_size = (frame.shape[1], frame.shape[0])



    frame_number += 1
    if frame_number % FRAME_STEP != 0:
        continue

    if success:
        # Применение модели YOLO для отслеживания объектов на кадре
        results = model.track(frame, persist=True)

        if results[0].boxes.id is not None:

            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.int().cpu().tolist()

            # Перебор распознанных объектов и их треков
            for box, track_id, class_id in zip(boxes, track_ids, classes):
                x, y, w, h = box

                track = track_history[track_id]
                point = (float(x), float(y) + float(h/2))
                track.append(point)

                storage.append((track_id, class_names[class_id], point))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    else:
        break


ids = list(set(item[0] for item in storage))

cars, vans, buses = [], [], []

for id in ids:
    desired_objects = [obj for obj in storage if obj[0] == id]

    class_count = {}  # создаем пустой словарь для подсчета частоты каждого класса

    for item in desired_objects:
        class_name = item[1]  # получаем класс из элемента
        if class_name in class_count:
            class_count[class_name] += 1  # увеличиваем счетчик, если класс уже есть в словаре
        else:
            class_count[class_name] = 1  # инициализируем счетчик для нового класса

    most_common_class = max(class_count, key=class_count.get)  # выбираем класс с наибольшей частотой

    class_name = most_common_class
    points_array = [item[2] for item in desired_objects]

    max_frames = 0

    for area in areas:
        real_coordinates_areas = convert_to_real_coordinates(area, frame_size)
        pts = np.array(real_coordinates_areas, np.int32)

        bbPath = mplPath.Path(pts)

        start_point, end_point = None, None
        track_frames_count = 0

        for point in points_array:
            if bbPath.contains_point(point):
                track_frames_count += 1
                if start_point is None:
                    start_point = point
            else:
                if start_point is not None:
                    end_point = point

        if track_frames_count > max_frames:
            max_frames = track_frames_count

    speed = 0

    if max_frames > 0:
        speed = 20 / ((max_frames * FRAME_STEP) / fps) * 3.6

        if class_name == 'car':
            cars.append(speed)

        if class_name == 'bus':
            buses.append(speed)

        if class_name == 'truck':
            vans.append(speed)


result = (VIDEO_NAME + ';' + str(len(cars)) + ';' + str(round(np.mean(cars), 2))
          + ';' + str(len(vans)) + ';' + str(round(np.mean(vans), 2))
          + ';' + str(len(buses)) + ';' + str(round(np.mean(buses), 2)))
print(result)

file_name = 'results.txt'
f = open(file_name, 'a+')  # open file in append mode
f.write(result + '\n')
f.close()

cap.release()
cv2.destroyAllWindows()
