import math
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

# Путь до модели YOLO
MODEL_PATH = 'best.pt'

# Опция для отображения пользовательского интерфейса
SHOW_UI = True

# Функция для преобразования координат в реальные координаты
def convert_to_real_coordinates(coordinates, resolution):
    new_coordinates = []

    for coord in coordinates:
        x = int(coord[0] * resolution[0])
        y = int(coord[1] * resolution[1])
        new_coordinates.append([x, y])

    return new_coordinates

# Чтение данных из JSON-файла
json_file = open(os.path.join(SOURCE_DIR, 'markup', 'jsons', VIDEO_NAME + '.json'))
json_data = json.load(json_file)
areas = json_data['areas']

# Инициализация модели YOLO
model = YOLO(MODEL_PATH)
model.to('cuda')

# Загрузка видео
cap = cv2.VideoCapture(os.path.join(SOURCE_DIR, 'video', VIDEO_NAME + '.mp4'))

# Инициализация структур для хранения истории движения объектов
track_center_history = defaultdict(lambda: [])
contains_center_history = defaultdict(lambda: [])

frame_num = 1

# Цикл обработки каждого кадра видео
while cap.isOpened():
    success, frame = cap.read()

    frame_num += 1
    if frame_num % 3 != 0:
        continue

    if success:
        # Применение модели YOLO для отслеживания объектов на кадре
        results = model.track(frame, persist=True)

        if results[0].boxes.id is not None:

            boxes = results[0].boxes.xywh.cpu()

            track_ids = results[0].boxes.id.int().cpu().tolist()

            annotated_frame = results[0].plot()

            # Отображение областей интереса на кадре
            for area in areas:
                real_coordinates_areas = convert_to_real_coordinates(area, (frame.shape[1], frame.shape[0]))
                pts = np.array(real_coordinates_areas, np.int32)
                cv2.polylines(annotated_frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

                bbPath = mplPath.Path(pts)

                # Отображение движущихся объектов и их истории движения
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box

                    track = track_center_history[track_id]
                    track.append((float(x), float(y) + float(h/2)))

                    contains_arr = contains_center_history[track_id]
                    contains_arr.append(bbPath.contains_point((float(x), float(y) + float(h/2))))

                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))

                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 0, 0), thickness=1)

                    contained_points = []

                    for point, contains in zip(points, contains_arr):
                        if contains:
                            contained_points.append(point)

                    if len(contained_points) > 0:
                        cv2.polylines(annotated_frame,
                                  [np.hstack(contained_points).astype(np.int32).reshape((-1, 1, 2))],
                                  isClosed=False,
                                  color=(0, 230, 0), thickness=6)


            # Отображение результатов отслеживания объектов
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    else:
        break

cap.release()
cv2.destroyAllWindows()
