1. Объединение меток классов в один файл - ./notebooks/data_preparation.ipynb
2. Очистка данных, формирование датасетов для обучения - ./notebooks/image_deduplication.ipynb
3. Подготовка датасета для обучения НС детектировать головы и лица людей - ./notebooks/data_people_preparation.ipynb
4. Обучение НС по детекции ГРЗ - ./notebooks/carplate.ipynb
5. Обучение НС по детекции людй и автомобилей - ./notebooks/car_human.ipynb
6. Обучение НС по детекции голов и лиц людей - ./notebooks/head_face.ipynb
7. Обработка детекции людй и автомобилей - ./notebooks/concatenate_detections.ipynb

Формирование ответа по задаче выполняется при помощи команд:
sed 2d sample*.csv > submission.csv
sed -i 's/,/;/g' submission.csv

Итоговый ответ доступен по ссылке - https://drive.google.com/drive/folders/1rfxAiIn_0_6fYSk5K07Vo6j7EzqfBHNp?usp=sharing

Для воспроизведения обкчения и работы нейронных сетей требуются проекты Yolov5, Yolov7.
Файл ./src/v5_detect_carplate.py необходимо скопировать в директорию с проектом Yolov5.
Файлы ./src/v7_detect_car_human.py, ./src/v7_detect_hhf.py необходимо скопировать в директорию с проектом Yolov7.

Подготовленные наборы данных доступны по ссылке - https://drive.google.com/drive/folders/12Wu98pj86vJ8iR6mrjjd3Q4brAsH2EQW?usp=sharing
Для использования их необходимо разархивировать и расположить по пути ./data/

Обученные модели доступны по ссылке - https://drive.google.com/drive/folders/1rCov9CxfGA62Nhzl33_aRgDf6zCsRYSu?usp=sharing
Для использования их необходимо разархивировать и расположить по пути ./yolov7/runs/train/
