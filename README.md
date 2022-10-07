# DL_CarPlates_23
Распознавание автомобильных номеров RU-региона на основе искусственного интеллекта

## Описание данных

В качестве исходных данных для решения задачи был использован [датасет](https://disk.yandex.ru/d/NANSgQklgRElog), состоящий из более чем 25 тысяч изображений автомобилей для обучения моделей и более чем 3 тысяч изображений для тестирования обученных моделей. Вместе с датасетом прилагается файл train.json, содержащий разметку для тестовой части.

## Выбор и обоснование метрик

В качестве метрики на этапе детекции была выбрана [IoU](https://medium.com/analytics-vidhya/iou-intersection-over-union-705a39e7acef) - метрика, рассчитываемая как отношение площади пересечения истинного и предсказанного bounding box'ов к площади их объединения.

![IoU](https://habrastorage.org/r/w1560/webt/hg/xi/zj/hgxizjtbx6vlispj9rl8nzfvhsu.png)

Данные по подбору гиперпараметров представлены в таблице ниже.

| Модель | Оптимизатор | Learning rate | IOU |
| --- | --- | --- | --- |
| FastRCNN | SGD | 0.005 | 0.8144 |
| FastRCNN | SGD | 0.001 | 0.843 |
| FastRCNN | Adam | 0.005 | 0.8123 |
| FastRCNN | Adam | 0.001 | 0.838 |

В качестве метрики на этапе распознания текста на автомобильных номерах было выбрано [расстояние Левенштейна](https://ru.wikipedia.org/wiki/%D0%A0%D0%B0%D1%81%D1%81%D1%82%D0%BE%D1%8F%D0%BD%D0%B8%D0%B5_%D0%9B%D0%B5%D0%B2%D0%B5%D0%BD%D1%88%D1%82%D0%B5%D0%B9%D0%BD%D0%B0) - метрика, измеряющая по модулю разность между двумя последовательностями символов. Она определяется как минимальное количество односимвольных операций (вставки, удаления, замены), необходимых для превращения одной последовательности символов в другую.

---
## Установка зависимостей

Наш проект используею python 3.10

Зависимости для нашего проекта можно установить 2-мя способами:

1. Используя файл requirements.txt:

    <code>python -m pip install -r requirements.txt</code>

1. Используя [poetry](https://python-poetry.org/)
        
    <code>poetry install</code>

> Важно: сейчас в зависимостях указаны версии torch и torchvision для gpu и cuda version 11.6, если у вас нет gpu, или установлена другая cuda - то нужно использовать [другую версию библиотек](https://download.pytorch.org/whl/torch/).

## Запуск пайплайна

1. Скачайте обученные модели и сохраните их в папку models:
    - [Модель для детекции](https://drive.google.com/file/d/1KzihK_plnYH-muKX_uLwXz3HXNJusatA/view?usp=sharing)
    - [Языковую модель](https://drive.google.com/file/d/1VTMGXVv6EhlLH8Mbn1njy0oCH5W3JWI3/view?usp=sharing)
    - [Модель для распознавания текста](https://drive.google.com/file/d/1VcI-mEBKsDv0k79lcocxwvAMAYX6dl9v/view?usp=sharing)

1. Скачайте [датасет от компании VK](https://disk.yandex.ru/d/NANSgQklgRElog) и распакуйте его в папку data

1. Запуск - inference.py

    Скрипт обрабатывает одно изображение:

    - детекция
    - обработка найденных автомобильных номеров
    - распознавание текста
    - постобработка
    
    Результат выводится в терминал, а также сохраняется в формате jpg в папку output

    Параметры для использования скрипта:

        -h, --help            show this help message and exit
        -img IMG              path to image
        -output OUTPUT        output path
        -detection_model DETECTION_MODEL
                                path to model
        -recognition_model RECOGNITION_MODEL
                                path to model

## Обучение моделей

1. Скачайте [датасет от компании VK](https://disk.yandex.ru/d/NANSgQklgRElog) и распакуйте его в папку data

1. Detection model - FasterRCNN - train_detection_model.py

    Параметры для использования скрипта::

        -h, --help            show this help message and exit
        -data DATA            dataset path
        -output OUTPUT        output path
        -num_epochs NUM_EPOCHS
                                train epochs
        -batch_size BATCH_SIZE
                                batch size
        -exp_name EXP_NAME    experiment name

1. Подготовьте датасет для обучения модели распознавания текста:

    create_ocr_dataset.py может быть использован с такими параметрами:

        -h, --help      show this help message and exit
        -data DATA      data path
        -output OUTPUT  output path

1. Обучение модели распознавания - train_recognition_model.py

    Параметры для использования скрипта::

        -h, --help            show this help message and exit
        -data DATA            dataset path
        -output OUTPUT        output path
        -num_epochs NUM_EPOCHS
                                train epochs
        -batch_size BATCH_SIZE
                                batch size
        -exp_name EXP_NAME    experiment name

1. Построение языковой модели для beam search'a - create_language_model.py


## Пример работы системы (on test.jpg):

Y654BE77 [ 96.55244 434.03857 185.2982  480.91248] p=0.9996976852416992

Сохраненный в виде изображения результат:

![Example](https://raw.githubusercontent.com/PetrovitchSharp/DL_CarPlates_23/blob/dev/img/inference_example.jpg)


---
## Описание системы

### Детекция
Мы используем сеть Faster RCNN с backbone resnet50.

### Обработка найденных автомобильных номеров:
- Увеличение контрастности изображений (приводим к одному и тому же значению контрастности)
- Выравнивание баланса цветов (каждый канал отдельно преобразовывался так чтобы <=15% пикселей были равны 0 или 255, а остальные растягивались на весь промежуток)
- Увеличение резкости с использованием гауссиана (изображение размывается гауссианом, а затем размытое, умноженное на константу, вычитается из исходного изображения)

Пример результата обработки изображения:

![Figure 1](https://raw.githubusercontent.com/PetrovitchSharp/DL_CarPlates_23/blob/dev/img/augmented_plates_1.png)

![Figure 2](https://raw.githubusercontent.com/PetrovitchSharp/DL_CarPlates_23/blob/dev/img/augmented_plates_2.png)

### Распознавание текста
Мы используем модель CRNN с CTC лоссом.

### Постобработка
Внесение правок в предсказание модели с использованием [beam search](https://towardsdatascience.com/foundations-of-nlp-explained-visually-beam-search-how-it-works-1586b9849a24)'a и языковой модели

## Производительность:

Рассчет производительности был выполнен на следующих комплектующих:

GPU: NVIDIA GeForce RTX 3090 (24 GB, 1.70 GHz, 10496 CUDA cores)

CPU: AMD Ryzen 9 5900X (12 cores, 24 Threads, 3.7 GHz) 

### Обучение:

Модель детекции: 15 min/epoch

Модель распознания текста: 1.25 min/epoch

### Инференс:

На GPU: 0.57 sec/image

На CPU: 1.33 sec/image

