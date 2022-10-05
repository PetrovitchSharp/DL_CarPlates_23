# DL_CarPlates_23
Car plates recognition command line application

1. Please download dataset from https://disk.yandex.ru/d/NANSgQklgRElog

1. Detection model - FasterRCNN - train_detection_model.py

    Training CLI can be used with following parameters:

        -h, --help            show this help message and exit
        -data DATA            dataset path
        -output OUTPUT        output path
        -num_epochs NUM_EPOCHS
                                train epochs
        -batch_size BATCH_SIZE
                                batch size
        -exp_name EXP_NAME    experiment name

1. Create dataset for text recognition

    create_ocr_dataset.py can be used with following parameters:

        -h, --help      show this help message and exit
        -data DATA      data path
        -output OUTPUT  output path

1. Recognition model training - train_recognition_model.py

    Training CLI can be used with following parameters:

        -h, --help            show this help message and exit
        -data DATA            dataset path
        -output OUTPUT        output path
        -num_epochs NUM_EPOCHS
                                train epochs
        -batch_size BATCH_SIZE
                                batch size
        -exp_name EXP_NAME    experiment name

1. Create car plates language model for beam search - create_language_model.py

1. Inference - inference.py

    Inference script make a prediction for an image. The result is printed to the console and saved to the output directory in jpg format.

    Inference CLI can be used with following parameters:

        -h, --help            show this help message and exit
        -img IMG              path to image
        -output OUTPUT        output path
        -detection_model DETECTION_MODEL
                                path to model
        -recognition_model RECOGNITION_MODEL
                                path to model

    Result will be printed to console and saved to the output folder as a jpg.


## Example of inference (on test.jpg):

Y654BE77 [ 96.55244 434.03857 185.2982  480.91248] p=0.9996976852416992

Also result is saved to output directory:

![Example](https://github.com/PetrovitchSharp/DL_CarPlates_23/blob/dev/inference_example.jpg)

## Performance:

Performance evaluation performed on the following hardware:

GPU: NVIDIA GeForce RTX 3090 (24 GB, 1.70 GHz, 10496 CUDA cores)

CPU: AMD Ryzen 9 5900X (12 cores, 24 Threads, 3.7 GHz) 

Train:

Recognition model: 15 min/epoch

Language model: 1.25 min/epoch

Inference:

On GPU: 0.57 sec/image

On CPU: 1.33 sec/image

## Choice and justification of the metric

The IOU metric was chosen as the detection metric. The selection of hyperparameters was carried out for the detection model. The data is presented in the table below.

| Model | Optimizer | Learning rate | Metric value (IOU) |
| --- | --- | --- | --- |
| FastRCNN | SGD | 0.005 | 0.8144 |
| FastRCNN | SGD | 0.001 | 0.843 |
| FastRCNN | Adam | 0.005 | 0.0 |
| FastRCNN | Adam | 0.001 | 0.0 |
