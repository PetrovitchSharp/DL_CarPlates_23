# DL_CarPlates_23
Car plates recognition command line application

1. Training - train.py

    Training CLI can be used with following parameters:

        -h, --help            show this help message and exit
        -data DATA            dataset path
        -output OUTPUT        output path
        -num_epochs NUM_EPOCHS
                                train epochs
        -batch_size BATCH_SIZE
                                batch size
        -exp_name EXP_NAME    experiment name

1. Inference - inference.py

    Inference script make a prediction for an image. The result is printed to the console and saved to the output directory in jpg format.

    Inference CLI can be used with following parameters:

        -h, --help      show this help message and exit
        -img IMG        path to image
        -output OUTPUT  output path
        -model MODEL    path to model