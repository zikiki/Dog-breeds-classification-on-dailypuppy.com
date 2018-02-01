## Dog Breeds Classification

This repo is intended to contain a set of scripts , Stanford dog breeds dataset and scrapped dog photos from dailypuppy.com for reproducing dog breed classification model training, analysis, and inference. Also this repo developed upon https://github.com/stormy-ua/dog-breeds-classification

### Prerequisites

1. Install Python 2.7
2. Install all required python dependencies: `pip install -r requirements.txt`

### Download Data

1. `cd` to this repo root directory
2. Execute setup script: `sh ./setup/setup.sh`. In its turn the script executes the following other scripts:
    * Creates all required directories: `sh ./create_dirs.sh`
    * Downloads Google Inception model: `sh ./inception_download.sh`. The link to the frozen TensorFlow model is taken from [here](https://github.com/tensorflow/models/blob/master/tutorials/image/imagenet/classify_image.py#L51)
    * Downloads [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/): `sh ./download_stanford_dogs_dataset.sh`

### Prepare Data

1. Convert downloaded Stanford Dogs Dataset to TensorFlow friendly [TFRecords](https://www.tensorflow.org/programmers_guide/datasets#consuming_tfrecord_data) file: `python -m src.data_preparation.stanford_ds_to_tfrecords`

### Train

This section describes how to build dog breed classification dense neural network model on top of the pre-trained by Google deep neural network (namely Inception model).

1. Give a name to the model your are going to train by assigning a name to `CURRENT_MODEL_NAME` variable in [consts.py](src/common/consts.py#L14) script
2. Configure number of layers and number of units in each layer by setting `HEAD_MODEL_LAYERS` variable in [consts.py](src/common/consts.py#L18)
3. Train the model: `python -m src.training.train`. It might take 25-35 minutes depending on the depth of your model and number of epochs (which could be configured in the train.py script itself). TensorBoard could be used to observe the training process: `tensorboard --logdir=./summary`

### Freeze Model

1. Freeze the model: `python -m src.freezing.freeze`. This will bind Inception model with the trained on the previous step "head" model and serialize it as a TensorFlow graph with variables represented as constants. This frozen model will be ready to use for classification tasks.

### Analyze

1. Produce CSV file with predicted vs actual breed. This could be used to analyze precision on the training data e.g. plot a confusion matrix (see [Confusion.ipynb](Confusion.ipynb)). Result CSV file goes to `metrics/training_confusion.csv`: `python -m src.analysis.training_perf_analysis`

### Use Selenium the web browser automation to scrap the dog images from http://www.dailypuppy.com/pupdates/. `python -m dog_scraper.webdriver`

### Infer and classify puppy photos on dailypuppy.com

1. Use the model frozen on the previous step to classify an image either available on the filesystem or downloadable as an HTTP resource:


`python -m src.inference.classify file images`
