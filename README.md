![image](https://github.com/user-attachments/assets/f0e81f2c-f04d-461e-9336-bd6ac7d371eb)


# Melanoma Case Study
To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.


<!-- You can include any other section that is pertinent to your problem -->

## General Information
Problem statement: To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution which can evaluate images and alert the dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.

The dataset consists of 2357 images of malignant and benign oncological diseases, which were formed from the International Skin Imaging Collaboration (ISIC). All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, with the exception of melanomas and moles, whose images are slightly dominant. The data set contains the following diseases:

Project Pipeline
Data Reading/Data Understanding → Defining the path for train and test images
Dataset Creation→ Create train & validation dataset from the train directory with a batch size of 32. Also, make sure you resize your images to 180*180.
Dataset visualisation → Create a code to visualize one instance of all the nine classes present in the dataset
Model Building & training : Create a CNN model, which can accurately detect 9 classes present in the dataset. While building the model rescale images to normalize pixel values between (0,1).
Choose an appropriate optimiser and loss function for model training
Train the model for ~20 epochs
Write your findings after the model fit, see if there is evidence of model overfit or underfit
Choose an appropriate data augmentation strategy to resolve underfitting/overfitting Model Building & training on the augmented data :

    Create a CNN model, which can accurately detect 9 classes present in the dataset. While building the model rescale images to normalize pixel values between (0,1).
    Choose an appropriate optimiser and loss function for model training
    Train the model for ~20 epochs
    Write your findings after the model fit, see if the earlier issue is resolved or not? **Class distribution: **
    Examine the current class distribution in the training dataset
    Which class has the least number of samples?
    Which classes dominate the data in terms of the proportionate number of samples? Handling class imbalances:
    Rectify class imbalances present in the training dataset with Augmentor library. Model Building & training on the rectified class imbalance data:
    Create a CNN model, which can accurately detect 9 classes present in the dataset. While building the model rescale images to normalize pixel values between (0,1).
    Choose an appropriate optimiser and loss function for model training
    Train the model for ~30 epochs

    Actinic keratosis
    Basal cell carcinoma
    Dermatofibroma
    Melanoma
    Nevus
    Pigmented benign keratosis
    Seborrheic keratosis
    Squamous cell carcinoma
    Vascular lesion

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->

## Technologies Used
```python
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import PIL
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical  # Correct import path
import Augmentor
```

## As the libraries versions keep on changing, it is recommended to use these versions of these libraries for this project
