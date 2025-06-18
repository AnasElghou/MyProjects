Project Overview

The goal of this project is to develop a deep learning model capable of accurately classifying brain tumors into different categories (e.g., glioma, meningioma, pituitary, and no tumor) using MRI scan images. The project utilizes a CNN architecture, data augmentation techniques, and TensorFlow/Keras for model training and evaluation.

Dataset

The dataset used for this project is the "Brain Tumor MRI Dataset" available on Kaggle Hub.

Training Data: Contains 5712 images belonging to 4 classes.
Testing Data: Contains 1311 images belonging to 4 classes.
The notebook splits the testing data further into validation and test sets for evaluation purposes.

Model Architecture

The CNN model is built using tf.keras.Sequential and consists of multiple convolutional layers followed by max-pooling layers, a flattening layer, and dense layers.

Input Shape: (256, 256, 3)
Layers:
Conv2D with 64 filters, (3,3) kernel, 'relu' activation
MaxPooling2D
Conv2D with 64 filters, (3,3) kernel, 'relu' activation
MaxPooling2D
Conv2D with 128 filters, (3,3) kernel, 'relu' activation
MaxPooling2D
Conv2D with 128 filters, (3,3) kernel, 'relu' activation
MaxPooling2D
Conv2D with 256 filters, (3,3) kernel, 'relu' activation
MaxPooling2D
Conv2D with 256 filters, (3,3) kernel, 'relu' activation
MaxPooling2D
Flatten
Dense with 256 units, 'relu' activation
Dense with 4 units (for 4 classes), 'softmax' activation

The model is compiled with the 'adam' optimizer and SparseCategoricalCrossentropy loss function, tracking 'accuracy' as a metric.

Results
The model achieves a high classification accuracy. During training, the accuracy on the validation set reached approximately 97.19%.

Precision: 0.87
Recall: 0.87
Accuracy: 0.935
Training and validation loss and accuracy plots are generated in the notebook to visualize the model's performance over epochs.