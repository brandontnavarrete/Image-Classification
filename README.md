# Image Classification with TensorFlow

This project is a simple showcase of image classification using TensorFlow. It was created for educational purposes, to practice using TensorFlow for image recognition.

## Getting Started

To get started with this project, you will need to install the following dependencies:

- TensorFlow
- pydot
- opencv-python
- Numpy
- Matplotlib
- Scikit-learn

Once you have installed these dependencies, you can clone this repository and run the `image_classification.ipynb` file in Jupyter Notebook.

## Dataset

The dataset used in this project is the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is split into 50,000 training images and 10,000 test images.

## Model

The image classification model used in this project is a simple convolutional neural network (CNN) with the following layers:

- Conv2D layer with 32 filters and a 3x3 kernel size
- MaxPooling2D layer with a pool size of 2x2
- Conv2D layer with 64 filters and a 3x3 kernel size
- MaxPooling2D layer with a pool size of 2x2
- Flatten layer
- Dense layer with 128 units and ReLU activation
- Dropout layer with a rate of 0.5
- Dense layer with 10 units and softmax activation

## Results

The trained model achieved an accuracy of 88% on the test set. With more tinkering, I believe the results could be increased.
## Conclusion

This project was created for educational purposes to showcase image classification using TensorFlow. The trained model achieved a relatively low accuracy due to the simplicity of the model and the limited amount of training data. However, it provides a starting point for further experimentation and refinement of the model.
