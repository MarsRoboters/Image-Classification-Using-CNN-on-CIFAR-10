# Image-Classification-Using-CNN-on-CIFAR-10

**Overview:**
  - This project involved developing a convolutional neural network (CNN) to classify images from the CIFAR-10 dataset. The model was trained to recognize 10 different classes of objects, achieving an accuracy of 77.78%.

**Problem Statement:** 
  - The goal was to create an efficient image classification model that can accurately identify objects in images, which is a fundamental task in computer vision with applications in various fields such as autonomous driving, facial recognition, and more.

**Dataset:**
  - The dataset used was CIFAR-10, which consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The dataset was split into 50,000 training images and 10,000 testing images. Key attributes included pixel values and class labels.

**Methodology:**

1. Data Preprocessing:
    - Normalized the pixel values to be between 0 and 1.
    - Applied one-hot encoding to the class labels.
2. Model Architecture:
    - Used a Sequential model with multiple Conv2D layers.
    - Applied ReLU activation functions and BatchNormalization.
    - Included Dropout layers to prevent overfitting.
    - Used MaxPooling2D layers to reduce spatial dimensions.
    - Flattened the output and added Dense layers with ReLU activation.
    - Final output layer used softmax activation for classification.
3. Training:
    - Compiled the model with categorical crossentropy loss and Adam optimizer.
    - Trained the model for 10 epochs with a batch size of 64.

**Key Results** 
  - The CNN model achieved an accuracy of 77.78% on the test set. This performance indicates the model’s effectiveness in classifying images into the correct categories.

**Challenges and Solutions:** 
  - One challenge was preventing overfitting due to the complexity of the model. This was addressed by incorporating Dropout layers and BatchNormalization, which helped in regularizing the model and maintaining consistent activation distributions.
