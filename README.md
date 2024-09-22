# Image-Classification-Using-CNN-on-CIFAR-10

**Overview:**
  - This project involved developing a convolutional neural network (CNN) to classify images from the CIFAR-10 dataset. The model was trained to recognize 10 different classes of objects, achieving an accuracy of 85.77%.

**Introduction:**
  - Image classification is a fundamental task in computer vision with applications in various fields such as autonomous driving, medical imaging, and more. The CIFAR-10 dataset is a widely used benchmark for evaluating image classification algorithms.

**Problem Statement:** 
  - The objective of this project was to develop a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The goal was to achieve high accuracy in recognizing these classes.

**Dataset:**

  - Dataset Used: CIFAR-10
  - Number of Classes: 10 
  - Classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
  - Training Samples: 50,000
  - Testing Samples: 10,000
  - Key attributes included pixel values and class labels.

**Methodology:**

  - Data Preprocessing:
    - Normalized the pixel values to be between 0 and 1.
    - Applied one-hot encoding to the class labels.
  - Model Architecture:
    - The CNN model was built using the Keras library with the following architecture:
      - Convolutional Layers: Used a Sequential model with multiple Conv2D layers with ReLU activation and Batch Normalization.
      - Pooling Layers: MaxPooling2D layers to reduce spatial dimensions.
      - Dropout Layers: Dropout layers to prevent overfitting.
      - Fully Connected Layers: Dense layers with ReLU activation and Batch Normalization.
      - Output Layer: Dense layer with softmax activation for classification.
  - Training:
    - Compiled the model with categorical cross-entropy loss and Adam optimizer.
    - Trained the model for 50 epochs with a batch size of 64.
    - The training process included data augmentation and callbacks for learning rate scheduling and early stopping.
   
**Techniques Implemented (for generalization):**
  - Data Augmentation: Applied rotation, width shift, height shift, horizontal flip, and zoom to increase the variability of the training data.
  - Learning Rate Scheduling: Adjusted the learning rate during training to improve convergence.
  - Early Stopping: Monitored validation loss to prevent overfitting.
  - Batch Normalization: Applied after each convolutional layer to stabilize and accelerate training.

**Key Results** 
  - The CNN model achieved an accuracy of 85.77% on the test set.
  - This performance indicates the model’s effectiveness in classifying images into the correct categories.
  - Visualization: The training and validation accuracy and loss were plotted to visualize the model’s performance over epochs.

**Future Work:**
  - Experiment with Different Architectures: Try deeper or more complex models.
  - Hyperparameter Tuning: Optimize batch size, learning rate, and other parameters.
  - Transfer Learning: Utilize pre-trained models and fine-tune them on the CIFAR-10 dataset.
  - Ensemble Methods: Combine predictions from multiple models to improve accuracy.
