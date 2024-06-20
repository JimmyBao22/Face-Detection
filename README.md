# Face-Detection

This program's goal is to implement a face detector in natural images. I used the Face Detection Data Set and Benchmark Home: http://vis-www.cs.umass.edu/fddb/.
Regarding training/testing, I used the first 9 folders for training and the last 2 folders for testing.
Details can be find in the readme file of FDDBhttp://vis-www.cs.umass.edu/fddb/README.txt:
- Training: FDDB-fold-00, FDDB-fold-01, · · · , FDDB-fold-08;
- Testing: FDDB-fold-09, FDDB-fold-10;

## Approach

An image patch will be taken as input, and an output will be generated distinguishing whether the image contains a face in it or not. The training data generation is critical: non-face patches and faces that overlap with human objects but are too big or too small will be counted as negative instances.

## Objectives

The primary objectives that the models will attempt to acheive is the following
- High Accuracy
- Fast runtime

## Design Choices

For the normal face detection classifier algorithm, I employed two distinct approaches. I adhered to the standard pipeline of sliding windows and classification. For the specific classification algorithm, I evaluated both AdaBoost and Support Vector Machine (SVM) to determine which yielded better performance. To obtain the positive data, I parsed the ellipse files and converted the ellipses into rectangles. For generating negative data, I experimented with two different methods to identify the most effective approach.

The first method involved generating random negative samples, where I randomly generated bounding boxes for each image while ensuring they did not overlap with any rectangles considered to be faces. The advantages of this approach included its speed, the ability to specify the number of negative boxes per image, and the assurance that none of the negative samples overlapped with face rectangles. However, a limitation was the absence of 'negative' rectangles that included only parts of a face, as the generated boxes did not overlap with any face rectangles.

The second method utilized a sliding window approach, where I iterated over all rectangles in the image based on a specified step size. For each window, I calculated the Intersection over Union (IoU) with the face rectangles in the ellipse file, using an IoU threshold of 0.5. The advantage of this method was that it allowed for partial faces to be classified as either face or non-face. However, it was time-consuming, and the preset step sizes resulted in a loss of the randomness factor that could be beneficial.

Additionally, to improve the accuracy of the face detection classifier, I decided to implement a convolutional neural network (CNN) and incorporate additional features. I selected the AlexNet model. For generating negative samples, I used the sliding window approach mentioned earlier.

Initially, I started with the original AlexNet model. I also decided to incorporate Dropout Regularization, which is built into the AlexNet model. Dropout regularization is a technique used to prevent overfitting and improve the generalization performance of models. Its primary benefit is enhancing model robustness by randomly deactivating (or "dropping out") a fraction of neurons in a given layer during each forward and backward pass of the training process.

Next, I implemented Data Augmentation using built-in functions. Data augmentation involves applying various transformations to the existing training data to create additional augmented examples, increasing the diversity of the training dataset. This can improve the generalization and robustness of machine learning models. Common data augmentation techniques include rotations, image flipping, scaling, translation, cropping, and brightness adjustments.

Lastly, I incorporated weight regularization and batch normalization. Weight regularization, specifically L2 regularization, prevents overfitting by adding a penalty term to the loss function that encourages the model to maintain smaller weights. Batch normalization offers several benefits for training neural networks, including faster convergence, stabilized learning, improved weight initialization, and better generalization.

These adjustments aimed to enhance the performance and robustness of the CNN for more accurate face detection.

Note: From now on, I will refer to these two methods as Random Generation and Sliding Window, respectively.

### Algorithm Performances

- Adaboost Random Generation
  - File: Adaboost - RG.py
  - Accuracy: 0.5784
  - Training Completed in 355.15 seconds
- Adaboost Sliding Window
  - File: Adaboost - SW.ipynb
  - Accuracy: 0.6487
  - Training Completed in 467.76 seconds
- SVM Random Generation
  - File: SVM - RG.py
  - Accuracy: **0.9401**
  - Training Completed in **50.83** seconds
- SVM Sliding Window
  - File: SVM - SW.py
  - Accuracy: 0.7183
  - Training Completed in 512.45 seconds
- AlexNet CNN
  - File: CNN Classifiers.ipynb
  - Test 1
    - Accuracy: 0.9141
    - Training Completed in 307.86 seconds
  - Test 2
    - Accuracy: 0.9282
    - Training Completed in 323.09 seconds
  - Test 3
    - Accuracy: 0.9266
    - Training Completed in 308.85 seconds
  - Test 4
    - Accuracy: 0.9258
    - Training Completed in 226.14 seconds
- Alexnet CNN with Data Augmentation
  - File: CNN Classifiers.ipynb
  - Accuracy: 0.9310
  - Training Completed in 851.98 seconds
- AlexNet CNN with Weight Regularization & Batch Normalization
  - File: CNN Classifiers.ipynb
  - Accuracy: 0.8940
  - Training Completed in 268.85 seconds
