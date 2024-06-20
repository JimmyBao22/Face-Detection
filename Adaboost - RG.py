import cv2
import numpy as np
import random
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, accuracy_score
import os
import time
import matplotlib.pyplot as plt

def cotan(angle):
    return -np.tan(angle + np.pi/2)

# Function to convert ellipse to rectangular box
def ellipse_to_rectangle(major_axis, minor_axis, angle, center_x, center_y, image_size):
    t1 = np.arctan(-minor_axis * np.tan(angle) / major_axis)
    t2 = np.arctan(minor_axis * cotan(angle) / major_axis)
    xoff = abs(major_axis * np.cos(t1) * np.cos(angle) - minor_axis * np.sin(t1) * np.sin(angle))
    yoff = abs(minor_axis * np.sin(t2) * np.cos(angle) - major_axis * np.cos(t2) * np.sin(angle))
    x1 = center_x - xoff
    y1 = center_y - yoff
    x2 = center_x + xoff
    y2 = center_y + yoff
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > image_size[1]:
        x2 = image_size[1]
    if y2 > image_size[0]:
        y2 = image_size[0]
    return [x1, y1, x2, y2]

def get_image(image_path, scale=6):
    image = cv2.imread(image_path)
    new_height = int(image.shape[0] / scale)
    new_width = int(image.shape[1] / scale)
    return cv2.resize(image, (new_width, new_height))

# Parse FDDB dataset files and convert ellipse annotations to rectangles.
# Split into training and testing datasets.
def parse_fddb_dataset(folder_path, num_training=8):
    training_data = []
    testing_data = []

    for i in range(1, 11):
        file_path = os.path.join(folder_path, f'FDDB-fold-{i:02}-ellipseList.txt')
        with open(file_path, 'r') as file:
            lines = file.readlines()

        current_image = None
        face_count = 0
        get_face_count = False
        for line in lines:
            if get_face_count:
                face_count = int(line.strip())
                get_face_count = False
            elif face_count == 0:
                if current_image is not None:
                    # Process previous image
                    if i <= num_training:
                        training_data.append((current_image, faces))
                    else:
                        testing_data.append((current_image, faces))

                current_image = os.path.join('originalPics/', line.strip() + '.jpg')
                faces = []
                get_face_count = True
            else:
                parts = [float(part) for part in line.split()]
                image = cv2.imread(current_image)
                rect = ellipse_to_rectangle(*parts[:5], image.shape)
                faces.append(rect)
                face_count -= 1

        # Process the last image
        if current_image is not None:
            if i <= num_training:
                training_data.append((current_image, faces))
            else:
                testing_data.append((current_image, faces))

    return training_data, testing_data

# Generate negative samples that don't overlap with face annotations.
def generate_negative_samples(image_size, faces, num_samples, scale):
    negatives = []
    max_attempts = num_samples * 10
    iter = 0

    while len(negatives) < num_samples and iter < max_attempts:
        # width, height = random.randint(20, 100), random.randint(20, 100)
        width, height = 32, 32
        x1 = random.randint(0, image_size[1] - width)
        y1 = random.randint(0, image_size[0] - height)
        x2, y2 = x1 + width, y1 + height

        rect = [x1, y1, x2, y2]
        if not any(is_overlap(rect, face, scale) for face in faces):
            negatives.append(rect)

        iter += 1

    return negatives

# Checks if two boxes overlap.
def is_overlap(box1, box2, scale):
    x1, y1, x2, y2 = [b * scale for b in box1]
    x3, y3, x4, y4 = box2
    return not (x3 > x2 or x4 < x1 or y3 > y2 or y4 < y1)

# Function to extract features
def extract_features(image, box, scale):
    # don't include this image if a dimension = 0
    if int(box[3] / scale) == int(box[1] / scale) or int(box[2] / scale) == int(box[0] / scale):
        return [-1]
    cropped_image = image[int(box[1] / scale):int(box[3] / scale), int(box[0] / scale):int(box[2] / scale)]
    feature_vector = cv2.resize(cropped_image, (32, 32)).flatten()
    return feature_vector

def extract_dataset_features(data):
    # Prepare data for SVM classifier
    X, y = [], []
    scale = 2
    for image_path, boxes in data:
        image = get_image(image_path, scale)
        for box in boxes:
            feature = extract_features(image, box, scale)
            if feature[0] == -1:
                continue
            X.append(feature)
            y.append(1)  # Label for face

        # Generate and add negative samples
        negatives = generate_negative_samples(image.shape, boxes, 5, scale)
        for neg_box in negatives:
            feature = extract_features(image, neg_box, scale)
            if feature[0] == -1:
                continue
            X.append(feature)
            y.append(0)  # Label for non-face

    return X, y

training_data, testing_data = parse_fddb_dataset('FDDB-folds')

# Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train, y_train = extract_dataset_features(training_data)
X_test, y_test = extract_dataset_features(testing_data)

print("Got the data")

def generate_random_haar_features(num_features):
    features = []
    for i in range(num_features):
        # (x, y) is the position, w = width, h = height
        w = np.random.randint(0, 32)
        h = np.random.randint(0, 32)
        x = np.random.randint(0, 32 - w)
        y = np.random.randint(0, 32 - h)
        features.append((w, h, x, y))
    return np.array(features)

def compute_integral_image(image):
    # grayscale image
    grayscale = cv2.cvtColor(image.reshape(32, 32, 3), cv2.COLOR_RGB2GRAY)

    integral_image = cv2.integral(grayscale)

    return integral_image

def apply_haar_features(integral_image, features):
    feature_values = []
    for feature in features:
        w, h, x, y = feature
        white_rectangle = integral_image[y + h, x + w // 2] - integral_image[y, x + w // 2] - integral_image[y + h, x] + integral_image[y, x]
        black_rectangle = integral_image[y + h, x + w] - integral_image[y + h, x + w // 2] - integral_image[y, x + w] + integral_image[y, x + w // 2]
        feature_values.append(white_rectangle - black_rectangle)

    return np.array(feature_values)

# train Adaboost classifiers for each class (one-versus-all)
# this uses the weak feature computation (haar features)
def adaboost_train(X_train, y_train, num_features, class_label, T):
    y_train_binary = (y_train == class_label)

    weak_classifiers = []

    # initialize weights based on y_i = 0 or 1
    weights = np.where(y_train == 0, 1.0 / (2 * np.sum(y_train_binary == 0)), 1.0 / (2 * np.sum(y_train_binary == 1)))

    # for t = 1...T
    for t in range(T):
        # normalize the weights
        weights = weights / np.sum(weights)

        # train a classifier for each feature
        min_error = float('inf')
        best_feature_index = 1
        best_threshold = 1
        for j in range(num_features):
            features = X_train[:, j]
            threshold = np.random.uniform(min(features), max(features))
            predictions = np.where(features >= threshold, 1, 0)

            error = np.sum(weights * abs(predictions - y_train_binary))

            if error < min_error:
                min_error = error
                best_feature_index = j
                # choose best classifier with lowest error
                best_threshold = threshold

        if min_error == float('inf'):
            continue

        if min_error == 0:
            break

        beta = (1 - min_error) / min_error
        alpha = np.log(1 / beta)

        # update the weights
        predictions = np.where(X_train[:, best_feature_index] >= best_threshold, 1, 0)
        weights *= np.power(beta, 1 - abs(predictions - y_train_binary))

        # add to full array of classifiers
        weak_classifiers.append([alpha, best_threshold, best_feature_index])

    return weak_classifiers

# multi-class image classifier combining one-versus-all classifiers
def train_multiclass_adaboost(X_train, y_train, T):
    num_samples, num_features = X_train.shape

    classes = np.unique(y_train)
    num_classes = len(classes)

    classifiers_list = []

    for i in range(num_classes):
        # Train one-versus-all classifier for each class
        classifiers_list.append(adaboost_train(X_train, y_train, num_features, classes[i], T))

    return classifiers_list

def adaboost_predict(X_test, classifiers_list):
    num_samples, num_features = X_test.shape

    predictions = np.zeros((num_samples, len(classifiers_list)))

    for i, classifier in enumerate(classifiers_list):
        current_scores = np.zeros(num_samples)
        # calculate scores for each classifier in classifier list
        for alpha, threshold, feature_index in classifier:
            weak_predictions = X_test[:, feature_index] >= threshold
            current_scores += alpha * weak_predictions
        predictions[:, i] = current_scores

    # return the class with the highest score
    return np.argmax(predictions, axis=1)

features = generate_random_haar_features(2000)
X_train_integral_images = [compute_integral_image(image) for image in X_train]
X_test_integral_images = [compute_integral_image(image) for image in X_test]

X_train_haar = np.array([apply_haar_features(image, features) for image in X_train_integral_images])
X_test_haar = np.array([apply_haar_features(image, features) for image in X_test_integral_images])

start_time = time.time()
adaboost_classifier_list = train_multiclass_adaboost(X_train_haar, y_train, 100)
training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds.")

# Calculate accuracy
predictions = adaboost_predict(X_test_haar, adaboost_classifier_list)
predictions = [1 if x == 1 else 0 for x in predictions]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

precision, recall, _ = precision_recall_curve(y_test, predictions)

# Plotting the Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label='Adaboost')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend()
plt.show()