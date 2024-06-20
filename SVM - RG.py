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
        width, height = 16, 16
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
    feature_vector = cv2.resize(cropped_image, (16, 16)).flatten()
    return feature_vector

def extract_dataset_features(data):
    # Prepare data for SVM classifier
    X, y = [], []
    scale = 3
    for image_path, boxes in data:
        image = get_image(image_path, scale)
        for box in boxes:
            feature = extract_features(image, box, scale)
            if feature[0] == -1:
                continue
            X.append(feature)
            y.append(1)  # Label for face

        # Generate and add negative samples
        negatives = generate_negative_samples(image.shape, boxes, 3, scale)
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

print("got the data")

class SimpleLinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        # n_samples, n_features = X.shape
        n_samples, n_features = len(X), len(X[0])
        y_ = np.where(np.array(y) <= 0, -1, 1)
        
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

# Train an SVM classifier on the provided features and labels.
def train_svm_classifier(X_train, y_train):
    classifier = SimpleLinearSVM()
    classifier.fit(X_train, y_train)
    # classifier = SVC(gamma='auto', verbose=True)
    # classifier.fit(X_train, y_train)
    return classifier

start_time = time.time()
classifier = train_svm_classifier(X_train, y_train)
training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds.")

# Calculate accuracy
predictions = classifier.predict(X_test)
predictions = [1 if x == 1 else 0 for x in predictions]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

precision, recall, _ = precision_recall_curve(y_test, predictions)

# Plotting the Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label='SVM')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend()
plt.show()