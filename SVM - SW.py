import cv2
import numpy as np
import random
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, accuracy_score
import os
from sklearn.utils import shuffle
import time
import matplotlib.pyplot as plt

def cotan(angle):
    return -np.tan(angle + np.pi/2)

# Function to convert ellipse to rectangular box
def ellipse_to_rectangle(major_axis, minor_axis, angle, center_x, center_y):
    t1 = np.arctan(-minor_axis * np.tan(angle) / major_axis)
    t2 = np.arctan(minor_axis * cotan(angle) / major_axis)
    xoff = abs(major_axis * np.cos(t1) * np.cos(angle) - minor_axis * np.sin(t1) * np.sin(angle))
    yoff = abs(minor_axis * np.sin(t2) * np.cos(angle) - major_axis * np.cos(t2) * np.sin(angle))
    x1 = center_x - xoff
    y1 = center_y - yoff
    x2 = center_x + xoff
    y2 = center_y + yoff
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
                rect = ellipse_to_rectangle(*parts[:5])
                faces.append(rect)
                face_count -= 1

        # Process the last image
        if current_image is not None:
            if i <= num_training:
                training_data.append((current_image, faces))
            else:
                testing_data.append((current_image, faces))

    return training_data, testing_data

def calculate_iou(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the intersection over union
    iou = intersection_area / float(boxAArea + boxBArea - intersection_area)
    return iou

def sliding_window(image, step_size, window_size):
    for x in range(0, image.shape[1], step_size):
        for y in range(0, image.shape[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def generate_samples_using_sliding_window(image, faces, window_size, step_size, iou_threshold, scale=6):
    positive_samples = []
    negative_samples = []
    
    for (x, y, window) in sliding_window(image, step_size, window_size):
        # if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
        #     continue

        scaled_box = [x * scale, y * scale, (x + window_size[0]) * scale, (y + window_size[1]) * scale]
        # print(scaled_box, faces)
        ious = [calculate_iou(scaled_box, face) for face in faces]
        max_iou = max(ious) if ious else 0
        # print(max_iou)

        current_box = [x, y, x + window_size[0], y + window_size[1]]
        if max_iou >= iou_threshold:
            positive_samples.append(current_box)
        else:
            negative_samples.append(current_box)
    
    return positive_samples, negative_samples

# Function to extract features in an image
def extract_features(image, box):
    # don't include this image if a dimension = 0
    # if int(box[3]) == int(box[1]) or int(box[2]) == int(box[0]):
    #     return [-1]
    cropped_image = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
    # feature_vector = cropped_image.flatten()
    feature_vector = cv2.resize(cropped_image, (32, 32)).flatten()
    return feature_vector

def extract_dataset_features(data):
    # Prepare data for classifier
    # X, y = [], []
    X_pos, X_neg, y_pos, y_neg = [], [], [], []
    count_pos = 0
    count_neg = 0
    scale = 3
    for image_path, boxes in data:
        image = get_image(image_path, scale)
        pos_samples, neg_samples = generate_samples_using_sliding_window(image, boxes, window_size=(32, 32), step_size=4, iou_threshold=0.5, scale=scale)
        # print(neg_samples)
        for box in pos_samples:
            feature = extract_features(image, box)
            # if feature[0] == -1:
            #     continue
            count_pos += 1
            X_pos.append(feature)
            y_pos.append(1)  # Label for face

        for neg_box in neg_samples:
            feature = extract_features(image, neg_box)
            # if feature[0] == -1:
            #     continue
            count_neg += 1
            X_neg.append(feature)
            y_neg.append(0)  # Label for non-face

    # standardize the dataset
    if count_pos < count_neg:
        data_size = count_pos
        random_rows = random.sample(range(count_neg), data_size)
        X_neg = [X_neg[i] for i in random_rows]
        y_neg = [y_neg[i] for i in random_rows]
    else:
        data_size = count_neg
        random_rows = random.sample(range(count_pos), data_size)
        X_pos = [X_pos[i] for i in random_rows]
        y_pos = [y_pos[i] for i in random_rows]

    X = np.array(X_pos + X_neg)
    y = np.array(y_pos + y_neg)

    X, y = shuffle(X, y)

    return X, y

training_data, testing_data = parse_fddb_dataset('FDDB-folds')

X_train, y_train = extract_dataset_features(training_data)
X_test, y_test = extract_dataset_features(testing_data)

print("got the data")

# Train an SVM classifier on the provided features and labels.
def train_svm_classifier(X_train, y_train):
    classifier = SVC(gamma='auto', verbose=True)
    classifier.fit(X_train, y_train)
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