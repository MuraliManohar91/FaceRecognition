import numpy as np
import cv2
import os
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical



# Path to the dataset
dataset_path = 'dataset'

# Parameters
IMG_HEIGHT, IMG_WIDTH = 150, 150

# Function to load images
def load_images(dataset_path, img_height, img_width):
    images = []
    labels = []
    label_map = {}
    current_label = 0

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_path):
            label_map[person_name] = current_label
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Error reading image {img_path}. It might be corrupted.")
                        continue
                    img = cv2.resize(img, (img_height, img_width))
                    images.append(img)
                    labels.append(current_label)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
            current_label += 1

    return np.array(images), np.array(labels), label_map

# Load dataset
images, labels, label_map = load_images(dataset_path, IMG_HEIGHT, IMG_WIDTH)

# Save the label map
with open('label_map.pickle', 'wb') as f:
    pickle.dump(label_map, f)

# Normalize images
images = images.astype('float32') / 255.0



# Convert labels to categorical
labels = to_categorical(labels, num_classes=len(label_map))

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

print(f"Number of training samples: {len(X_train)}")
print(f"Number of testing samples: {len(X_test)}")
print(f"Shape of images: {images.shape}")
print(f"Shape of labels: {labels.shape}")
