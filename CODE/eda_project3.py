import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import numpy as np
from collections import Counter

DATASET_PATH = "./Desktop/Spring 2025/DS Prototyping/Project3/COVID19-ImageDataset"  # <- update if needed

classes = ['COVID', 'PNEUMONIA', 'NORMAL']

# Count Images
def count_images(dataset_path, classes):
    counts = {}
    for label in classes:
        path = os.path.join(dataset_path, label)
        counts[label] = len(os.listdir(path))
    return counts

# Image Shape Dimensions
def get_image_shapes(dataset_path, classes, num_samples=100):
    shapes = []
    for label in classes:
        path = os.path.join(dataset_path, label)
        images = os.listdir(path)
        sampled_images = random.sample(images, min(num_samples, len(images)))
        for img_name in sampled_images:
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                shapes.append(img.shape)
    return shapes

# Visualizing Random Images (not rly needed)
def show_random_images(dataset_path, classes, num_images=5):
    plt.figure(figsize=(15, 5))
    for idx, label in enumerate(classes):
        path = os.path.join(dataset_path, label)
        img_name = random.choice(os.listdir(path))
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.subplot(1, len(classes), idx + 1)
        plt.imshow(img)
        plt.title(label)
        plt.axis('off')
    plt.show()

# Class distribution
def plot_class_distribution(counts):
    sns.barplot(x=list(counts.keys()), y=list(counts.values()))
    plt.title('Class Distribution')
    plt.ylabel('Number of Images')
    plt.show()
    
    

def plot_image_size_distribution(shapes):
    pixel_counts = [w * h for (h, w, _) in shapes]
    plt.figure(figsize=(8, 5))
    sns.histplot(pixel_counts, bins=30, kde=True)
    plt.title('Image Size (Pixel Count) Distribution')
    plt.xlabel('Number of Pixels (width x height)')
    plt.ylabel('Frequency')
    plt.show()

def plot_aspect_ratio_distribution(shapes):
    aspect_ratios = [w / h for (h, w, _) in shapes if h != 0]
    plt.figure(figsize=(8, 5))
    sns.histplot(aspect_ratios, bins=30, kde=True)
    plt.title('Aspect Ratio Distribution')
    plt.xlabel('Aspect Ratio (width / height)')
    plt.ylabel('Frequency')
    plt.show()


def plot_average_brightness(dataset_path, classes, num_samples=100):
    brightness = {label: [] for label in classes}
    
    for label in classes:
        path = os.path.join(dataset_path, label)
        images = os.listdir(path)
        sampled_images = random.sample(images, min(num_samples, len(images)))
        
        for img_name in sampled_images:
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Grayscale for brightness
            if img is not None:
                brightness[label].append(np.mean(img))
    
    plt.figure(figsize=(10, 6))
    for label, values in brightness.items():
        sns.kdeplot(values, label=label)
    plt.title('Average Brightness Distribution by Class')
    plt.xlabel('Average Pixel Intensity')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

# Hypothetical split
def sketch_split(counts, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    print("Proposed dataset split:")
    for label, total in counts.items():
        train = int(total * train_ratio)
        val = int(total * val_ratio)
        test = total - train - val
        print(f"{label}: Train={train}, Validation={val}, Test={test}")
        


if __name__ == "__main__":
    counts = count_images(DATASET_PATH, classes)
    print("Image counts per class:", counts)

    shapes = get_image_shapes(DATASET_PATH, classes)
    print("\nUnique image shapes found:", Counter(shapes))

    plot_class_distribution(counts)
    show_random_images(DATASET_PATH, classes)
    
    plot_image_size_distribution(shapes)
    plot_aspect_ratio_distribution(shapes)
    plot_average_brightness(DATASET_PATH, classes)

    sketch_split(counts)