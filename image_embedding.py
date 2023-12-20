import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset, ConcatDataset
from torchvision import datasets, models


# On crée une class custom pour labelliser nos images
class CustomDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = [os.path.join(folder_path, img_name) for img_name in os.listdir(folder_path)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")

        # On utilise le titre de l'image comme label
        label = os.path.splitext(os.path.basename(img_path))[0]

        if self.transform:
            img = self.transform(img)

        return img, label   


# on crée une classe pour le dataset des vecteurs de features des images transformées
class CustomVectorDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx][0]

        if self.transform:
            sample = self.transform(sample)
        
        return sample, label

# Image loading and feature extraction
def extract_features(image_folder_path, model, transform=None):
    dataset = CustomDataset(image_folder_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    features_list = []

    model.eval()
    with torch.no_grad():
        for images, label in dataloader:
            features = model(images)
            features = features.cpu().numpy().flatten()
            label_current = label[0]
            features_list.append((features,label))

    return features_list

# Create a new dataset of features extracted from training images
def create_train_dataset_features(image_folder_path, model, nb_transformation_per_image, transform=None):
	print("Creating dataset of embeddings..")
	features_list = []
	for _ in range(nb_transformation_per_image):
		features_list.extend(extract_features(image_folder_path, model, transform))

	all_features = np.array([features for features, _ in features_list])
	all_labels = np.array([label for _, label in features_list])

	all_features_tensor = torch.tensor(all_features)

	vectordataset = CustomVectorDataset(all_features_tensor, all_labels)
	vectordataloader = DataLoader(vectordataset,batch_size=1, shuffle=True)

	print("Dataset of embeddings created")
	return vectordataloader

# Load pre-trained ResNet18 model
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1])) # on enlève la dernière couche de classification

# Image folder path
image_folder_path = 'data/DAM_extraction/'

# Transformation applied to each image
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),
    #transforms.RandomErasing(),

    # les transformations pour normaliser avec les données d'entrainement de ResNet18
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Number of transformation used by image
nb_transformation_per_image = 8

extracted_features_data_set = create_train_dataset_features(image_folder_path, model, nb_transformation_per_image ,transform=train_transform)
# torch.save(train_dataloader,'train_dataloader.pth')

# Function to calculate cosine similarity between two feature vectors
def calculate_cosine_similarity(feature1, feature2):
	similarity = cosine_similarity(feature1, feature2)
	return similarity

def get_top_similar_images(input_image_path, features_dataloader, model, preprocess, top_k=10):
    print("Looking for similar images..")
    # Load the input image
    input_image = Image.open(input_image_path).convert("RGB")

    # Preprocess the input image
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    # Extract features for the input image
    with torch.no_grad():
        input_features = np.array(model(input_batch).cpu().numpy().flatten())
    input_features = np.reshape(input_features, (1, -1))

    # Calculate cosine similarity with the features in the features_dataloader
    similarities = []

    for features, label in features_dataloader:
        similarity = calculate_cosine_similarity(input_features, features)
        label_current = label[0]
        similarities.append((similarity,label_current))

    sorted_similarities = sorted(similarities, key=lambda x: x[0], reverse=True)
    top_k_labels = [label for similarity, label in sorted_similarities[:top_k]]

    return top_k_labels

if __name__ == '__main__':
    input_image_path = "data/DAM/02JHE090I610C905.jpeg"

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    top_10 = get_top_similar_images(input_image_path, extracted_features_data_set, model, preprocess, top_k=10)

    print("The label to find is : 02JHE090I610C905")
    print("The top 10 similar images are :")
    for label in top_10:
        print(label)
        image_path = "data/DAM/"+label+".jpeg"
        img = mpimg.imread(image_path)
        plt.imshow(img)
        plt.title(label)
        plt.show()
