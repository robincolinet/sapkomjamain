print("Starting to import useful modules")

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

from itertools import combinations
import random

from rembg import remove
import cv2

print("Imported Modules")


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
        label = os.path.splitext(os.path.basename(img_path))[0] # On utilise le titre de l'image comme label
        if self.transform:
            img = self.transform(img)
        return img, label   


# On crée une classe pour le dataset des vecteurs de features des images transformées
class CustomVectorDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)
        
        return sample, label


# Image loading and feature extraction
def extract_features(training_image_folder_path, model_embedding, transform=None,n_transformed_images=1):
	dataset = CustomDataset(training_image_folder_path, transform=transform)
	# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
	features_list = []
	model_embedding.eval()
	with torch.no_grad():
		for i,(_,_) in enumerate(dataset):
			for j in range(n_transformed_images):
				image = dataset[i][0]
				features = model_embedding(image.unsqueeze(0))
				features = features.cpu().numpy().flatten()
				label = dataset[i][1]
				# label_current = label[0]
				features_list.append((features,label))
	return features_list


# Create a new dataset of features extracted from training images
def create_train_dataset_features(training_image_folder_path, model_embedding, nb_transformation_per_image, transform=None):
	print("Creating dataset of embeddings..")
	# features_list = []
	# for _ in range(nb_transformation_per_image):
	features_list=extract_features(training_image_folder_path, model_embedding, transform=transform,n_transformed_images=nb_transformation_per_image)

	print(f"Embeddings size : {np.shape(features_list[0][0])}")
	all_features = np.array([features for (features,_) in features_list])
	all_labels = np.array([label for (_,label) in features_list])

	vectordataset = CustomVectorDataset(all_features, all_labels)

	print("Dataset of embeddings created")
	return vectordataset


# To create the dataset the first time and to save it
def create_and_save_dataset(training_image_folder_path,model_embedding,nb_transformation_per_image,train_transform):
	extracted_features_data_set = create_train_dataset_features(training_image_folder_path, model_embedding, nb_transformation_per_image ,transform=train_transform)
	if os.path.exists('./extracted_features_data_set.pt'):
		os.remove('./extracted_features_data_set.pt')
	torch.save(extracted_features_data_set,'./extracted_features_data_set.pt')
	return extracted_features_data_set

if __name__ == '__main__':
	training_image_folder_path = 'data/DAM_extraction/'

	# Load pre-trained ResNet18 model
	model_embedding = models.resnet18(pretrained=True)
	model_embedding = torch.nn.Sequential(*(list(model_embedding.children())[:-1])) # On supprime la dernière couche de classification

	# Transformation applied to each image
	train_transform = transforms.Compose([
		transforms.RandomResizedCrop(256, scale=(0.7, 1.2)),
		transforms.RandomHorizontalFlip(),
		transforms.RandomVerticalFlip(),
		transforms.RandomRotation(22),
		# les transformations pour normaliser avec les données d'entrainement de ResNet18
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	nb_transformation_per_image = 16

	extracted_features_data_set = create_and_save_dataset(training_image_folder_path,model_embedding,nb_transformation_per_image,train_transform)
