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
        """
        Initializes a custom dataset for labeling images.
        Parameters:
        - folder_path (str): Path to the folder containing images.
        - transform (callable, optional): Optional transform to be applied on the images.
        Returns:
        None
        """
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = [os.path.join(folder_path, img_name) for img_name in os.listdir(folder_path)]

    def __len__(self):
        """
        Returns the number of images in the dataset.
        Parameters:
        None
        Returns:
        int: Number of images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves an image and its label from the dataset.
        Parameters:
        - idx (int): Index of the image to be retrieved.
        Returns:
        tuple: Tuple containing the image and its label.
        """
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        label = os.path.splitext(os.path.basename(img_path))[0] # On utilise le titre de l'image comme label
        if self.transform:
            img = self.transform(img)
        return img, label   


# On crée une classe pour le dataset des vecteurs de features des images transformées
class CustomVectorDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        """
        Initializes a custom dataset for vectors of image features.
        Parameters:
        - data (Tensor): Tensor containing feature vectors.
        - labels (ndarray): Numpy array containing labels corresponding to the feature vectors.
        - transform (callable, optional): Optional transform to be applied on the feature vectors.
        Returns:
        None
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """
        Returns the number of feature vectors in the dataset.
        Parameters:
        None
        Returns:
        int: Number of feature vectors in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a feature vector and its label from the dataset.
        Parameters:
        - idx (int): Index of the feature vector to be retrieved.
        Returns:
        tuple: Tuple containing the feature vector and its label.
        """
        sample = self.data[idx]
        label = self.labels[idx][0]

        if self.transform:
            sample = self.transform(sample)
        
        return sample, label


# Image loading and feature extraction
def extract_features(image_folder_path, model, transform=None):
    """
    Extracts features from images using a specified model.

    Parameters:
    - image_folder_path (str): Path to the folder containing images.
    - model (nn.Module): PyTorch model for feature extraction.
    - transform (callable, optional): Optional transform to be applied on the images.

    Returns:
    list: List of tuples containing feature vectors and their corresponding labels.
    """
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
    """
    Creates a dataset of feature vectors extracted from training images.

    Parameters:
    - image_folder_path (str): Path to the folder containing training images.
    - model (nn.Module): PyTorch model for feature extraction.
    - nb_transformation_per_image (int): Number of transformations per image.
    - transform (callable, optional): Optional transform to be applied on the images.

    Returns:
    DataLoader: PyTorch DataLoader containing the dataset of feature vectors.
    """
    print("Creating dataset of embeddings..")
    features_list = []
    for _ in range(nb_transformation_per_image):
        features_list.extend(extract_features(image_folder_path, model, transform))

    print("Creation of embeddings of size "+str(np.shape(features_list[0][0][0]))+" for each image")
    all_features = np.array([features for features, _ in features_list])
    all_labels = np.array([label for _, label in features_list])

    all_features_tensor = torch.tensor(all_features)

    vectordataset = CustomVectorDataset(all_features_tensor, all_labels)
    vectordataloader = DataLoader(vectordataset,batch_size=1, shuffle=True)

    print("Dataset of embeddings created")
    return vectordataloader


# Load pre-trained ResNet18 model
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1])) # On supprime la dernière couche de classification


# Image folder path
image_folder_path = 'data/DAM_extraction/'


# Transformation applied to each image
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),
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
    """
    Calculates the cosine similarity between two feature vectors.
    Parameters:
    - feature1 (ndarray): First feature vector.
    - feature2 (ndarray): Second feature vector.
    Returns:
    ndarray: Cosine similarity between the two feature vectors.
    """
    similarity = cosine_similarity(feature1, feature2)
    return similarity


# Fonction qui renvoie un top_k d'images dans la base de donnée les plus proches de l'image donnée en entrée
def get_top_similar_images(input_image_path, features_dataloader, model, preprocess, top_k=10):
    """
    Finds the top k similar images in the dataset given an input image.

    Parameters:
    - input_image_path (str): Path to the input image.
    - features_dataloader (DataLoader): PyTorch DataLoader containing the dataset of feature vectors.
    - model (nn.Module): PyTorch model for feature extraction.
    - preprocess (callable): Preprocessing function for input images.
    - top_k (int, optional): Number of similar images to retrieve.

    Returns:
    list: List of labels for the top k similar images.
    """
    print("Looking for similar images..")
    input_image = Image.open(input_image_path).convert("RGB")
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        input_features = np.array(model(input_batch).cpu().numpy().flatten())
    input_features = np.reshape(input_features, (1, -1))
    
    similarities = []
    for features, label in features_dataloader:
        similarity = calculate_cosine_similarity(input_features, features)
        label_current = label[0]
        similarities.append((similarity,label_current))

    sorted_similarities = sorted(similarities, key=lambda x: x[0], reverse=True)
    top_k_labels = [label for similarity, label in sorted_similarities[:top_k]]
    return top_k_labels

# Executed part of the code
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
