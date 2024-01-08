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
# from dataset_creation import CustomVectorDataset

print("Imported Modules")

torch.manual_seed(42)
random.seed(42)


# On crée une class custom pour labelliser nos images
class CustomDataset(Dataset):
	def __init__(self, folder_path, transform=None):
		self.folder_path = folder_path
		self.transform = transform
		self.image_paths = [os.path.join(folder_path, img_name) for img_name in os.listdir(folder_path)]
		data = []
		labels = []
		for image_path in self.image_paths:
			img = Image.open(image_path).convert("RGB")
			label = os.path.splitext(os.path.basename(image_path))[0] # On utilise le titre de l'image comme label
			if self.transform:
				img = self.transform(img)
			labels.append(label)
			data.append(img)
		self.data = data
		self.labels = labels

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, idx):
		return self.data[idx], self.labels[idx]


# On crée une classe pour le dataset des vecteurs de features des images transformées
class CustomVectorDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
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
	np.save('embeddings_dataset.npy',extracted_features_data_set.data)
	np.save('dataset_labels.npy',extracted_features_data_set.labels)
	# if os.path.exists('./extracted_features_data_set.pt'):
		# os.remove('./extracted_features_data_set.pt')
	# torch.save(extracted_features_data_set,'./extracted_features_data_set.pt')
	return extracted_features_data_set


# To load the already created dataset
def load_existing_dataset():
	print("Loading existing training dataset")
	loaded_data = np.load('embeddings_dataset.npy')
	loaded_labels = np.load('dataset_labels.npy')
	extracted_features_data_set = CustomVectorDataset(loaded_data, loaded_labels)
	# extracted_features_data_set = torch.load('extracted_features_data_set.pt')	
	print("Loaded dataset")
	return extracted_features_data_set



# Fonction qui renvoie un top_k d'images dans la base de donnée les plus proches de l'image donnée en entrée
def top_k_cosine_similarity(input_image_path, features_dataset, model_embedding, preprocess, top_k=10,verbose = 0):
	print("Looking for similar images based on a cosine similarity measure")

	input_image = Image.open(input_image_path)
	image = countoring_one_pic(input_image)

	if verbose == 1:
		plt.imshow(input_image)
		plt.show()

		plt.imshow(image)
		plt.show()

	input_tensor = preprocess(image)
	input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
	with torch.no_grad():
		input_features = np.array(model_embedding(input_batch).cpu().numpy().flatten())
	input_features = np.reshape(input_features, (1, -1))

	similarities = []
	for i,(features, label) in enumerate(features_dataset):
		similarity = cosine_similarity(input_features, features.reshape(1, -1))
		similarities.append((similarity,label,i))

	sorted_similarities = sorted(similarities, key=lambda x: x[0], reverse=True)
	top_k_images = sorted_similarities[:top_k]
	return top_k_images,input_features


def resize_and_center_on_white_background(image_backgroundless, new_size=(256, 256)):
	image_with_alpha = np.array(image_backgroundless)
	if image_with_alpha.shape[2] != 4:
		raise ValueError("The image does not have an alpha channel")

	# Calculate the resizing ratio
	ratio = min(new_size[0] / image_with_alpha.shape[1], new_size[1] / image_with_alpha.shape[0])
	new_dimensions = (int(image_with_alpha.shape[1] * ratio), int(image_with_alpha.shape[0] * ratio))

	# Resize the image with alpha channel
	resized_image_with_alpha = cv2.resize(image_with_alpha, new_dimensions, interpolation=cv2.INTER_AREA)

	# Create a white background
	white_background = np.ones((new_size[1], new_size[0], 3), dtype=np.uint8) * 255

	# Calculate offset to center the resized image on the white background
	x_offset = (new_size[0] - new_dimensions[0]) // 2
	y_offset = (new_size[1] - new_dimensions[1]) // 2

	y1, y2 = y_offset, y_offset + new_dimensions[1]
	x1, x2 = x_offset, x_offset + new_dimensions[0]

	# Blend the resized image with the white background using the alpha channel
	alpha_s = resized_image_with_alpha[:, :, 3] / 255.0
	alpha_l = 1.0 - alpha_s

	white_background[y1:y2, x1:x2] = white_background[y1:y2, x1:x2] * alpha_l[:, :, np.newaxis] + resized_image_with_alpha[:, :, :3] * alpha_s[:, :, np.newaxis]

	result_image = Image.fromarray(white_background)
	return result_image



def countoring_one_pic(image):
	output = remove(image)
	white_background_image = resize_and_center_on_white_background(output)
	return white_background_image


def create_pairs(vectors_list,labels_list,number_pairs,n_trans_im):
	possible_index_pairs = []
	for i in range(len(vectors_list)):
		for j in range(i+1,len(vectors_list)):
			possible_index_pairs.append((i,j))
	index_list = random.sample(possible_index_pairs,number_pairs//2)
	pairs_list = []
	# On commence par ajouter la moitié de pairs prises entièrement au hasard, la plupart d'images différentes donc
	for (i,j) in index_list:
		if random.random()<0.5:
			pair = torch.cat((torch.tensor(vectors_list[i]),torch.tensor(vectors_list[j]))),float(labels_list[i]==labels_list[j])
		else : 
			pair = torch.cat((torch.tensor(vectors_list[j]),torch.tensor(vectors_list[i]))),float(labels_list[j]==labels_list[i])
		pairs_list.append(pair)
	index_list = random.sample(range(len(vectors_list)),(number_pairs+1)//2)
	# Puis on ajoute des pairs d'images identiques qui ont subi des transformations différentes
	for i in index_list:
		rang = i//n_trans_im
		j = rang * n_trans_im + ((random.randint(1,n_trans_im-1) + (i % n_trans_im)) % n_trans_im)
		pair = torch.cat((torch.tensor(vectors_list[i]),torch.tensor(vectors_list[j]))),float(labels_list[i]==labels_list[j])
		pairs_list.append(pair)

	return pairs_list


class MyCombinedDataset(Dataset):
	def __init__(self, original_dataset,number_pairs,n_trans_im):
		self.original_dataset = original_dataset
		self.labels = original_dataset.labels 
		self.vectors = original_dataset.data
		self.number_pairs = number_pairs
		self.pairs_list = create_pairs(self.vectors,self.labels,number_pairs,n_trans_im)

	def __len__(self):
		return self.number_pairs

	def __getitem__(self, idx):
		return self.pairs_list[idx][0],self.pairs_list[idx][1]


# Define the SimpleMLP model
class SimpleMLP(nn.Module):
	def __init__(self, input_dim=512*2, hidden_dim=256*2):
		super(SimpleMLP, self).__init__()
		self.fc1 = nn.Linear(input_dim, hidden_dim)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(hidden_dim, 1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.sigmoid(x)
		return x


def create_and_train_model(original_dataset,number_pairs,n_trans_im):
	# Create the PairsDataset and Dataloader for training pairs
	pairs_dataset = MyCombinedDataset(original_dataset,number_pairs,n_trans_im)
	pairs_dataloader = DataLoader(pairs_dataset, batch_size=1, shuffle=True)

	# On vérifie la proportion de pairs correspondant au même objet dans l'entrainement
	total_vectors = 0
	total_identical = 0
	for (vector,label) in pairs_dataset:
		total_vectors+=1
		if label==1:
			total_identical+=1
	print(f'Il y a dans le dataset de pairs {int(total_identical/total_vectors*100)}% de pairs du même objet')

	# Instantiate the SimpleMLP model
	model_similarity = SimpleMLP()

	# Define the loss function and optimizer
	criterion = nn.BCELoss()
	optimizer = optim.Adam(model_similarity.parameters(), lr=0.001)

	# Training loop
	num_epochs = 10
	for epoch in range(num_epochs):
		model_similarity.train()
		for i,(inputs,labels) in enumerate(pairs_dataloader):

			optimizer.zero_grad()

			output = model_similarity(inputs)
			target = torch.tensor([[labels.item()]]).float()

			loss = criterion(output, target)

			loss.backward()
			optimizer.step()

		print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

	torch.save(model_similarity, 'trained_model_similarity.pth')
	return model_similarity


def top_k_MLP_similarity(top_k,input_embedding,features_dataset,model_similarity, top_1=1):
	print("Refining ranking with MLP similarity measure")

	model_similarity.eval()
	pairs_list_to_rank = []
	input_embedding_1d = torch.tensor(input_embedding.flatten())
	for _,_,i in top_k:
		pair = torch.cat((input_embedding_1d,torch.tensor(features_dataset[i][0])))
		pairs_list_to_rank.append(pair)

	similarity_scores = []
	for i,pair in enumerate(pairs_list_to_rank):
		similarities_scores.append((model_similarity(pair),features_dataset[i][1]))

	sorted_similarities = sorted(similarities_scores, key=lambda x: x[0], reverse=True)
	top = sorted_similarities[:top_1]
	return top


if __name__ == '__main__':
	# Images paths
	training_image_folder_path = 'data/DAM_extraction/'
	testing_image_folder_path = os.listdir('data/test_image_headmind')
	input_image_path = "data/test_image_headmind/image-20210928-103217-38e9a47d.jpg"

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

	# Creating new dataset of training images embeddings, or loading it
	# extracted_features_data_set = create_and_save_dataset(training_image_folder_path,model_embedding,nb_transformation_per_image,train_transform)
	extracted_features_data_set = load_existing_dataset()


	preprocess = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	top_k = 100

	top_100,input_embedding = top_k_cosine_similarity(input_image_path, extracted_features_data_set, model_embedding, preprocess, top_k, verbose = 0)

	print("The label to find is : CAL44551N0")
	print("The top 10 similar images based on cosine similarity are :")
	for t,(similarity,label,i) in enumerate(top_100[:10]):
		print(label)
		image_path = "data/DAM_extraction/"+label+".jpeg"
		img = mpimg.imread(image_path)
		plt.imshow(img)
		plt.title(f"L'image classée en top {t+1} par cosine similarity est {label}")
		plt.show()

	number_training_pairs = 1000
	model_similarity = create_and_train_model(extracted_features_data_set,number_training_pairs,nb_transformation_per_image)
	#model_similarity = torch.load('trained_model_similarity.pth')

	top = top_k_MLP_similarity(top_100,input_embedding,extracted_features_data_set,model_similarity, top_1=10)

	print("The label to find is : CAL44551N0")
	print("The top 10 similar images based on the neural network are :")
	for t,(similarity,label) in enumerate(top):
		print(label)
		image_path = "data/DAM_extraction/"+label+".jpeg"
		img = mpimg.imread(image_path)
		plt.imshow(img)
		plt.title(f"L'image classée en top {t+1} par le réseau de neurone est {label}")
		plt.show()

