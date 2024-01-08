print("Starting to import useful modules")

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

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

torch.manual_seed(42)
random.seed(42)

class EmbeddingsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
       return self.data[idx]


def create_embeddings_dataset(folder_path, model_embedding, transform=None, n_trans_per_image=1):
	print("Creating dataset of embeddings")
	image_paths = [os.path.join(folder_path, img_name) for img_name in os.listdir(folder_path)]
	data = []
	model_embedding.eval()

	start_time = time.time()
	total_images = len(image_paths)
	progress_step = total_images // 10  # Print progress every 10%

	with torch.no_grad(), tqdm(total=total_images, desc="Processing Images") as pbar:
		for idx, image_path in enumerate(image_paths, 1):
			label = os.path.splitext(os.path.basename(image_path))[0]
			img = Image.open(image_path).convert("RGB")

			for i in range(n_trans_per_image):
				if transform:
					img_transformed = transform(img)
				embedding = model_embedding(img_transformed.unsqueeze(0))
				embedding = embedding.cpu().numpy().flatten()
				data.append((embedding, label))

			pbar.update(1)  # Update the progress bar

	print("Creation of embeddings dataset done")
	return data

def input_preprocessing(image):
	image_backgroundless = remove(image)
	new_size=(256, 256)	
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


# Fonction qui renvoie un top_k d'images dans la base de donnée les plus proches de l'image donnée en entrée
def top_k_cosine_similarity(input_image_path, embeddings_dataset, model_embedding, preprocess, top_k=10,verbose = 0):
	print("Looking for similar images based on a cosine similarity measure")
	model_embedding.eval()

	input_image = Image.open(input_image_path)
	image = input_preprocessing(input_image)

	if verbose == 1:
		plt.imshow(input_image)
		plt.show()

		plt.imshow(image)
		plt.show()

	input_tensor = preprocess(image)
	input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
	with torch.no_grad():
		input_embedding = np.array(model_embedding(input_batch).cpu().numpy().flatten())
	input_embedding = np.reshape(input_embedding, (1, -1))

	similarities = []
	for i,(embedding, label) in enumerate(embeddings_dataset):
		similarity = cosine_similarity(input_embedding, embedding.reshape(1, -1))
		similarities.append((similarity,label,i))

	sorted_similarities = sorted(similarities, key=lambda x: x[0], reverse=True)
	top_k_images = sorted_similarities[:top_k]
	return top_k_images,input_embedding

def create_pairs(embeddings_dataset,number_pairs,n_trans_im):
	
	possible_index_pairs = []
	for i in range(len(embeddings_dataset)):
		for j in range(i+1,len(embeddings_dataset)):
			possible_index_pairs.append((i,j))

	pairs_list = []
	
	# On commence par ajouter la moitié de pairs prises entièrement au hasard, la plupart d'images différentes donc
	index_list = random.sample(possible_index_pairs,number_pairs//2)
	for (i,j) in index_list:
		if random.random()<0.5:
			pair = torch.cat((torch.tensor(embeddings_dataset[i][0]),torch.tensor(embeddings_dataset[j][0]))),float(embeddings_dataset[i][1]==embeddings_dataset[j][1])
		else : 
			pair = torch.cat((torch.tensor(embeddings_dataset[j][0]),torch.tensor(embeddings_dataset[i][0]))),float(embeddings_dataset[j][1]==embeddings_dataset[i][1])
		pairs_list.append(pair)

	# Puis on ajoute des pairs d'images identiques qui ont subi des transformations différentes
	index_list = random.sample(range(len(embeddings_dataset)),(number_pairs+1)//2)
	for i in index_list:
		rang = i//n_trans_im
		j = rang * n_trans_im + ((random.randint(1,n_trans_im-1) + (i % n_trans_im)) % n_trans_im)
		pair = torch.cat((torch.tensor(embeddings_dataset[i][0]),torch.tensor(embeddings_dataset[j][0]))),1
		pairs_list.append(pair)

	return pairs_list


class TrainingPairsDataset(Dataset):
	def __init__(self, embeddings_dataset,number_pairs,n_trans_im):
		self.embeddings_dataset = embeddings_dataset
		self.number_pairs = number_pairs
		self.pairs_list = create_pairs(self.embeddings_dataset,number_pairs,n_trans_im)

	def __len__(self):
		return self.number_pairs

	def __getitem__(self, idx):
		return self.pairs_list[idx]

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

def create_and_train_model(training_pairs_dataset,number_pairs,n_trans_im):
	training_pairs_dataloader = DataLoader(training_pairs_dataset, batch_size=1, shuffle=True)
	
	# On vérifie la proportion de pairs correspondant au même objet dans l'entrainement
	total_vectors = 0
	total_identical = 0
	for (vector,label) in training_pairs_dataset:
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
		for i,(inputs,labels) in enumerate(training_pairs_dataloader):

			optimizer.zero_grad()

			output = model_similarity(inputs)
			target = torch.tensor([[labels.item()]]).float()

			loss = criterion(output, target)

			loss.backward()
			optimizer.step()

		print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

	return model_similarity


def top_k_MLP_similarity(top_k,input_embedding,features_dataset,model_similarity, top_1=1):
	print("Refining ranking with MLP similarity measure")

	model_similarity.eval()
	pairs_list_to_rank = []
	input_embedding_1d = torch.tensor(input_embedding.flatten())
	for _,_,i in top_k:
		pair = torch.cat((input_embedding_1d,torch.tensor(features_dataset[i][0])))
		pairs_list_to_rank.append(pair)

	similarities_scores = []
	for i,pair in enumerate(pairs_list_to_rank):
		similarities_scores.append((model_similarity(pair),features_dataset[i][1]))

	sorted_similarities = sorted(similarities_scores, key=lambda x: x[0], reverse=True)
	top = sorted_similarities[:top_1]
	return top


if __name__ == '__main__':
	training_image_folder_path = 'data/DAM_extraction/'

	model_embedding = models.resnet18(pretrained=True)
	model_embedding = torch.nn.Sequential(*(list(model_embedding.children())[:-1])) # On supprime la dernière couche de classification

	data_augmentation_transform = transforms.Compose([
			transforms.RandomResizedCrop(256, scale=(0.7, 1.2)),
			transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
			transforms.RandomHorizontalFlip(),
			transforms.RandomVerticalFlip(),
			transforms.RandomRotation(22),
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])

	augmentations_per_image=16

	embeddings_dataset = EmbeddingsDataset(create_embeddings_dataset(training_image_folder_path,model_embedding,transform = data_augmentation_transform,n_trans_per_image = augmentations_per_image))
	torch.save(embeddings_dataset,'./embeddings_dataset.pt')
	# embeddings_dataset = torch.load('embeddings_dataset.pt')	

	input_image_path = "data/test_image_headmind/image-20210928-103217-38e9a47d.jpg"

	preprocess = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])

	k = 100

	top100,input_embedding = top_k_cosine_similarity(input_image_path, embeddings_dataset, model_embedding, preprocess, top_k=k,verbose = 1)

	print("The label to find is : CAL44551N0")
	print("The top 10 similar images based on cosine similarity are :")
	for t,(similarity,label,i) in enumerate(top100[:10]):
		print(label)
		image_path = "data/DAM_extraction/"+label+".jpeg"
		img = mpimg.imread(image_path)
		plt.imshow(img)
		plt.title(f"L'image classée en top {t+1} par cosine similarity est {label}")
		plt.show()

	number_pairs = 1000

	training_pairs_dataset = TrainingPairsDataset(embeddings_dataset,number_pairs,augmentations_per_image)
	torch.save(training_pairs_dataset,'./training_pairs_dataset.pt')
	# training_pairs_dataset = torch.load('training_pairs_dataset.pt')	


	model_similarity = create_and_train_model(training_pairs_dataset,number_pairs,augmentations_per_image)
	torch.save(model_similarity, 'trained_model_similarity.pth')
	# model_similarity = torch.load('trained_model_similarity.pth')


	top = top_k_MLP_similarity(top100,input_embedding,embeddings_dataset,model_similarity, top_1=10)

	print("The label to find is : CAL44551N0")
	print("The top 10 similar images based on the neural network are :")
	for t,(similarity,label) in enumerate(top):
		print(label)
		image_path = "data/DAM_extraction/"+label+".jpeg"
		img = mpimg.imread(image_path)
		plt.imshow(img)
		plt.title(f"L'image classée en top {t+1} par le réseau de neurone est {label}")
		plt.show()

