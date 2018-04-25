from siamese_network import SiameseNetwork, ContrastiveLoss
from dataset_generator import SiameseNetworkDataset, Config
from torch.autograd import Variable
from utils import imshow, show_plot, save_checkpoint

from torchvision import models
import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
import argparse
from PIL import Image
import PIL.ImageOps
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-img', type=str, default='test.jpg')
	# parser.add_argument('-ques', type=str, default='What vechile is in the picture?')
	args = parser.parse_args()
	img = args.img
	folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
	siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
										transform=transforms.Compose([transforms.Resize((100,100)),
																	  transforms.ToTensor()
																	  ])
									   ,should_invert=False)


	img = Image.open(img)
	img = img.convert("L")
	transform=transforms.Compose([transforms.Resize((100,100)),
								  transforms.ToTensor()
								  ])
	img = transform(img)
	# Add a dimension to image to make padding possible.
	img = img[None,:,:,:]

	test_dataloader = DataLoader(siamese_dataset,num_workers=3,batch_size=1)
	dataiter = iter(test_dataloader)
	net = SiameseNetwork()
	net.load_state_dict(torch.load("trained_weights.pt"))

	for i in range(4):
		   _,x1,label2 = next(dataiter)
		   concatenated = torch.cat((img,x1),0)

		   output1,output2 = net(Variable(img),Variable(x1))
		   euclidean_distance = F.pairwise_distance(output1, output2)
		   imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: \
					{:.2f}'.format(euclidean_distance.data.numpy()[0][0]))



if __name__ == "__main__":
	main()
