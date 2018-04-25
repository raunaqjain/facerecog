import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torchvision
from utils import imshow

from dataset_generator import SiameseNetworkDataset, Config


def main():
    folder_dataset = dset.ImageFolder(root=Config.training_dir)
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                            transform=transforms.Compose([transforms.Resize((100,100)),
                                                                          transforms.ToTensor()
                                                                          ])
                                           ,should_invert=False)

    vis_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=8)

    dataiter = iter(vis_dataloader)

    example_batch = next(dataiter)
    concatenated = torch.cat((example_batch[0],example_batch[1]),0)
    print(example_batch[2].numpy())
    imshow(torchvision.utils.make_grid(concatenated))
    print ("Checking of visualization done.")

if __name__ == "__main__":
    main()
