import torch
from torchvision import transforms as TF
from torch.utils.data import Dataset


class Mydataset(Dataset):
    def __init__(self, dataset):
        super(Mydataset, self).__init__()
        self.dataset = dataset
        self.image2tensor = TF.PILToTensor()

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        label = self.dataset[idx][1]
        image = self.image2tensor(image)
        # image = TF.Normalize(mean=(), std=())
        label = torch.tensor(label)
        return image.float(), label
    
    def __len__(self):
        return len(self.dataset)