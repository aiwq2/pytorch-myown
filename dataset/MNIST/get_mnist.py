import torchvision
import os

file_path_abs=os.path.abspath(__file__)

save_data_path=os.path.join(os.path.dirname(file_path_abs),'data')



MNIST_Train_data = torchvision.datasets.MNIST(save_data_path, train=True, download=True,
                                       transform=torchvision.transforms.ToTensor())

MNIST_Test_data = torchvision.datasets.MNIST(save_data_path, train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())