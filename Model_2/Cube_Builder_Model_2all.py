path = '/export/bkolstoe/UMD-Scattering-Transform/'

MSI_dir = '/export/bkolstoe/MSI_CIE_TEST'
NIR_dir = '/export/bkolstoe/NIR_TRAIN'
VIS_dir = '/export/bkolstoe/VIS_TRAIN'

###################################################################################

import sys

sys.path.insert(1, path+'Hyper_Skin_Utils/hsiData')


import numpy as np
import h5py
#from Hyper_Skin_Utils.hsiData import HyperSkinData, hsi
import HyperSkinData
import hsi
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import hdf5storage
import torch
import torch.nn as nn

from kymatio.torch import Scattering2D as Scattering2

load_model = False
BATCH_SIZE = 6


checkpoint = torch.load('Model_2.pth')



device ="cuda" if torch.cuda.is_available() else "cpu"

#### load data


scale = tuple(0.5 for i in range(66))

transforms_to_apply1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(scale,scale)
])


train_data = HyperSkinData.Load_msi_visnir(MSI_dir,VIS_dir,NIR_dir, None, transforms_to_apply1)#,transforms_to_apply2)

#### Create Scattering Transform

J2 = 2
L=4
scattering2 = Scattering2(J2,(1024,1024),L=L).to(device)


zdim = 4*25*256*256

num_input_channels = 25
num_hidden_channels = num_input_channels


#### Network Architecture for Inverse Scattering Network

class Generator2(nn.Module):
    def __init__(self, num_input_channels, num_hidden_channels, zdim,  size_firstlayer = 256, num_output_channels=31, filter_size=3):
        super(Generator2, self).__init__()
        self.num_input_channels = num_input_channels
        self.num_hidden_channels = num_hidden_channels
        self.num_output_channels = num_output_channels
        self.filter_size = filter_size
        self.zdim = zdim
        self.size_firstlayer = size_firstlayer
        self.build()

    def build(self):
        padding = (self.filter_size - 1) // 2
        self.main = nn.Sequential(

        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.ReflectionPad2d(padding),
        nn.Conv2d(self.num_input_channels, self.num_hidden_channels, self.filter_size, bias=False),
        nn.BatchNorm2d(self.num_hidden_channels, eps=0.001, momentum=0.9),
        nn.ReLU(inplace=True),

        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.ReflectionPad2d(padding),
        nn.Conv2d(self.num_hidden_channels, self.num_hidden_channels, self.filter_size, bias=False),
        nn.BatchNorm2d(self.num_hidden_channels, eps=0.001, momentum=0.9),
        nn.ReLU(inplace=True),

        nn.ReflectionPad2d(padding),
        nn.Conv2d(self.num_hidden_channels, self.num_output_channels, self.filter_size, bias=False),
        nn.BatchNorm2d(self.num_output_channels, eps=0.001, momentum=0.9),
        nn.Tanh()
                                  )

    def forward(self, input_tensor):
        return self.main(input_tensor)

#### Network Architecture for Matching Network

class MorphNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.c1 = nn.Conv2d(4*25, 60*25, kernel_size=1, bias=True)
        self.c2 = nn.Conv2d(60*25, 31*25, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.c1(x))
        x = self.tanh(self.c2(x))
        return x


#### Network Architecture for Intensity Matching Network

class Intensity_Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.c1 = nn.Linear(62,150)
        self.c2 = nn.Linear(150,62)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.c1(x))
        x = self.tanh(self.c2(x))
        return x


#### load trained model parameters

checkpoint_even = checkpoint['model_match_even'] # model parameters for even channel matching network

matching_model_even = MorphNetwork().to(device)
matching_model_even = nn.DataParallel(matching_model_even).to(device)
matching_model_even.load_state_dict(checkpoint_even['model'])
matching_model_even.eval()

checkpoint_odd = checkpoint['model_match_odd'] # model parameters for odd channel matching network

matching_model_odd = MorphNetwork().to(device)
matching_model_odd = nn.DataParallel(matching_model_odd).to(device)
matching_model_odd.load_state_dict(checkpoint_odd['model'])
matching_model_odd.eval()



checkpoint_even_inv = checkpoint['model_invert_even'] # model parameters for even channel inverse network
checkpoint_odd_inv = checkpoint['model_invert_odd']  # model parameters for odd channel inverse network

model_even = Generator2(31*num_input_channels, 31*num_hidden_channels,zdim).to(device)
model_even = nn.DataParallel(model_even).to(device)
model_even.load_state_dict(checkpoint_even_inv['model'])
model_even.eval()

model_odd = Generator2(31*num_input_channels, 31*num_hidden_channels,zdim).to(device)
model_odd = nn.DataParallel(model_odd).to(device)
model_odd.load_state_dict(checkpoint_odd_inv['model'])
model_odd.eval()


if len(checkpoint.keys()) == 5:
    checkpoint_int = checkpoint['model_intensity_match'] # model parameters for intensity matching network

    model_int = Intensity_Network().to(device)
    model_int = nn.DataParallel(model_int).to(device)
    model_int.load_state_dict(checkpoint_int['model'])
    model_int.eval()    


numbers = {0:'001',1:'002',2:'003',3:'012',4:'016',5:'030',6:'035',7:'036',8:'039'}

#### build all cubes:
for i in range(53):

    number = numbers[i//6]
    expression = 'neutral' if i%6 < 3 else 'smile'
    position = 'front' if (i%6)%3==0 else 'left' if (i%6)%3 == 1 else 'right'
    print(f'Building p{number}_{expression}_{position}')

    #### load MSI image and take its scattering transform
    batch_MSI = train_data[i][0].to(device)
    batch_MSI_scattering = scattering2(batch_MSI).squeeze(1)
    batch_MSI_scattering = torch.reshape(batch_MSI_scattering, (1,-1, batch_MSI_scattering.size()[-2],batch_MSI_scattering.size()[-1]))
    
    #### find the corresponding (even and odd channel) scattering coefficients for the HSI images
    batch_matched_even = matching_model_even(batch_MSI_scattering)
    batch_matched_odd = matching_model_odd(batch_MSI_scattering)
    
    #### invert the matched scattering coefficients to get HSI images (even and odd channels)
    img_pred_even = model_even(batch_matched_even.contiguous()).squeeze().detach()
    img_pred_odd = model_odd(batch_matched_odd.contiguous()).squeeze().detach()
    img_pred_even = np.transpose(img_pred_even.cpu(),(1,2,0)).to(device)
    img_pred_odd = np.transpose(img_pred_odd.cpu(),(1,2,0)).to(device)
    
    #### build cube from even and odd channels
    cube_pred = torch.zeros((1024,1024,62)).to(device)
    cube_pred[:,:,0::2] = img_pred_even
    cube_pred[:,:,1::2] = img_pred_odd
    
    #### find mask of non-skin
    mask = batch_MSI.squeeze()[3,:,:] < np.percentile(batch_MSI.squeeze()[3,:,:].cpu(),70)
    mask_repeat = mask.repeat(62,1,1).cpu()
    
    #### mask out non-skin and remove masked spectra 
    cube_pred_masked = torch.clone(cube_pred*0.5+0.5)
    cube_pred_masked[np.transpose(mask_repeat,(1,2,0)).bool()] = 0
    cube_pred_masked = torch.reshape(cube_pred_masked,(1024**2,62))
    cube_pred_masked_less = cube_pred_masked[cube_pred_masked.abs().sum(dim=1).bool(),:].to(device)
    
    #### match intensity of individual spectra
    pixel_pred = model_int((cube_pred_masked_less-0.5)/0.5)
    
    #### change intensity values of skin spectra
    cube_pred = torch.reshape(cube_pred,(1024**2,62))
    cube_pred[cube_pred_masked.abs().sum(dim=1).bool(),:] = pixel_pred
    cube_pred = torch.reshape(cube_pred,(1024,1024,62))
    img_pred_array = torch.Tensor.numpy(cube_pred.detach().cpu())
    
    #### build final reconstructed cube and return it to [0,1] range.
    cube_all_r = np.zeros((1024,1024,61))
    cube_all_r[:,:,0:30] = img_pred_array[:,:,0:30]
    cube_all_r[:,:,31:] = img_pred_array[:,:,32:]
    cube_all_r[:,:,30] = 0.5*(img_pred_array[:,:,30]+img_pred_array[:,:,31])
    cube_r_shifted = cube_all_r * 0.5+0.5
    
    
    #### save cube
    hdf5storage.savemat(f'p{number}_{expression}_{position}.mat',{'cube':cube_r_shifted},format = '7.3', store_python_metadata=True)
    
    
