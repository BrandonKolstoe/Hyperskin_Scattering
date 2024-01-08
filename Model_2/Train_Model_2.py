path = '/export/bkolstoe/UMD-Scattering-Transform/'

MSI_dir = '/export/bkolstoe/MSI_CIE_TRAIN'
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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import torch.optim as optim
from torch.autograd import Variable

from kymatio.torch import Scattering2D as Scattering2

BATCH_SIZE = 6

device = "cuda" if torch.cuda.is_available() else "cpu"

#### load data


scale = tuple(0.5 for i in range(66))

transforms_to_apply1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(scale,scale)
])

train_data = HyperSkinData.Load_msi_visnir(MSI_dir,VIS_dir,NIR_dir, None, transforms_to_apply1)#,transforms_to_apply2)

train_dataloader_sq = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)

#### Create Scattering Transform

J2 = 2
L=4
scattering2 = Scattering2(J2,(1024,1024),L=L).to(device)


fixed_batch_sq = next(iter(train_dataloader_sq))

fixed_batch_sq = fixed_batch_sq[0][1].to(device)
scattering_fixed_batch_sq = scattering2(fixed_batch_sq).squeeze(1)

zdim = scattering_fixed_batch_sq.shape[0]*scattering_fixed_batch_sq.shape[1]*scattering_fixed_batch_sq.shape[2]*scattering_fixed_batch_sq.shape[3]

num_input_channels = scattering_fixed_batch_sq.shape[1]
num_hidden_channels = num_input_channels


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


#### Initialize even channel matching model for training

torch.manual_seed(20230418)

matching_model_even = MorphNetwork().to(device)
matching_model_even = nn.DataParallel(matching_model_even).to(device)

optimiser_even = optim.Adam(matching_model_even.parameters())

matching_model_even.train()


loss_fn = torch.nn.MSELoss()
epochs = 100


# Create training loop
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-------")

    train_loss = 0

    for ct, current_batch in enumerate(train_dataloader_sq):
        matching_model_even.train()

        batch_MSI = Variable(current_batch[0]).to(device)
        batch_HSI_all = Variable(current_batch[1]).to(device)
        batch_HSI = batch_HSI_all[:,0::2,:,:].contiguous()

        batch_MSI_scattering = scattering2(batch_MSI).squeeze(1)
        batch_VIS_scattering = scattering2(batch_HSI).squeeze(1)

        batch_MSI_scattering = torch.reshape(batch_MSI_scattering, (BATCH_SIZE,-1, batch_MSI_scattering.size()[-2],batch_MSI_scattering.size()[-1]))
        batch_VIS_scattering = torch.reshape(batch_VIS_scattering, (BATCH_SIZE,-1, batch_VIS_scattering.size()[-2],batch_VIS_scattering.size()[-1]))

        batch_matched_VIS_scattering = matching_model_even(batch_MSI_scattering)

        loss = loss_fn(batch_matched_VIS_scattering,batch_VIS_scattering)

        # Optimiser zero grad
        optimiser_even.zero_grad()

        # Loss backward
        loss.backward()

        # Optimiser step
        optimiser_even.step()
        train_loss += loss.item()

        print(ct)


    # Average loss per batch per epoch
    train_loss /= len(train_dataloader_sq)

    ## Print out what's happening
    print(f"Train loss: {train_loss:.6f} \n")

matching_model_even.eval()
checkpoint_even = {'model':matching_model_even.state_dict(),'optimiser':optimiser_even.state_dict()}


#### Initialize odd channel matching model for training

torch.manual_seed(20230418)

matching_model_odd = MorphNetwork().to(device)
matching_model_odd = nn.DataParallel(matching_model_odd).to(device)

optimiser_odd = optim.Adam(matching_model_odd.parameters())

matching_model_odd.train()


loss_fn = torch.nn.MSELoss()
epochs = 100




# Create training loop
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-------")

    train_loss = 0

    for ct, current_batch in enumerate(train_dataloader_sq):
        matching_model_odd.train()

        batch_MSI = Variable(current_batch[0]).to(device)
        batch_HSI_all = Variable(current_batch[1]).to(device)
        batch_HSI = batch_HSI_all[:,1::2,:,:].contiguous()

        batch_MSI_scattering = scattering2(batch_MSI).squeeze(1)
        batch_VIS_scattering = scattering2(batch_HSI).squeeze(1)

        batch_MSI_scattering = torch.reshape(batch_MSI_scattering, (BATCH_SIZE,-1, batch_MSI_scattering.size()[-2],batch_MSI_scattering.size()[-1]))
        batch_VIS_scattering = torch.reshape(batch_VIS_scattering, (BATCH_SIZE,-1, batch_VIS_scattering.size()[-2],batch_VIS_scattering.size()[-1]))

        batch_matched_VIS_scattering = matching_model_odd(batch_MSI_scattering)

        loss = loss_fn(batch_matched_VIS_scattering,batch_VIS_scattering)

        # Optimiser zero grad
        optimiser_odd.zero_grad()

        # Loss backward
        loss.backward()

        # Optimiser step
        optimiser_odd.step()
        train_loss += loss.item()

        print(ct)


    # Average loss per batch per epoch
    train_loss /= len(train_dataloader_sq)

    ## Print out what's happening
    print(f"Train loss: {train_loss:.6f} \n")

matching_model_odd.eval()
checkpoint_odd = {'model':matching_model_odd.state_dict(),'optimiser':optimiser_odd.state_dict()}




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
        #nn.ConstantPad2d(padding,0),
        nn.Conv2d(self.num_hidden_channels, self.num_output_channels, self.filter_size, bias=False),
        nn.BatchNorm2d(self.num_output_channels, eps=0.001, momentum=0.9),
        nn.Tanh()
                                  )

    def forward(self, input_tensor):
        return self.main(input_tensor)


#### Initialize even channel inverse model for training

torch.manual_seed(20230418)

model_even_inv = Generator2(31*num_input_channels, 31*num_hidden_channels,zdim).to(device)

model_even_inv = nn.DataParallel(model_even_inv).to(device)


optimiser_even_inv = optim.Adam(model_even_inv.parameters())

model_even_inv.train()


loss_fn = torch.nn.L1Loss()
epochs = 150


# Create training loop
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-------")

    train_loss = 0

    for ct, current_batch in enumerate(train_dataloader_sq):
        model_even_inv.train()

        # Forward pass
        batch_images = Variable(current_batch[1]).to(device)
        batch_images = batch_images[:,0::2,:,:].contiguous()
        batch_scattering = scattering2(batch_images).squeeze(1)
        batch_scattering = torch.reshape(batch_scattering, (BATCH_SIZE, -1, batch_scattering.size()[-2], batch_scattering.size()[-1]))
        batch_inverse_scattering = model_even_inv(batch_scattering)

        # Calculate loss
        loss = loss_fn(batch_inverse_scattering, batch_images)

        # Optimiser zero grad
        optimiser_even_inv.zero_grad()

        # Loss backward
        loss.backward()

        # Optimiser step
        optimiser_even_inv.step()
        train_loss += loss.item()

        print(ct)

    # Average loss per batch per epoch
    train_loss /= len(train_dataloader_sq)

    ## Print out what's happening
    print(f"Train loss: {train_loss:.6f} \n")



model_even_inv.eval()
checkpoint_even_inv = {'model':model_even_inv.state_dict(),'optimiser':optimiser_even_inv.state_dict()}


#### Initialize even channel inverse model for training

torch.manual_seed(20230418)

model_odd_inv = Generator2(31*num_input_channels, 31*num_hidden_channels,zdim).to(device)

model_odd_inv = nn.DataParallel(model_odd_inv).to(device)


optimiser_odd_inv = optim.Adam(model_odd_inv.parameters())

model_odd_inv.train()


loss_fn = torch.nn.L1Loss()
epochs = 150



# Create training loop
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-------")

    train_loss = 0

    for ct, current_batch in enumerate(train_dataloader_sq):
        model_odd_inv.train()

        # Forward pass
        batch_images = Variable(current_batch[1]).to(device)
        batch_images = batch_images[:,1::2,:,:].contiguous()
        batch_scattering = scattering2(batch_images).squeeze(1)
        batch_scattering = torch.reshape(batch_scattering, (BATCH_SIZE, -1, batch_scattering.size()[-2], batch_scattering.size()[-1]))
        batch_inverse_scattering = model_odd_inv(batch_scattering)

        # Calculate loss
        loss = loss_fn(batch_inverse_scattering, batch_images)

        # Optimiser zero grad
        optimiser_odd_inv.zero_grad()

        # Loss backward
        loss.backward()

        # Optimiser step
        optimiser_odd_inv.step()
        train_loss += loss.item()

        print(ct)

    # Average loss per batch per epoch
    train_loss /= len(train_dataloader_sq)

    ## Print out what's happening
    print(f"Train loss: {train_loss:.6f} \n")

model_odd_inv.eval()
checkpoint_odd_inv = {'model':model_odd_inv.state_dict(),'optimiser':optimiser_odd_inv.state_dict()}



#################################################

#### initialize data again but with smaller batch size

load_model = False
BATCH_SIZE = 1

device ="cuda" if torch.cuda.is_available() else "cpu"
print(device)

MSI_dir = '/export/bkolstoe/MSI_CIE_TRAIN'
NIR_dir = '/export/bkolstoe/NIR_TRAIN'
VIS_dir = '/export/bkolstoe/VIS_TRAIN'

scale = tuple(0.5 for i in range(66))

transforms_to_apply1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(scale,scale)
])


train_data = HyperSkinData.Load_msi_visnir(MSI_dir,VIS_dir,NIR_dir, None, transforms_to_apply1)#,transforms_to_apply2)

train_dataloader_sq = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)

fixed_batch_sq = next(iter(train_dataloader_sq))

fixed_batch_sq = fixed_batch_sq[0][0].to(device)
scattering_fixed_batch_sq = scattering2(fixed_batch_sq).squeeze(1)

zdim = scattering_fixed_batch_sq.shape[0]*scattering_fixed_batch_sq.shape[1]*scattering_fixed_batch_sq.shape[2]*scattering_fixed_batch_sq.shape[3]

num_input_channels = scattering_fixed_batch_sq.shape[1]
num_hidden_channels = num_input_channels




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


#### Initialize Intensity matching model for training

model_int = Intensity_Network().to(device)
model_int = nn.DataParallel(model_int).to(device)

optimiser_int = optim.Adam(model_int.parameters())

model_int.train()

loss_fn = torch.nn.MSELoss()

epochs = 30


# Create training loop
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-------")

    train_loss = 0

    for ct, current_batch in enumerate(train_dataloader_sq):


        #### load MSI and HSI image and take MSI scattering transform
        batch_MSI = Variable(current_batch[0]).to(device)
        batch_HSI_all = Variable(current_batch[1]).squeeze().to(device)
        batch_MSI_scattering = scattering2(batch_MSI).squeeze(1)
        batch_MSI_scattering = torch.reshape(batch_MSI_scattering, (BATCH_SIZE,-1, batch_MSI_scattering.size()[-2],batch_MSI_scattering.size()[-1]))

        #### use the models trained above to match scattering coefficinets and invert
        batch_matched_even = matching_model_even(batch_MSI_scattering)
        batch_matched_odd = matching_model_odd(batch_MSI_scattering)
        img_pred_even = model_even_inv(batch_matched_even.contiguous()).squeeze().detach()
        img_pred_odd = model_odd_inv(batch_matched_odd.contiguous()).squeeze().detach()

        #### build hyperspectral cube
        img_pred_even = np.transpose(img_pred_even.cpu(),(1,2,0)).to(device)
        img_pred_odd = np.transpose(img_pred_odd.cpu(),(1,2,0)).to(device)
        cube_pred = torch.zeros((1024,1024,62)).to(device)
        cube_pred[:,:,0::2] = img_pred_even*0.5+0.5
        cube_pred[:,:,1::2] = img_pred_odd*0.5+0.5

        #### mask out non-skin
        mask = batch_MSI.squeeze()[3,:,:] < np.percentile(batch_MSI.squeeze()[3,:,:].cpu(),70)
        mask_repeat = mask.repeat(62,1,1).cpu()
        cube_pred[np.transpose(mask_repeat,(1,2,0)).bool()] = 0
        cube_pred = torch.reshape(cube_pred,(1024**2,62))

        batch_HSI_all = batch_HSI_all*0.5 + 0.5
        batch_HSI_all[mask_repeat] = 0
        batch_HSI_all = torch.reshape(np.transpose(batch_HSI_all.cpu(),(1,2,0)),(1024**2,62)).to(device)

        cube_pred = cube_pred[cube_pred.abs().sum(dim=1).bool(),:]
        batch_HSI_all = batch_HSI_all[batch_HSI_all.abs().sum(dim=1).bool(),:]

        #### make predicted spectra values
        cube_pred = Variable((cube_pred-0.5)/0.5,requires_grad = True).to(device)
        pixel_pred = model_int(cube_pred)


        loss = loss_fn(pixel_pred,(batch_HSI_all-0.5)/0.5)

        # Optimiser zero grad
        optimiser_int.zero_grad()

        # Loss backward
        loss.backward()

        # Optimiser step
        optimiser_int.step()
        train_loss += loss.item()

        if ct % 10 == 0:
            print(ct)

    # Average loss per batch per epoch
    train_loss /= len(train_dataloader_sq)

    ## Print out what's happening
    print(f"Train loss: {train_loss:.6f} \n")


checkpoint_int = {'model':model_int.state_dict(),'optimiser':optimiser_int.state_dict()}

checkpoint = {'model_match_even':checkpoint_even,'model_match_odd':checkpoint_odd, 'model_invert_even':checkpoint_even_inv,'model_invert_odd':checkpoint_odd_inv,'model_intensity_match':checkpoint_int}

torch.save(checkpoint,'Model_2.pth')
