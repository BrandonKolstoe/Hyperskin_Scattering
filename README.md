# UMD-Scattering-Transform

Code for training and testing our 3 submitted models. 

## Installation
Use pip to install the requirements file:
```bash
pip install -r requirements.txt
```
Python>=3.11.5 is needed. Additionally, as I am using Github's Large File Storage for my saved models, git-lfs needs to be installed before the repository is downloaded. This can be done by either going to [here](https://git-lfs.com/) and downloading the package, or using
```bash
conda install -c conda-forge git-lfs
```

## How to Reconstruct Cubes

For each model, two different files are given to reconstruct the cubes. Cube_Builder_Model_i (where i is the model number) is used to recosntruct individual cubes. First, specify the location of the repository:
```python
path = '/export/bkolstoe/UMD-Scattering-Transform/'
```
Then, specify what image (0 through 53) in the test set should be reconstructed: 
```python
i = 0 #### select which Test Image you want to reconstruct (0<= i <= 53)
```
Finally, specify the location of the image folders:
```python
MSI_dir = '/export/bkolstoe/MSI_CIE_TEST'
NIR_dir = '/export/bkolstoe/NIR_TRAIN'
VIS_dir = '/export/bkolstoe/VIS_TRAIN'
```

To reconstruct all cubes in the testing set, run Cube_Builder_Model_iall (where i is the model number). Like above, you will need to specify the location of the repository and the location of the image folders.

## Training the models

To train the models, run Train_Model_i (where i is the model number). Like above, you will need to specify the location of the repository and the location of the image folders.  

Please let me know if you have any questions!
