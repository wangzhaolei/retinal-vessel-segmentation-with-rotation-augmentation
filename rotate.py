import matplotlib

#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, './lib/')
from help_functions import *
#from augmentations_update import Augmentation
from augmentation_constant import Augmentation

tran1=Augmentation(rg=-5,zoom_range=(1,1))
tran2=Augmentation(rg=-10,zoom_range=(1,1))
tran3=Augmentation(rg=-15,zoom_range=(1,1))
tran4=Augmentation(rg=-20,zoom_range=(1,1))
tran5=Augmentation(rg=-25,zoom_range=(1,1))
tran6=Augmentation(rg=-30,zoom_range=(1,1))
tran7=Augmentation(rg=-35,zoom_range=(1,1))
tran8=Augmentation(rg=-40,zoom_range=(1,1))
tran9=Augmentation(rg=-45,zoom_range=(1,1))
tran10=Augmentation(rg=-50,zoom_range=(1,1))
tran11=Augmentation(rg=-55,zoom_range=(1,1))
tran12=Augmentation(rg=-60,zoom_range=(1,1))
tran13=Augmentation(rg=-65,zoom_range=(1,1))
tran14=Augmentation(rg=-70,zoom_range=(1,1))
tran15=Augmentation(rg=-75,zoom_range=(1,1))
tran16=Augmentation(rg=-80,zoom_range=(1,1))
tran17=Augmentation(rg=-85,zoom_range=(1,1))
tran18=Augmentation(rg=-90,zoom_range=(1,1))

tran_list=[tran1,tran2,tran3,tran4,tran5,tran6,tran7,tran8,tran9,tran10,
          tran11,tran12,tran13,tran14,tran15,tran16,tran17,tran18]

train_imgs=load_hdf5('DRIVE_datasets_training_testing/DRIVE_dataset_imgs_train.hdf5')
train_masks=load_hdf5('DRIVE_datasets_training_testing/DRIVE_dataset_groundTruth_train.hdf5')
train_masks=train_masks/255
train_masks=train_masks.astype(np.int8)
print(train_imgs.dtype)
print(train_masks.dtype)
train_imgs= train_imgs.transpose(0,2,3,1)
train_masks=train_masks.transpose(0,2,3,1)

m=0
for tran in tran_list:
    m=m+5
    augment_imgs=np.empty(train_imgs.shape)
    augment_masks=np.empty(train_masks.shape)
    
    for i in range(train_imgs.shape[0]):
        img, target = tran(train_imgs[i], train_masks[i])
        
        print(np.bincount(train_masks[i,:,:,0].flatten()))
        print(np.bincount(target[...,0].flatten()))


        #imshow(train_imgs[i]/255,train_masks[i,:,:,0],img/255,target[...,0],title=['orig','gt','aug','augmask'])
        #plt.show()

        augment_imgs[i] = img
        augment_masks[i]=target
    
    augment_imgs=augment_imgs.transpose(0,3,1,2)
    augment_masks=augment_masks.transpose(0,3,1,2)
    print(augment_imgs.shape)
    print(augment_masks.shape)
    print(m)
    
    np.save('DRIVE_datasets_training_testing/rotation_neg_'+str(m)+'_imgs.npy',augment_imgs)
    np.save('DRIVE_datasets_training_testing/rotation_neg_'+str(m)+'_masks.npy',augment_masks)


