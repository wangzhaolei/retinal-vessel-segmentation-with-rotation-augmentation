import numpy as np
import random
import configparser
import sys
sys.path.insert(0, './lib/')
import os
from help_functions import load_hdf5
from help_functions import visualize
from help_functions import group_images
from pre_processing import my_PreProc


#To select the same images
#random.seed(10)

def get_data_training_add_rotation(patch_height,patch_width,N_subimgs,inside_FOV):
    
    img_pos_30=np.load('DRIVE_datasets_training_testing/rotation_pos_30_imgs.npy')
    img_pos_60 = np.load('DRIVE_datasets_training_testing/rotation_pos_60_imgs.npy')
    img_pos_90 = np.load('DRIVE_datasets_training_testing/rotation_pos_90_imgs.npy')
    img_neg_30 = np.load('DRIVE_datasets_training_testing/rotation_neg_30_imgs.npy')
    img_neg_60 = np.load('DRIVE_datasets_training_testing/rotation_neg_60_imgs.npy')
    img_neg_90 = np.load('DRIVE_datasets_training_testing/rotation_neg_90_imgs.npy')
    

    mask_pos_30 = np.load('DRIVE_datasets_training_testing/rotation_pos_30_masks.npy')
    mask_pos_60 = np.load('DRIVE_datasets_training_testing/rotation_pos_60_masks.npy')
    mask_pos_90 = np.load('DRIVE_datasets_training_testing/rotation_pos_90_masks.npy')
    mask_neg_30 = np.load('DRIVE_datasets_training_testing/rotation_neg_30_masks.npy')
    mask_neg_60 = np.load('DRIVE_datasets_training_testing/rotation_neg_60_masks.npy')
    mask_neg_90 = np.load('DRIVE_datasets_training_testing/rotation_neg_90_masks.npy')
    
    
    train_imgs_original=np.concatenate((img_pos_30,img_pos_60,img_pos_90,img_neg_30,img_neg_60,img_neg_90),axis=0)
    train_masks=np.concatenate((mask_pos_30,mask_pos_60,mask_pos_90,mask_neg_30,mask_neg_60,mask_neg_90),axis=0)
    ##visualize(group_images(train_imgs_original[0:20,:,:,:],5),'imgs_train')#.show()  #check original imgs train

    train_imgs = my_PreProc(train_imgs_original)
    #train_masks = train_masks / 255.
    print('mask_min_max_value:',np.min(train_masks),np.max(train_masks))
 
    train_imgs = train_imgs[:, :, 9:574, :]  # cut bottom and top so now it is 565*565
    train_masks = train_masks[:, :, 9:574, :]  # cut bottom and top so now it is 565*565
    # train_thin_masks=train_thin_masks[:,:,9:574,:]
    data_consistency_check(train_imgs, train_masks)

    # check masks are within 0-1
    assert (int(np.min(train_masks)) == 0 and int(np.max(train_masks)) == 1)

    print("\ntrain images/masks shape:")
    print(train_imgs.shape)
    print("train images range (min-max): " + str(np.min(train_imgs)) + ' - ' + str(np.max(train_imgs)))
    print("train masks are within 0-1\n")

    
    # extract the TRAINING patches from the full images
    patches_imgs_train, patches_masks_train = extract_random(train_imgs, train_masks, patch_height, patch_width,
                                                             N_subimgs, inside_FOV)
    data_consistency_check(patches_imgs_train, patches_masks_train)
    # print("thin mask shape:",patches_thin_mask.shape)

    print("\ntrain PATCHES images/masks shape:")
    print(patches_imgs_train.shape)
    print("train PATCHES images range (min-max): " + str(np.min(patches_imgs_train)) + ' - ' + str(
        np.max(patches_imgs_train)))
    
    return patches_imgs_train, patches_masks_train

def get_data_training_add_rotation_more(imgrot,maskrot,patch_height,patch_width,N_subimgs,angle,inside_FOV):
    train_imgs_original=imgrot
    train_masks=maskrot
    train_imgs_original=train_imgs_original[np.newaxis,...]
    train_masks=train_masks[np.newaxis,...]
    ##visualize(group_images(train_imgs_original[0:20,:,:,:],5),'imgs_train')#.show()  #check original imgs train

    train_imgs = my_PreProc(train_imgs_original)
    #train_masks = train_masks / 255.
    print('mask_min_max_value:',np.min(train_masks),np.max(train_masks))
 
    train_imgs = train_imgs[:, :, 9:574, :]  # cut bottom and top so now it is 565*565
    train_masks = train_masks[:, :, 9:574, :]  # cut bottom and top so now it is 565*565
    # train_thin_masks=train_thin_masks[:,:,9:574,:]
    data_consistency_check(train_imgs, train_masks)

    # check masks are within 0-1
    assert (int(np.min(train_masks)) == 0 and int(np.max(train_masks)) == 1)

    print("\ntrain images/masks shape:")
    print(train_imgs.shape)
    print("train images range (min-max): " + str(np.min(train_imgs)) + ' - ' + str(np.max(train_imgs)))
    print("train masks are within 0-1\n")

    
    # extract the TRAINING patches from the full images
    extract_random(train_imgs, train_masks, patch_height, patch_width,N_subimgs,angle,inside_FOV)
	
	
def get_data_training(DRIVE_train_imgs_original,
                      DRIVE_train_groudTruth,
                      patch_height,
                      patch_width,
                      N_subimgs,
					  angle,
                      inside_FOV):
    train_imgs_original = load_hdf5(DRIVE_train_imgs_original)
    train_masks = load_hdf5(DRIVE_train_groudTruth) #masks always the same

    train_imgs = my_PreProc(train_imgs_original)
    
    train_masks = train_masks/255.
   
    train_imgs = train_imgs[:,:,9:574,:]  #cut bottom and top so now it is 565*565
    train_masks = train_masks[:,:,9:574,:]  #cut bottom and top so now it is 565*565
    #train_imgs=train_imgs[np.newaxis,...]
    #train_masks=train_masks[np.newaxis,...]
    data_consistency_check(train_imgs,train_masks)

    #check masks are within 0-1
    assert(np.min(train_masks)==0 and np.max(train_masks)==1)

    print ("\ntrain images/masks shape:")
    print (train_imgs.shape)
    print ("train images range (min-max): " +str(np.min(train_imgs)) +' - '+str(np.max(train_imgs)))
    print ("train masks are within 0-1\n")
    
    #extract the TRAINING patches from the full images
    extract_random(train_imgs,train_masks,patch_height,patch_width,
				   N_subimgs,angle,inside_FOV)
    #data_consistency_check(patches_imgs_train, patches_masks_train)

    #print ("\ntrain PATCHES images/masks shape:")
    #print (patches_imgs_train.shape)
    #print ("train PATCHES images range (min-max): " +str(np.min(patches_imgs_train)) +' - '+str(np.max(patches_imgs_train)))
    
#return patches_imgs_train, patches_masks_train

#data consinstency check
def data_consistency_check(imgs,masks):
    assert(len(imgs.shape) == len(masks.shape))
    assert(imgs.shape[0] == masks.shape[0])
    assert(imgs.shape[2] == masks.shape[2])
    assert(imgs.shape[3] == masks.shape[3])
    assert(masks.shape[1] == 1)
    #assert(imgs.shape[1] == 1 or imgs.shape[1] == 3)

def extract_random(full_imgs,full_masks, patch_h,patch_w, N_patches,angle, inside=True):
    print(angle)
    '''
    if (N_patches%full_imgs.shape[0] != 0):
        print ("N_patches: plase enter a multiple of 20")
        exit()
    '''
    assert (len(full_imgs.shape) == 4 and len(full_masks.shape) == 4)  #4D arrays
    assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  #check the channel is 1 or 3
    assert (full_masks.shape[1] == 1)   #masks only black and white
    assert (full_imgs.shape[2] == full_masks.shape[2] and full_imgs.shape[3] == full_masks.shape[3])
    #patches = np.empty((N_patches,full_imgs.shape[1],patch_h,patch_w))
    #patches_masks = np.empty((N_patches,full_masks.shape[1],patch_h,patch_w))
    img_h = full_imgs.shape[2]  #height of the full image
    img_w = full_imgs.shape[3] #width of the full image
    # (0,0) in the center of the image
    patch_per_img = int(N_patches / full_imgs.shape[0])  #N_patches equally divided in the full images
    
    print ("patches per full image: " +str(patch_per_img))
    #iter_tot = 0   #iter over the total numbe rof patches (N_patches)
    for i in range(full_imgs.shape[0]):  #loop over the full images
        k = 0
        while k <patch_per_img:
            x_center = random.randint(0+int(patch_w/2),img_w-int(patch_w/2))
            # print "x_center " +str(x_center)
            y_center = random.randint(0+int(patch_h/2),img_h-int(patch_h/2))
            # print "y_center " +str(y_center)
            #check whether the patch is fully contained in the FOV
            if inside == True:
                if is_patch_inside_FOV(x_center,y_center,img_w,img_h,patch_h)==False:
                    continue
            patch = full_imgs[i,:,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]
            patch_mask = full_masks[i,:,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]
            #datarootdir/masks/neg90/6_4.npy
			#datarootdir/angle+'/'+str(i)+'_'+str(j)
            np.save('All_angle_data/imgs/'+angle+'/'+str(i)+'_'+str(k)+'.npy',patch)
            np.save('All_angle_data/masks/'+angle+'/'+str(i)+'_'+str(k)+'.npy',patch_mask)
			#patches[iter_tot] = patch
            #patches_masks[iter_tot] = patch_mask
            #iter_tot += 1   #total
            k += 1  #per full_img
			

#return patches, patches_masks

#check if the patch is fully contained in the FOV
def is_patch_inside_FOV(x,y,img_w,img_h,patch_h):
    x_ = x - int(img_w / 2) # origin (0,0) shifted to image center
    y_ = y - int(img_h / 2)  # origin (0,0) shifted to image center
    R_inside = 270 - int(patch_h * np.sqrt(2.0) / 2.0) #radius is 270 (from DRIVE db docs), minus the patch diagonal (assumed it is a square #this is the limit to contain the full patch in the FOV
    radius = np.sqrt((x_*x_)+(y_*y_))
    if radius < R_inside:
        return True
    else:
        return False
def create_folder(dir_name):
	print("\nCreate directory for the results (if not already existing)")
	if os.path.exists(dir_name):
		print ("Dir already existing")
	else:
		os.system('mkdir -p ' +dir_name)
if __name__=='__main__':
    
    ph=48
    pw=48
    ## 45454 * 11 angles = 50万个patches
    img_data_path='./DRIVE_datasets_training_testing/DRIVE_dataset_imgs_train.hdf5'
    mask_data_path='./DRIVE_datasets_training_testing/DRIVE_dataset_groundTruth_train.hdf5'
    get_data_training(
        DRIVE_train_imgs_original=img_data_path,
        DRIVE_train_groudTruth=mask_data_path,
        patch_height=ph,
        patch_width=pw,
        N_subimgs=45454,
		angle='orig',
        inside_FOV=False)
    
    
    img_pos_30=np.load('DRIVE_datasets_training_testing/rotation_pos_30_imgs.npy')
    mask_pos_30 = np.load('DRIVE_datasets_training_testing/rotation_pos_30_masks.npy')
    get_data_training_add_rotation_more(
		imgrot=img_pos_30[0],
		maskrot=mask_pos_30[0],
		patch_height=ph,
        patch_width=pw,
        N_subimgs=45454,
		angle='pos30',
		inside_FOV=False)
    
    img_neg_30 = np.load('DRIVE_datasets_training_testing/rotation_neg_30_imgs.npy')
    mask_neg_30 = np.load('DRIVE_datasets_training_testing/rotation_neg_30_masks.npy')
    get_data_training_add_rotation_more(
		imgrot=img_neg_30[0],
		maskrot=mask_neg_30[0],
		patch_height=ph,
        patch_width=pw,
        N_subimgs=45454,
		angle='neg30',
		inside_FOV=False)
    
    img_pos_60 = np.load('DRIVE_datasets_training_testing/rotation_pos_60_imgs.npy')
    mask_pos_60 = np.load('DRIVE_datasets_training_testing/rotation_pos_60_masks.npy')
    get_data_training_add_rotation_more(
		imgrot=img_pos_60[0],
		maskrot=mask_pos_60[0],
		patch_height=ph,
        patch_width=pw,
        N_subimgs=45454,
		angle='pos60',
		inside_FOV=False)
    
    img_neg_60 = np.load('DRIVE_datasets_training_testing/rotation_neg_60_imgs.npy')
    mask_neg_60 = np.load('DRIVE_datasets_training_testing/rotation_neg_60_masks.npy')
    get_data_training_add_rotation_more(
		imgrot=img_neg_60[0],
		maskrot=mask_neg_60[0],
		patch_height=ph,
        patch_width=pw,
        N_subimgs=45454,
		angle='neg60',
		inside_FOV=False)
   
    img_pos_90 = np.load('DRIVE_datasets_training_testing/rotation_pos_90_imgs.npy')
    mask_pos_90 = np.load('DRIVE_datasets_training_testing/rotation_pos_90_masks.npy')
    get_data_training_add_rotation_more(
		imgrot=img_pos_90[0],
		maskrot=mask_pos_90[0],
		patch_height=ph,
        patch_width=pw,
        N_subimgs=45454,
		angle='pos90',
		inside_FOV=False)
    
    img_neg_90 = np.load('DRIVE_datasets_training_testing/rotation_neg_90_imgs.npy')
    mask_neg_90 = np.load('DRIVE_datasets_training_testing/rotation_neg_90_masks.npy')
    get_data_training_add_rotation_more(
		imgrot=img_neg_90[0],
		maskrot=mask_neg_90[0],
		patch_height=ph,
        patch_width=pw,
        N_subimgs=45454,
		angle='neg90',
		inside_FOV=False)
    
    img_pos_120 = np.load('DRIVE_datasets_training_testing/rotation_pos_120_imgs.npy')
    mask_pos_120 = np.load('DRIVE_datasets_training_testing/rotation_pos_120_masks.npy')
    get_data_training_add_rotation_more(
		imgrot=img_pos_120[0],
		maskrot=mask_pos_120[0],
		patch_height=ph,
        patch_width=pw,
        N_subimgs=45454,
		angle='pos120',
		inside_FOV=False)
    
    img_neg_120 = np.load('DRIVE_datasets_training_testing/rotation_neg_120_imgs.npy')
    mask_neg_120 = np.load('DRIVE_datasets_training_testing/rotation_neg_120_masks.npy')
    get_data_training_add_rotation_more(
		imgrot=img_neg_120[0],
		maskrot=mask_neg_120[0],
		patch_height=ph,
        patch_width=pw,
        N_subimgs=45454,
		angle='neg120',
		inside_FOV=False)
    
    img_pos_150 = np.load('DRIVE_datasets_training_testing/rotation_pos_150_imgs.npy')
    mask_pos_150 = np.load('DRIVE_datasets_training_testing/rotation_pos_150_masks.npy')
    get_data_training_add_rotation_more(
		imgrot=img_pos_150[0],
		maskrot=mask_pos_150[0],
		patch_height=ph,
        patch_width=pw,
        N_subimgs=45454,
		angle='pos150',
		inside_FOV=False)
    
    img_neg_150 = np.load('DRIVE_datasets_training_testing/rotation_neg_150_imgs.npy')
    mask_neg_150 = np.load('DRIVE_datasets_training_testing/rotation_neg_150_masks.npy')
    get_data_training_add_rotation_more(
		imgrot=img_neg_150[0],
		maskrot=mask_neg_150[0],
		patch_height=ph,
        patch_width=pw,
        N_subimgs=45454,
		angle='neg150',
		inside_FOV=False)
    
    
	


   

