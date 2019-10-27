import numpy as np
import random
import configparser

from help_functions import load_hdf5
from help_functions import visualize
from help_functions import group_images
from skimage.morphology import opening,square
from pre_processing import my_PreProc


#To select the same images
random.seed(10)
def get_data_testing_big_input(DRIVE_test_imgs_original, DRIVE_test_groudTruth, patch_height, patch_width):
    train_imgs_original = load_hdf5(DRIVE_test_imgs_original)
    train_masks = load_hdf5(DRIVE_test_groudTruth)  # masks always the same

    train_imgs = my_PreProc(train_imgs_original)
    train_masks = train_masks / 255.

    train_imgs = train_imgs[:, :, 9:574, :]  # cut bottom and top so now it is 565*565
    train_masks = train_masks[:, :, 9:574, :]  # cut bottom and top so now it is 565*565

    train_imgs = np.pad(train_imgs, ((0, 0), (0, 0), (21, 21), (21, 21)), 'reflect')
    train_masks = np.pad(train_masks, ((0, 0), (0, 0), (21, 21), (21, 21)), 'reflect')

    data_consistency_check(train_imgs, train_masks)

    # check masks are within 0-1
    assert (np.min(train_masks) == 0 and np.max(train_masks) == 1)

    print("\ntest images/masks shape:")
    print(train_imgs.shape)
    print("test images range (min-max): " + str(np.min(train_imgs)) + ' - ' + str(np.max(train_imgs)))
    print("test masks are within 0-1\n")

    # extract the TRAINING patches from the full images
    patches_imgs_train, patches_masks_train = extract_random_big_input(train_imgs, train_masks, patch_height,
                                                                       patch_width,
                                                                       N_patches=10000)
    # data_consistency_check(patches_imgs_train, patches_masks_train)

    print("\ntrain PATCHES images shape:")
    print(patches_imgs_train.shape)
    print("train PATCHES images range (min-max): " + str(np.min(patches_imgs_train)) + ' - ' + str(
        np.max(patches_imgs_train)))
    print("\ntrain PATCHES masks shape:")
    print(patches_masks_train.shape)

    return patches_imgs_train, patches_masks_train




def get_data_training_big_input(DRIVE_train_imgs_original,DRIVE_train_groudTruth,patch_height,patch_width,N_subimgs,inside_FOV):
    train_imgs_original = load_hdf5(DRIVE_train_imgs_original)
    train_masks = load_hdf5(DRIVE_train_groudTruth)  # masks always the same


    train_imgs = my_PreProc(train_imgs_original)
    train_masks = train_masks / 255.

    train_imgs = train_imgs[:, :, 9:574, :]  # cut bottom and top so now it is 565*565
    train_masks = train_masks[:, :, 9:574, :]  # cut bottom and top so now it is 565*565

    train_imgs = np.pad(train_imgs, ((0, 0), (0, 0), (21, 21), (21, 21)), 'reflect')
    train_masks = np.pad(train_masks, ((0, 0), (0, 0), (21, 21), (21, 21)), 'reflect')

    data_consistency_check(train_imgs, train_masks)

    # check masks are within 0-1
    assert (np.min(train_masks) == 0 and np.max(train_masks) == 1)

    print("\ntrain images/masks shape:")
    print(train_imgs.shape)
    print("train images range (min-max): " + str(np.min(train_imgs)) + ' - ' + str(np.max(train_imgs)))
    print("train masks are within 0-1\n")

    # extract the TRAINING patches from the full images
    patches_imgs_train, patches_masks_train = extract_random_big_input(train_imgs, train_masks, patch_height, patch_width,
                                                             N_subimgs)
    #data_consistency_check(patches_imgs_train, patches_masks_train)

    print("\ntrain PATCHES images shape:")
    print(patches_imgs_train.shape)
    print("train PATCHES images range (min-max): " + str(np.min(patches_imgs_train)) + ' - ' + str(
        np.max(patches_imgs_train)))
    print("\ntrain PATCHES masks shape:")
    print(patches_masks_train.shape)

    return patches_imgs_train, patches_masks_train


def extract_random_big_input(full_imgs, full_masks, patch_h, patch_w, N_patches):
    if (N_patches % full_imgs.shape[0] != 0):
        print("N_patches: plase enter a multiple of 20")
        exit()
    assert (len(full_imgs.shape) == 4 and len(full_masks.shape) == 4)  # 4D arrays
    # assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  #check the channel is 1 or 3
    assert (full_masks.shape[1] == 1)  # masks only black and white
    assert (full_imgs.shape[2] == full_masks.shape[2] and full_imgs.shape[3] == full_masks.shape[3])
    patches = np.empty((N_patches, full_imgs.shape[1], 90, 90))
    patches_masks = np.empty((N_patches, full_masks.shape[1], patch_h, patch_w))
    # patches_thin_mask=np.empty((N_patches,full_masks.shape[1],patch_h,patch_w))
    img_h = full_imgs.shape[2]  # height of the full image
    img_w = full_imgs.shape[3]  # width of the full image
    # (0,0) in the center of the image
    patch_per_img = int(N_patches / full_imgs.shape[0])  # N_patches equally divided in the full images

    print("patches per full image: " + str(patch_per_img))
    iter_tot = 0  # iter over the total numbe rof patches (N_patches)
    for i in range(full_imgs.shape[0]):  # loop over the full images
        k = 0
        while k < patch_per_img:
            x_center = random.randint(0 + int(90 / 2), img_w - int(90 / 2))
            # print "x_center " +str(x_center)
            y_center = random.randint(0 + int(90 / 2), img_h - int(90 / 2))
            # print "y_center " +str(y_center)
            # check whether the patch is fully contained in the FOV
            patch = full_imgs[i, :, y_center - int(90 / 2):y_center + int(90 / 2),
                    x_center - int(90 / 2):x_center + int(90 / 2)]
            patch_mask = full_masks[i, :, y_center - int(patch_h / 2):y_center + int(patch_h / 2),
                         x_center - int(patch_w / 2):x_center + int(patch_w / 2)]
            patches[iter_tot] = patch
            patches_masks[iter_tot] = patch_mask

            iter_tot += 1  # total
            k += 1  # per full_img
    return patches, patches_masks  # ,patches_thin_mask
# Load the original data and return the extracted patches for training/testing
# N_subimgs,
# inside_FOV
def get_data_training_extract_ordered_overlap(DRIVE_train_imgs_original,
                      DRIVE_train_groudTruth,
                      patch_height,
                      patch_width,
                      stride_height,
                      stride_width):
    train_imgs_original = DRIVE_train_imgs_original
    train_masks = DRIVE_train_groudTruth
    print('train ////////  all image shape:',train_imgs_original.shape)
    print('train ////////  all image shape:',train_masks.shape)
    train_imgs = my_PreProc(train_imgs_original)
    train_masks = train_masks / 255.
    train_imgs = train_imgs[:, :, 9:574, :]  # cut bottom and top so now it is 565*565
    train_masks = train_masks[:, :, 9:574, :]  # cut bottom and top so now it is 565*565

    train_imgs = paint_border_overlap(train_imgs, patch_height, patch_width,
                                      stride_height, stride_width)
    train_masks = paint_border_overlap(train_masks, patch_height, patch_width,
                                       stride_height, stride_width)

    data_consistency_check(train_imgs, train_masks)

    # check masks are within 0-1
    assert (np.min(train_masks) == 0 and np.max(train_masks) == 1)

    print("\ntrain images/masks shape:")
    print(train_imgs.shape)
    print("train images range (min-max): " + str(np.min(train_imgs)) + ' - ' + str(np.max(train_imgs)))
    print("train masks are within 0-1\n")

    # extract the TRAINING patches from the full images
    patches_imgs_train = extract_ordered_overlap(train_imgs,patch_height,patch_width,
                                                 stride_height,stride_width)
    patches_masks_train = extract_ordered_overlap(train_masks, patch_height, patch_width,
                                                 stride_height, stride_width)
    data_consistency_check(patches_imgs_train, patches_masks_train)

    print("\ntrain PATCHES images/masks shape:")
    print(patches_imgs_train.shape)
    print("train PATCHES images range (min-max): " + str(np.min(patches_imgs_train)) + ' - ' + str(
        np.max(patches_imgs_train)))

    return patches_imgs_train, patches_masks_train,patches_imgs_train.shape[0]
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
def get_data_training_HRF(DRIVE_train_imgs_original,
                      DRIVE_train_groudTruth,
                      patch_height,
                      patch_width,
                      N_subimgs,
                      inside_FOV):
    train_imgs_original = load_hdf5(DRIVE_train_imgs_original)
    train_masks = load_hdf5(DRIVE_train_groudTruth) #masks always the same

    train_imgs = my_PreProc(train_imgs_original)
    train_masks = train_masks/255.
    #train_thin_masks=train_thin_masks[:,:,9:574,:]
    data_consistency_check(train_imgs,train_masks)

    #check masks are within 0-1
    assert(np.min(train_masks)==0 and np.max(train_masks)==1)

    print ("\ntrain images/masks shape:")
    print (train_imgs.shape)
    print ("train images range (min-max): " +str(np.min(train_imgs)) +' - '+str(np.max(train_imgs)))
    print ("train masks are within 0-1\n")

    #extract the TRAINING patches from the full images
    patches_imgs_train, patches_masks_train= extract_random(train_imgs,train_masks,patch_height,patch_width,N_subimgs,inside_FOV)
    data_consistency_check(patches_imgs_train, patches_masks_train)

    print ("\ntrain PATCHES images/masks shape:")
    print (patches_imgs_train.shape)
    print ("train PATCHES images range (min-max): " +str(np.min(patches_imgs_train)) +' - '+str(np.max(patches_imgs_train)))

    return patches_imgs_train, patches_masks_train
#Load the original data and return the extracted patches for training/testing
def get_data_training(DRIVE_train_imgs_original,
                      DRIVE_train_groudTruth,
                      patch_height,
                      patch_width,
                      N_subimgs,
                      inside_FOV):
    train_imgs_original = load_hdf5(DRIVE_train_imgs_original)
    train_masks = load_hdf5(DRIVE_train_groudTruth) #masks always the same
    
    ####select the test imgs as the new train imgs  (test id:04-13)
    

    ###hard_imgs=np.load('DRIVE_datasets_training_testing/test_hard_imgs.npy')
    ###hard_masks=np.load('DRIVE_datasets_training_testing/test_hard_masks.npy')
    ###train_imgs_original=np.concatenate((train_imgs_original,hard_imgs),axis=0)
    ###train_masks=np.concatenate((train_masks,hard_masks),axis=0)
    ##visualize(group_images(train_imgs_original[0:20,:,:,:],5),'imgs_train')#.show()  #check original imgs train


    train_imgs = my_PreProc(train_imgs_original)
    ##train_thin_masks=np.load('/home/zhaolei/pytorch_patches/DRIVE_datasets_training_testing/train_thin_gts.npy')
    train_masks = train_masks/255.
    ##print('thin_mask_max_value',np.max(train_thin_masks))
    ##train_imgs=np.load(DRIVE_train_imgs_original)
    ###edges=np.load('DRIVE_datasets_training_testing/train_edges.npy')
    ###train_imgs=np.concatenate((train_imgs,edges),axis=1)
    ####pred_channel=np.load('drive_train_val/train_predictions_out.npy')
    ####train_imgs=np.concatenate((train_imgs,pred_channel),axis=1)
    ###print('train_imgs with edges:',train_imgs.shape)
    train_imgs = train_imgs[:,:,9:574,:]  #cut bottom and top so now it is 565*565
    train_masks = train_masks[:,:,9:574,:]  #cut bottom and top so now it is 565*565
    #train_thin_masks=train_thin_masks[:,:,9:574,:]
    data_consistency_check(train_imgs,train_masks)

    #check masks are within 0-1
    assert(np.min(train_masks)==0 and np.max(train_masks)==1)

    print ("\ntrain images/masks shape:")
    print (train_imgs.shape)
    print ("train images range (min-max): " +str(np.min(train_imgs)) +' - '+str(np.max(train_imgs)))
    print ("train masks are within 0-1\n")
    
    #extract the TRAINING patches from the full images
    patches_imgs_train, patches_masks_train= extract_random(train_imgs,train_masks,patch_height,patch_width,N_subimgs,inside_FOV)
    #data_consistency_check(patches_imgs_train, patches_masks_train)
    #print("thin mask shape:",patches_thin_mask.shape)

    print ("\ntrain PATCHES images/masks shape:")
    print (patches_imgs_train.shape)
    #print ("train PATCHES images range (min-max): " +str(np.min(patches_imgs_train)) +' - '+str(np.max(patches_imgs_train)))
    
    return patches_imgs_train, patches_masks_train#,patches_thin_mask, patches_imgs_test, patches_masks_test

def get_data_training_with_loss_mask(DRIVE_train_imgs_original,
                      DRIVE_train_groudTruth,
                      mask_path,
                      patch_height,
                      patch_width,
                      N_subimgs,
                      inside_FOV):
    train_imgs_original = load_hdf5(DRIVE_train_imgs_original)
    train_masks = load_hdf5(DRIVE_train_groudTruth) #masks always the same

    train_imgs = my_PreProc(train_imgs_original)
    train_masks = train_masks/255.
    dilation_masks=np.load(mask_path)
    print('dilation_mask_max_value',np.max(dilation_masks))
 
    train_imgs = train_imgs[:,:,9:574,:]  #cut bottom and top so now it is 565*565
    train_masks = train_masks[:,:,9:574,:]  #cut bottom and top so now it is 565*565
    dilation_masks=dilation_masks[:,:,9:574,:]
    data_consistency_check(train_imgs,train_masks)

    #check masks are within 0-1
    assert(np.min(train_masks)==0 and np.max(train_masks)==1)

    print ("\ntrain images/masks shape:")
    print (train_imgs.shape)
    print ("train images range (min-max): " +str(np.min(train_imgs)) +' - '+str(np.max(train_imgs)))
    print ("train masks are within 0-1\n")

    #extract the TRAINING patches from the full images
    patches_imgs_train, patches_masks_train,patches_dilation_mask= extract_random_loss_mask(train_imgs,train_masks,dilation_masks,patch_height,patch_width,N_subimgs,inside_FOV)
    data_consistency_check(patches_imgs_train, patches_masks_train)
    #print("thin mask shape:",patches_thin_mask.shape)

    print ("\ntrain PATCHES images/masks shape:")
    print (patches_imgs_train.shape)
    print ("train PATCHES images range (min-max): " +str(np.min(patches_imgs_train)) +' - '+str(np.max(patches_imgs_train)))

    return patches_imgs_train, patches_masks_train,patches_dilation_mask#, patches_imgs_test, patches_masks_test

def get_data_patches(image_array,mask_array,patch_height,patch_width,N_subimgs,inside_FOV):
    train_imgs_original = image_array
    train_masks = mask_array
    train_imgs = my_PreProc(train_imgs_original)
    ##train_thin_masks=np.load('/home/zhaolei/pytorch_patches/DRIVE_datasets_training_testing/train_thin_gts.npy')
    train_masks = train_masks/255.
    train_imgs = train_imgs[:,:,9:574,:]  #cut bottom and top so now it is 565*565
    train_masks = train_masks[:,:,9:574,:]  #cut bottom and top so now it is 565*565
    #train_thin_masks=train_thin_masks[:,:,9:574,:]
    data_consistency_check(train_imgs,train_masks)

    #check masks are within 0-1
    assert(np.min(train_masks)==0 and np.max(train_masks)==1)

    print ("\ntrain images/masks shape:")
    print (train_imgs.shape)
    print ("train images range (min-max): " +str(np.min(train_imgs)) +' - '+str(np.max(train_imgs)))
    print ("train masks are within 0-1\n")

    #extract the TRAINING patches from the full images
    patches_imgs_train, patches_masks_train= extract_random(train_imgs,train_masks,patch_height,patch_width,N_subimgs,inside_FOV)
    '''
    red_patches=np.load('./all_train_hard_patch/all_red_patches.npy')
    red_patches_masks=np.load('./all_train_hard_patch/all_red_patches_masks.npy')
    blue_patches=np.load('./all_train_hard_patch/all_blue_patches.npy')
    blue_patches_masks = np.load('./all_train_hard_patch/all_blue_patches_masks.npy')
    patches_imgs_train=np.concatenate((patches_imgs_train,red_patches,blue_patches),axis=0)
    patches_masks_train=np.concatenate((patches_masks_train,red_patches_masks,blue_patches_masks),axis=0)
    '''
    data_consistency_check(patches_imgs_train, patches_masks_train)
    
    #print("thin mask shape:",patches_thin_mask.shape)

    print ("\ntrain PATCHES images/masks shape:")
    print (patches_imgs_train.shape)
    print ("train PATCHES images range (min-max): " +str(np.min(patches_imgs_train)) +' - '+str(np.max(patches_imgs_train)))

    return patches_imgs_train, patches_masks_train

def get_data_patches_1(image_array,mask_array,patch_height,patch_width,N_subimgs,inside_FOV):
    train_imgs_original = load_hdf5(image_array)
    train_masks = load_hdf5(mask_array)
    train_imgs = my_PreProc(train_imgs_original)
    ##train_thin_masks=np.load('/home/zhaolei/pytorch_patches/DRIVE_datasets_training_testing/train_thin_gts.npy')
    train_masks = train_masks/255.
    train_imgs = train_imgs[:,:,9:574,:]  #cut bottom and top so now it is 565*565
    train_masks = train_masks[:,:,9:574,:]  #cut bottom and top so now it is 565*565
    #train_thin_masks=train_thin_masks[:,:,9:574,:]
    data_consistency_check(train_imgs,train_masks)

    #check masks are within 0-1
    assert(np.min(train_masks)==0 and np.max(train_masks)==1)

    print ("\ntrain images/masks shape:")
    print (train_imgs.shape)
    print ("train images range (min-max): " +str(np.min(train_imgs)) +' - '+str(np.max(train_imgs)))
    print ("train masks are within 0-1\n")

    #extract the TRAINING patches from the full images
    #all_train_hard_patch_12
    patches_imgs_train, patches_masks_train= extract_random(train_imgs,train_masks,patch_height,patch_width,N_subimgs,inside_FOV)
    red_patches=np.load('./all_train_hardbig_patch_constant_21/all_red_patches.npy')
    red_patches_masks=np.load('./all_train_hardbig_patch_constant_21/all_red_patches_masks.npy')
    blue_patches=np.load('./all_train_hardbig_patch_constant_21/all_blue_patches.npy')
    blue_patches_masks = np.load('./all_train_hardbig_patch_constant_21/all_blue_patches_masks.npy')
    patches_imgs_train=np.concatenate((patches_imgs_train,red_patches,blue_patches),axis=0)
    patches_masks_train=np.concatenate((patches_masks_train,red_patches_masks,blue_patches_masks),axis=0)
    data_consistency_check(patches_imgs_train, patches_masks_train)
    #print("thin mask shape:",patches_thin_mask.shape)

    print ("\ntrain PATCHES images/masks shape:")
    print (patches_imgs_train.shape)
    print ("train PATCHES images range (min-max): " +str(np.min(patches_imgs_train)) +' - '+str(np.max(patches_imgs_train)))

    return patches_imgs_train, patches_masks_train

def get_data_patches_2(image_array,mask_array,patch_height,patch_width,N_subimgs,inside_FOV):
    train_imgs_original = load_hdf5(image_array)
    train_masks = load_hdf5(mask_array)
    train_imgs = my_PreProc(train_imgs_original)
    ##train_thin_masks=np.load('/home/zhaolei/pytorch_patches/DRIVE_datasets_training_testing/train_thin_gts.npy')
    train_masks = train_masks/255.
    train_imgs = train_imgs[:,:,9:574,:]  #cut bottom and top so now it is 565*565
    train_masks = train_masks[:,:,9:574,:]  #cut bottom and top so now it is 565*565
    #train_thin_masks=train_thin_masks[:,:,9:574,:]
    data_consistency_check(train_imgs,train_masks)

    #check masks are within 0-1
    assert(np.min(train_masks)==0 and np.max(train_masks)==1)

    print ("\ntrain images/masks shape:")
    print (train_imgs.shape)
    print ("train images range (min-max): " +str(np.min(train_imgs)) +' - '+str(np.max(train_imgs)))
    print ("train masks are within 0-1\n")

    #extract the TRAINING patches from the full images
    patches_imgs_train, patches_masks_train= extract_random(train_imgs,train_masks,patch_height,patch_width,N_subimgs,inside_FOV)
    print('hard noise patches from all_train_hardbigpatchconstant_nosmallscale14')
    red_patches=np.load('./all_train_hardbigpatchconstant_nosmallscale14/all_red_patches.npy')
    red_patches_masks=np.load('./all_train_hardbigpatchconstant_nosmallscale14/all_red_patches_masks.npy')
    blue_patches=np.load('./all_train_hardbigpatchconstant_nosmallscale14/all_blue_patches.npy')
    blue_patches_masks = np.load('./all_train_hardbigpatchconstant_nosmallscale14/all_blue_patches_masks.npy')
    patches_imgs_train=np.concatenate((patches_imgs_train,red_patches,blue_patches),axis=0)
    patches_masks_train=np.concatenate((patches_masks_train,red_patches_masks,blue_patches_masks),axis=0)
    data_consistency_check(patches_imgs_train, patches_masks_train)
    #print("thin mask shape:",patches_thin_mask.shape)

    print ("\ntrain PATCHES images/masks shape:")
    print (patches_imgs_train.shape)
    print ("train PATCHES images range (min-max): " +str(np.min(patches_imgs_train)) +' - '+str(np.max(patches_imgs_train)))

    return patches_imgs_train, patches_masks_train

def get_data_patches_3(image_array,mask_array,patch_height,patch_width,N_subimgs,inside_FOV):
    train_imgs_original = load_hdf5(image_array)
    train_masks = load_hdf5(mask_array)
    train_imgs = my_PreProc(train_imgs_original)
    ##train_thin_masks=np.load('/home/zhaolei/pytorch_patches/DRIVE_datasets_training_testing/train_thin_gts.npy')
    train_masks = train_masks/255.
    train_imgs = train_imgs[:,:,9:574,:]  #cut bottom and top so now it is 565*565
    train_masks = train_masks[:,:,9:574,:]  #cut bottom and top so now it is 565*565
    #train_thin_masks=train_thin_masks[:,:,9:574,:]
    data_consistency_check(train_imgs,train_masks)

    #check masks are within 0-1
    assert(np.min(train_masks)==0 and np.max(train_masks)==1)

    print ("\ntrain images/masks shape:")
    print (train_imgs.shape)
    print ("train images range (min-max): " +str(np.min(train_imgs)) +' - '+str(np.max(train_imgs)))
    print ("train masks are within 0-1\n")

    #extract the TRAINING patches from the full images
    patches_imgs_train, patches_masks_train= extract_random(train_imgs,train_masks,patch_height,patch_width,N_subimgs,inside_FOV)
    split=424949
    red_patches=np.load('./all_train_constant_trans14_hard_select_5fold_redonly/all_red_patches.npy')
    red_patches_masks=np.load('./all_train_constant_trans14_hard_select_5fold_redonly/all_red_patches_masks.npy')
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(red_patches)
    np.random.seed(seed)
    np.random.shuffle(red_patches_masks)
    red_patches=red_patches[:split]
    red_patches_masks=red_patches_masks[:split]
    
    blue_patches=np.load('./all_train_hardbigpatchconstant_nosmallscale14/all_blue_patches.npy')
    blue_patches_masks = np.load('./all_train_hardbigpatchconstant_nosmallscale14/all_blue_patches_masks.npy')
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(blue_patches)
    np.random.seed(seed)
    np.random.shuffle(blue_patches_masks)
    blue_patches=blue_patches[:split]
    blue_patches_masks=blue_patches_masks[:split]
    
    patches_imgs_train=np.concatenate((patches_imgs_train,red_patches,blue_patches),axis=0)
    patches_masks_train=np.concatenate((patches_masks_train,red_patches_masks,blue_patches_masks),axis=0)
    data_consistency_check(patches_imgs_train, patches_masks_train)
    #print("thin mask shape:",patches_thin_mask.shape)

    print ("\ntrain PATCHES images/masks shape:")
    print (patches_imgs_train.shape)
    print ("train PATCHES images range (min-max): " +str(np.min(patches_imgs_train)) +' - '+str(np.max(patches_imgs_train)))

    return patches_imgs_train, patches_masks_train
def get_data_patches_4(image_array,mask_array,patch_height,patch_width,N_subimgs,inside_FOV,hard_number):
    train_imgs_original = load_hdf5(image_array)
    train_masks = load_hdf5(mask_array)
    train_imgs = my_PreProc(train_imgs_original)
    ##train_thin_masks=np.load('/home/zhaolei/pytorch_patches/DRIVE_datasets_training_testing/train_thin_gts.npy')
    train_masks = train_masks/255.
    train_imgs = train_imgs[:,:,9:574,:]  #cut bottom and top so now it is 565*565
    train_masks = train_masks[:,:,9:574,:]  #cut bottom and top so now it is 565*565
    #train_thin_masks=train_thin_masks[:,:,9:574,:]
    data_consistency_check(train_imgs,train_masks)

    #check masks are within 0-1
    assert(np.min(train_masks)==0 and np.max(train_masks)==1)

    print ("\ntrain images/masks shape:")
    print (train_imgs.shape)
    print ("train images range (min-max): " +str(np.min(train_imgs)) +' - '+str(np.max(train_imgs)))
    print ("train masks are within 0-1\n")

    #extract the TRAINING patches from the full images
    patches_imgs_train, patches_masks_train= extract_random(train_imgs,train_masks,patch_height,patch_width,N_subimgs,inside_FOV)
    if hard_number < 1100000:
        red_patches=np.load('./all_train_constant_trans14_hard_select_5fold_redonly/all_red_patches.npy')
        red_patches_masks=np.load('./all_train_constant_trans14_hard_select_5fold_redonly/all_red_patches_masks.npy')
        red_patches_add=np.load('./all_train_hardbigpatchconstant_nosmallscale14/add_red_patches.npy')
        red_patches_masks_add=np.load('./all_train_hardbigpatchconstant_nosmallscale14/add_red_mask.npy')
        patches_imgs_train_red=np.concatenate((red_patches,red_patches_add),axis=0)
        patches_masks_train_red=np.concatenate((red_patches_masks,red_patches_masks_add),axis=0)
        seed = 547
        np.random.seed(seed)
        np.random.shuffle(patches_imgs_train_red)
        np.random.seed(seed)
        np.random.shuffle(patches_masks_train_red)
        red_patches=patches_imgs_train_red[:int(hard_number/2)]
        red_patches_masks=patches_masks_train_red[:int(hard_number/2)]

        blue_patches=np.load('./all_train_hardbigpatchconstant_nosmallscale14/all_blue_patches.npy')
        blue_patches_masks = np.load('./all_train_hardbigpatchconstant_nosmallscale14/all_blue_patches_masks.npy')
        seed = 547
        np.random.seed(seed)
        np.random.shuffle(blue_patches)
        np.random.seed(seed)
        np.random.shuffle(blue_patches_masks)
        blue_patches=blue_patches[:int(hard_number/2)]
        blue_patches_masks=blue_patches_masks[:int(hard_number/2)]
    else:
        red_patches=np.load('./all_train_constant_trans14_hard_select_5fold_redonly/all_red_patches.npy')
        red_patches_masks=np.load('./all_train_constant_trans14_hard_select_5fold_redonly/all_red_patches_masks.npy')
        red_patches_add=np.load('./all_train_hardbigpatchconstant_nosmallscale14/add_red_patches.npy')
        red_patches_masks_add=np.load('./all_train_hardbigpatchconstant_nosmallscale14/add_red_mask.npy')
        patches_imgs_train_red=np.concatenate((red_patches,red_patches_add),axis=0)
        patches_masks_train_red=np.concatenate((red_patches_masks,red_patches_masks_add),axis=0)
        seed = 547
        np.random.seed(seed)
        np.random.shuffle(patches_imgs_train_red)
        np.random.seed(seed)
        np.random.shuffle(patches_masks_train_red)
        red_patches=patches_imgs_train_red[:(hard_number-543536)]
        red_patches_masks=patches_masks_train_red[:(hard_number-543536)]
        
        blue_patches=np.load('./all_train_hardbigpatchconstant_nosmallscale14/all_blue_patches.npy')
        blue_patches_masks = np.load('./all_train_hardbigpatchconstant_nosmallscale14/all_blue_patches_masks.npy')
        seed = 547
        np.random.seed(seed)
        np.random.shuffle(blue_patches)
        np.random.seed(seed)
        np.random.shuffle(blue_patches_masks)
        blue_patches=blue_patches[:543536]
        blue_patches_masks=blue_patches_masks[:543536]
    
    patches_imgs_train=np.concatenate((patches_imgs_train,red_patches,blue_patches),axis=0)
    patches_masks_train=np.concatenate((patches_masks_train,red_patches_masks,blue_patches_masks),axis=0)
    data_consistency_check(patches_imgs_train, patches_masks_train)
    #print("thin mask shape:",patches_thin_mask.shape)

    print ("\ntrain PATCHES images/masks shape:")
    print (patches_imgs_train.shape)
    print ("train PATCHES images range (min-max): " +str(np.min(patches_imgs_train)) +' - '+str(np.max(patches_imgs_train)))

    return patches_imgs_train, patches_masks_train
def get_data_patches_5(image_array,mask_array,patch_height,patch_width,N_subimgs,inside_FOV,fold):
    train_imgs_original = load_hdf5(image_array)
    train_masks = load_hdf5(mask_array)
    train_imgs = my_PreProc(train_imgs_original)
    ##train_thin_masks=np.load('/home/zhaolei/pytorch_patches/DRIVE_datasets_training_testing/train_thin_gts.npy')
    train_masks = train_masks/255.
    train_imgs = train_imgs[:,:,9:574,:]  #cut bottom and top so now it is 565*565
    train_masks = train_masks[:,:,9:574,:]  #cut bottom and top so now it is 565*565
    #train_thin_masks=train_thin_masks[:,:,9:574,:]
    data_consistency_check(train_imgs,train_masks)

    #check masks are within 0-1
    assert(np.min(train_masks)==0 and np.max(train_masks)==1)

    print ("\ntrain images/masks shape:")
    print (train_imgs.shape)
    print ("train images range (min-max): " +str(np.min(train_imgs)) +' - '+str(np.max(train_imgs)))
    print ("train masks are within 0-1\n")

    #extract the TRAINING patches from the full images
    patches_imgs_train, patches_masks_train= extract_random(train_imgs,train_masks,patch_height,patch_width,N_subimgs,inside_FOV)
    red_patches=np.load('./all_train_constant_trans7_hard_select_'+str(fold)+'fold/all_red_patches.npy')
    red_patches_masks=np.load('./all_train_constant_trans7_hard_select_'+str(fold)+'fold/all_red_patches_masks.npy')
    blue_patches=np.load('./all_train_constant_trans7_hard_select_'+str(fold)+'fold/all_blue_patches.npy')
    blue_patches_masks =np.load('./all_train_constant_trans7_hard_select_'+str(fold)+'fold/all_blue_patches_masks.npy')
    patches_imgs_train=np.concatenate((patches_imgs_train,red_patches,blue_patches),axis=0)
    patches_masks_train=np.concatenate((patches_masks_train,red_patches_masks,blue_patches_masks),axis=0)
    data_consistency_check(patches_imgs_train, patches_masks_train)
    #print("thin mask shape:",patches_thin_mask.shape)

    print ("\ntrain PATCHES images/masks shape:")
    print (patches_imgs_train.shape)
    print ("train PATCHES images range (min-max): " +str(np.min(patches_imgs_train)) +' - '+str(np.max(patches_imgs_train)))

    return patches_imgs_train, patches_masks_train

def get_data_patches_false_positives(image_array,mask_array,patch_height,patch_width,N_subimgs,inside_FOV):
    train_imgs_original = load_hdf5(image_array)
    train_masks = load_hdf5(mask_array)
    train_imgs = my_PreProc(train_imgs_original)
    ##train_thin_masks=np.load('/home/zhaolei/pytorch_patches/DRIVE_datasets_training_testing/train_thin_gts.npy')
    train_masks = train_masks/255.
    train_imgs = train_imgs[:,:,9:574,:]  #cut bottom and top so now it is 565*565
    train_masks = train_masks[:,:,9:574,:]  #cut bottom and top so now it is 565*565
    #train_thin_masks=train_thin_masks[:,:,9:574,:]
    data_consistency_check(train_imgs,train_masks)

    #check masks are within 0-1
    assert(np.min(train_masks)==0 and np.max(train_masks)==1)

    print ("\ntrain images/masks shape:")
    print (train_imgs.shape)
    print ("train images range (min-max): " +str(np.min(train_imgs)) +' - '+str(np.max(train_imgs)))
    print ("train masks are within 0-1\n")

    #extract the TRAINING patches from the full images
    patches_imgs_train, patches_masks_train= extract_random(train_imgs,train_masks,patch_height,patch_width,N_subimgs,inside_FOV)
    print('hard noise patches from all_train_hardbigpatchconstant_nosmallscale14')
    
    blue_patches=np.load('./all_train_hardbigpatchconstant_nosmallscale14/all_blue_patches.npy')
    blue_patches_masks = np.load('./all_train_hardbigpatchconstant_nosmallscale14/all_blue_patches_masks.npy')
    patches_imgs_train=np.concatenate((patches_imgs_train,blue_patches),axis=0)
    patches_masks_train=np.concatenate((patches_masks_train,blue_patches_masks),axis=0)
    data_consistency_check(patches_imgs_train, patches_masks_train)
    #print("thin mask shape:",patches_thin_mask.shape)

    print ("\ntrain PATCHES images/masks shape:")
    print (patches_imgs_train.shape)
    print ("train PATCHES images range (min-max): " +str(np.min(patches_imgs_train)) +' - '+str(np.max(patches_imgs_train)))

    return patches_imgs_train, patches_masks_train

def get_data_patches_false_negatives(image_array,mask_array,patch_height,patch_width,N_subimgs,inside_FOV):
    train_imgs_original = load_hdf5(image_array)
    train_masks = load_hdf5(mask_array)
    train_imgs = my_PreProc(train_imgs_original)
    ##train_thin_masks=np.load('/home/zhaolei/pytorch_patches/DRIVE_datasets_training_testing/train_thin_gts.npy')
    train_masks = train_masks/255.
    train_imgs = train_imgs[:,:,9:574,:]  #cut bottom and top so now it is 565*565
    train_masks = train_masks[:,:,9:574,:]  #cut bottom and top so now it is 565*565
    #train_thin_masks=train_thin_masks[:,:,9:574,:]
    data_consistency_check(train_imgs,train_masks)

    #check masks are within 0-1
    assert(np.min(train_masks)==0 and np.max(train_masks)==1)
    print ("\ntrain images/masks shape:")
    print (train_imgs.shape)
    print ("train images range (min-max): " +str(np.min(train_imgs)) +' - '+str(np.max(train_imgs)))
    print ("train masks are within 0-1\n")
    patches_imgs_train, patches_masks_train=extract_random(train_imgs,train_masks,patch_height,patch_width,N_subimgs,inside_FOV)
    red_patches=np.load('./all_train_constant_trans7_hard_select_5fold/all_red_patches.npy')
    red_patches_masks=np.load('./all_train_constant_trans7_hard_select_5fold/all_red_patches_masks.npy')
    red_patches_from=np.load('./all_train_hardbigpatchconstant_nosmallscale14/all_red_patches.npy')
    red_patches_masks_from=np.load('./all_train_hardbigpatchconstant_nosmallscale14/all_red_patches_masks.npy')
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(red_patches_from)
    np.random.seed(seed)
    np.random.shuffle(red_patches_masks_from)
    red_add=red_patches_from[:73297]
    red_add_mask=red_patches_masks_from[:73297]
    
    patches_imgs_train=np.concatenate((patches_imgs_train,red_patches,red_add),axis=0)
    patches_masks_train=np.concatenate((patches_masks_train,red_patches_masks,red_add_mask),axis=0)
    data_consistency_check(patches_imgs_train, patches_masks_train)
    #print("thin mask shape:",patches_thin_mask.shape)

    print ("\ntrain PATCHES images/masks shape:")
    print (patches_imgs_train.shape)
    print ("train PATCHES images range (min-max): " +str(np.min(patches_imgs_train)) +' - '+str(np.max(patches_imgs_train)))

    return patches_imgs_train, patches_masks_train
#Load the original data and return the extracted patches for training/testing
def get_data_testing(DRIVE_test_imgs_original, DRIVE_test_groudTruth, Imgs_to_test, patch_height, patch_width):
    ### test
    test_imgs_original = load_hdf5(DRIVE_test_imgs_original)
    test_masks = load_hdf5(DRIVE_test_groudTruth)

    test_imgs = my_PreProc(test_imgs_original)
    test_masks = test_masks/255.

    #extend both images and masks so they can be divided exactly by the patches dimensions
    test_imgs = test_imgs[0:Imgs_to_test,:,:,:]
    test_masks = test_masks[0:Imgs_to_test,:,:,:]
    test_imgs = test_imgs[:, :, 9:574, :]  # cut bottom and top so now it is 565*565
    test_masks = test_masks[:, :, 9:574, :]  # cut bottom and top so now it is 565*565
    #test_imgs = paint_border(test_imgs,patch_height,patch_width)
    #test_masks = paint_border(test_masks,patch_height,patch_width)

    data_consistency_check(test_imgs, test_masks)

    #check masks are within 0-1
    assert(np.max(test_masks)==1  and np.min(test_masks)==0)

    print ("\ntest images/masks shape:")
    print (test_imgs.shape)
    print ("test images range (min-max): " +str(np.min(test_imgs)) +' - '+str(np.max(test_imgs)))
    print ("test masks are within 0-1\n")

    #extract the TEST patches from the full images
    patches_imgs_test = extract_ordered(test_imgs,patch_height,patch_width)
    patches_masks_test = extract_ordered(test_masks,patch_height,patch_width)
    data_consistency_check(patches_imgs_test, patches_masks_test)

    print ("\ntest PATCHES images/masks shape:")
    print (patches_imgs_test.shape)
    print ("test PATCHES images range (min-max): " +str(np.min(patches_imgs_test)) +' - '+str(np.max(patches_imgs_test)))

    return patches_imgs_test, patches_masks_test
def get_data_testing_test(DRIVE_test_imgs_original, DRIVE_test_groudTruth, Imgs_to_test, patch_height, patch_width):
    ### test
    test_imgs_original = load_hdf5(DRIVE_test_imgs_original)
    test_masks = load_hdf5(DRIVE_test_groudTruth)

    test_imgs = my_PreProc(test_imgs_original)
    test_masks = test_masks/255.

    #extend both images and masks so they can be divided exactly by the patches dimensions
    test_imgs = test_imgs[0:Imgs_to_test,:,:,:]
    test_masks = test_masks[0:Imgs_to_test,:,:,:]
    test_imgs = paint_border_test(test_imgs,patch_height,patch_width)
    test_masks = paint_border_test(test_masks,patch_height,patch_width)

    data_consistency_check(test_imgs, test_masks)

    #check masks are within 0-1
    assert(np.max(test_masks)==1  and np.min(test_masks)==0)

    print ("\ntest images/masks shape:")
    print (test_imgs.shape)
    print ("test images range (min-max): " +str(np.min(test_imgs)) +' - '+str(np.max(test_imgs)))
    print ("test masks are within 0-1\n")

    #extract the TEST patches from the full images
    patches_imgs_test = extract_ordered(test_imgs,patch_height,patch_width)
    patches_masks_test = extract_ordered(test_masks,patch_height,patch_width)
    data_consistency_check(patches_imgs_test, patches_masks_test)

    print ("\ntest PATCHES images/masks shape:")
    print (patches_imgs_test.shape)
    print ("test PATCHES images range (min-max): " +str(np.min(patches_imgs_test)) +' - '+str(np.max(patches_imgs_test)))

    return patches_imgs_test, patches_masks_test

def get_data_testing_overlap_second_channel(DRIVE_test_imgs_original, DRIVE_test_groudTruth,Imgs_to_test, patch_height, patch_width, stride_height, stride_width):
    test_imgs_orig= load_hdf5(DRIVE_test_imgs_original)
    test_masks = load_hdf5(DRIVE_test_groudTruth)
    test_imgs = my_PreProc(test_imgs_orig)
    test_masks = test_masks/255.
    #extend both images and masks so they can be divided exactly by the patches dimensions
    test_imgs = test_imgs[0:Imgs_to_test,:,:,:]
    test_masks = test_masks[0:Imgs_to_test,:,:,:]
    test_imgs = paint_border_overlap(test_imgs, patch_height, patch_width, stride_height, stride_width)
    second_channel=np.load('drive_train_with_alternate_20train_1/predictions_out_M1.npy')
    second_channel=paint_border_overlap(second_channel, patch_height, patch_width, stride_height, stride_width)

    #check masks are within 0-1
    assert(np.max(test_masks) == 1  and np.min(test_masks) == 0)

    print("\ntest images shape:")
    print(test_imgs.shape)
    print("\ntest mask shape:")
    print(test_masks.shape)
    print("test images range (min-max): " +str(np.min(test_imgs)) +' - '+str(np.max(test_imgs)))
    print("test masks are within 0-1\n")

    #extract the TEST patches from the full images
    test_imgs=np.concatenate((test_imgs,second_channel),axis=1)
    patches_imgs_test= extract_ordered_overlap(test_imgs,patch_height,patch_width,stride_height,stride_width)
    
    print("\ntest PATCHES images shape:")
    print(patches_imgs_test.shape)
    print("test PATCHES images range (min-max): " +str(np.min(patches_imgs_test)) +' - '+str(np.max(patches_imgs_test)))
    return patches_imgs_test, test_imgs.shape[2], test_imgs.shape[3], test_masks


# Load the original data and return the extracted patches for testing
# return the ground truth in its original shape
def get_data_testing_overlap_1(DRIVE_test_imgs_original, DRIVE_test_groudTruth,Imgs_to_test, patch_height, patch_width, stride_height, stride_width,
    pad):
    test_imgs_orig = DRIVE_test_imgs_original ###load_hdf5
    test_masks = DRIVE_test_groudTruth ###load_hdf5

    test_imgs = my_PreProc(test_imgs_orig)
    test_masks = test_masks / 255.
    # extend both images and masks so they can be divided exactly by the patches dimensions
    test_imgs = test_imgs[0:Imgs_to_test, :, :, :]
    test_masks = test_masks[0:Imgs_to_test, :, :, :]
    #test_imgs = test_imgs[:, :, 9:574, :]  # cut bottom and top so now it is 565*565
    test_imgs=np.pad(test_imgs,((0,0),(0,0),(pad,pad),(pad,pad)),'constant')
    test_imgs = paint_border_overlap(test_imgs, patch_height, patch_width, stride_height, stride_width)
    #cut_h=test_imgs.shape[2]-test_imgs_orig.shape[2]
    #cut_w=test_imgs.shape[3]-test_imgs_orig.shape[3]

    # check masks are within 0-1
    assert (np.max(test_masks) == 1 and np.min(test_masks) == 0)

    print("\ntest images shape:")
    print(test_imgs.shape)
    print("\ntest mask shape:")
    print(test_masks.shape)
    print("test images range (min-max): " + str(np.min(test_imgs)) + ' - ' + str(np.max(test_imgs)))
    print("test masks are within 0-1\n")

    # extract the TEST patches from the full images
    patches_imgs_test = extract_ordered_overlap(test_imgs, patch_height, patch_width, stride_height, stride_width)

    print("\ntest PATCHES images shape:")
    print(patches_imgs_test.shape)
    print("test PATCHES images range (min-max): " + str(np.min(patches_imgs_test)) + ' - ' + str(
        np.max(patches_imgs_test)))

    return patches_imgs_test, test_imgs.shape[2]-2*pad, test_imgs.shape[3]-2*pad,test_masks


def get_data_testing_overlap_fold(DRIVE_test_imgs_original, DRIVE_test_groudTruth,Imgs_to_test, patch_height, patch_width, stride_height, stride_width):
    
    test_imgs_orig=DRIVE_test_imgs_original  ###
    test_masks =DRIVE_test_groudTruth ###load_hdf5

    test_imgs = my_PreProc(test_imgs_orig)
    test_masks = test_masks/255.
    #extend both images and masks so they can be divided exactly by the patches dimensions
    test_imgs = test_imgs[0:Imgs_to_test,:,:,:]
    test_masks = test_masks[0:Imgs_to_test,:,:,:]
    test_imgs = paint_border_overlap(test_imgs,patch_height,patch_width, stride_height, stride_width)
    
    #check masks are within 0-1
    assert(np.max(test_masks) == 1  and np.min(test_masks) == 0)

    print("\ntest images shape:")
    print(test_imgs.shape)
    print("\ntest mask shape:")
    print(test_masks.shape)
    print("test images range (min-max): " +str(np.min(test_imgs)) +' - '+str(np.max(test_imgs)))
    print("test masks are within 0-1\n")

    #extract the TEST patches from the full images
    #test_imgs=np.concatenate((test_imgs,second_channel),axis=1)
    patches_imgs_test= extract_ordered_overlap(test_imgs,patch_height,patch_width,stride_height,stride_width)
    #patches_masks_test= extract_ordered_overlap(test_masks,patch_height,patch_width,stride_height,stride_width)
    
    print("\ntest PATCHES images shape:")
    print(patches_imgs_test.shape)
    print("test PATCHES images range (min-max): " +str(np.min(patches_imgs_test)) +' - '+str(np.max(patches_imgs_test)))

    return patches_imgs_test, test_imgs.shape[2], test_imgs.shape[3],test_masks
def get_data_testing_overlap(DRIVE_test_imgs_original, DRIVE_test_groudTruth,Imgs_to_test, patch_height, patch_width, stride_height, stride_width):
    ### test
    ###test_imgs_orig=np.load(DRIVE_test_imgs_original)
    ###test_masks=np.load(DRIVE_test_groudTruth)
    
    test_imgs_orig=load_hdf5(DRIVE_test_imgs_original)  ###
    test_masks = load_hdf5(DRIVE_test_groudTruth) ###load_hdf5
    ##test_imgs_orig=np.concatenate((test_imgs_orig[0:3],test_imgs_orig[13:20]),axis=0)
    ##test_masks=np.concatenate((test_masks[0:3],test_masks[13:20]),axis=0)

    ###test_masks = np.concatenate((test_masks_ori[0:4],test_masks_ori[7:20]),axis=0)
    ###test_imgs_orig = np.concatenate((test_imgs_original[0:4],test_imgs_original[7:20]),axis=0)

    test_imgs = my_PreProc(test_imgs_orig)
    ###edges=np.load('DRIVE_datasets_training_testing/test_edges.npy')
    ###test_imgs=np.concatenate((test_imgs,edges),axis=1)
    ###predontest=np.load('drive_train_val/predictions.npy')
    ###test_imgs=np.concatenate((test_imgs,predontest),axis=1)
    test_masks = test_masks/255.
    #extend both images and masks so they can be divided exactly by the patches dimensions
    test_imgs = test_imgs[0:Imgs_to_test,:,:,:]
    test_masks = test_masks[0:Imgs_to_test,:,:,:]
    test_imgs = paint_border_overlap(test_imgs,patch_height,patch_width, stride_height, stride_width)
    #second_channel=np.load('drive_train_with_alternate_20train_1/predictions_out_M1.npy')
    #test_masks=paint_border_overlap(test_masks, patch_height, patch_width, stride_height, stride_width)

    #check masks are within 0-1
    assert(np.max(test_masks) == 1  and np.min(test_masks) == 0)

    print("\ntest images shape:")
    print(test_imgs.shape)
    print("\ntest mask shape:")
    print(test_masks.shape)
    print("test images range (min-max): " +str(np.min(test_imgs)) +' - '+str(np.max(test_imgs)))
    print("test masks are within 0-1\n")

    #extract the TEST patches from the full images
    #test_imgs=np.concatenate((test_imgs,second_channel),axis=1)
    patches_imgs_test= extract_ordered_overlap(test_imgs,patch_height,patch_width,stride_height,stride_width)
    #patches_masks_test= extract_ordered_overlap(test_masks,patch_height,patch_width,stride_height,stride_width)
    
    print("\ntest PATCHES images shape:")
    print(patches_imgs_test.shape)
    print("test PATCHES images range (min-max): " +str(np.min(patches_imgs_test)) +' - '+str(np.max(patches_imgs_test)))

    return patches_imgs_test, test_imgs.shape[2], test_imgs.shape[3],test_masks
def get_data_testing_overlap_light(DRIVE_test_imgs_original, DRIVE_test_groudTruth,Imgs_to_test, patch_height, patch_width, stride_height, stride_width):
    
    test_imgs_orig=load_hdf5(DRIVE_test_imgs_original)  ###
    test_masks = load_hdf5(DRIVE_test_groudTruth) ###load_hdf5
   

    test_imgs = my_PreProc(test_imgs_orig)
    
    test_masks = test_masks/255.
    #extend both images and masks so they can be divided exactly by the patches dimensions
    test_imgs = test_imgs[0:Imgs_to_test,:,:,:]
    test_masks = test_masks[0:Imgs_to_test,:,:,:]
    test_imgs = paint_border_overlap(test_imgs,patch_height,patch_width, stride_height, stride_width)

    #check masks are within 0-1
    assert(np.max(test_masks) == 1  and np.min(test_masks) == 0)

    print("\ntest images shape:")
    print(test_imgs.shape)
    print("\ntest mask shape:")
    print(test_masks.shape)
    print("test images range (min-max): " +str(np.min(test_imgs)) +' - '+str(np.max(test_imgs)))
    print("test masks are within 0-1\n")

    return test_imgs.shape[2], test_imgs.shape[3],test_masks
def get_data_testing_complete(DRIVE_test_imgs_original, DRIVE_test_groudTruth):
    
    test_imgs_orig= load_hdf5(DRIVE_test_imgs_original)
    test_masks = load_hdf5(DRIVE_test_groudTruth)
   
    test_imgs = my_PreProc(test_imgs_orig)
    
    test_masks = test_masks/255.
  
    
    #check masks are within 0-1
    assert(np.max(test_masks) == 1  and np.min(test_masks) == 0)

    print("\ntest images shape:")
    print(test_imgs.shape)
    print("\ntest mask shape:")
    print(test_masks.shape)
    print("test images range (min-max): " +str(np.min(test_imgs)) +' - '+str(np.max(test_imgs)))
    print("test masks are within 0-1\n")
    return test_imgs,test_masks


#data consinstency check
def data_consistency_check(imgs,masks):
    assert(len(imgs.shape) == len(masks.shape))
    assert(imgs.shape[0] == masks.shape[0])
    assert(imgs.shape[2] == masks.shape[2])
    assert(imgs.shape[3] == masks.shape[3])
    assert(masks.shape[1] == 1)
    #assert(imgs.shape[1] == 1 or imgs.shape[1] == 3)

def extract_random(full_imgs,full_masks, patch_h,patch_w, N_patches, inside=True):
    '''
    if (N_patches%full_imgs.shape[0] != 0):
        print ("N_patches: plase enter a multiple of 20")
        exit()
    '''
    assert (len(full_imgs.shape) == 4 and len(full_masks.shape) == 4)  #4D arrays
    #assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  #check the channel is 1 or 3
    assert (full_masks.shape[1] == 1)   #masks only black and white
    assert (full_imgs.shape[2] == full_masks.shape[2] and full_imgs.shape[3] == full_masks.shape[3])
    patches = np.empty((N_patches,full_imgs.shape[1],patch_h,patch_w))
    patches_masks = np.empty((N_patches,full_masks.shape[1],patch_h,patch_w))
    #patches_thin_mask=np.empty((N_patches,full_masks.shape[1],patch_h,patch_w))
    img_h = full_imgs.shape[2]  #height of the full image
    img_w = full_imgs.shape[3] #width of the full image
    # (0,0) in the center of the image
    patch_per_img = int(N_patches / full_imgs.shape[0])  #N_patches equally divided in the full images
    
    print ("patches per full image: " +str(patch_per_img))
    iter_tot = 0   #iter over the total numbe rof patches (N_patches)
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
            patches[iter_tot] = patch
            patches_masks[iter_tot] = patch_mask
    
            iter_tot += 1   #total
            k += 1  #per full_img
    return patches, patches_masks #,patches_thin_mask
def extract_random_target(full_imgs,full_masks, patch_h,patch_w, N_patches, inside=True):
    if (N_patches%full_imgs.shape[0] != 0):
        print ("N_patches: plase enter a multiple of 20")
        exit()
    assert (len(full_imgs.shape) == 4 and len(full_masks.shape) == 4)  #4D arrays
    #assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  #check the channel is 1 or 3
    assert (full_masks.shape[1] == 1)   #masks only black and white
    assert (full_imgs.shape[2] == full_masks.shape[2] and full_imgs.shape[3] == full_masks.shape[3])
    patches = np.empty((N_patches,full_imgs.shape[1],patch_h,patch_w))
    patches_masks = np.empty((N_patches,full_masks.shape[1],48,48))
    #patches_thin_mask=np.empty((N_patches,full_masks.shape[1],patch_h,patch_w))
    img_h = full_imgs.shape[2]  #height of the full image
    img_w = full_imgs.shape[3] #width of the full image
    # (0,0) in the center of the image
    patch_per_img = int(N_patches / full_imgs.shape[0])  #N_patches equally divided in the full images
    
    print ("patches per full image: " +str(patch_per_img))
    iter_tot = 0   #iter over the total numbe rof patches (N_patches)
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
            patch_mask = full_masks[i,:,y_center-int(48/2):y_center+int(48/2),x_center-int(48/2):x_center+int(48/2)]
            patches[iter_tot] = patch
            patches_masks[iter_tot] = patch_mask
    
            iter_tot += 1   #total
            k += 1  #per full_img
    return patches, patches_masks
#extract patches randomly in the full training images
#  -- Inside OR in full image
def extract_random_thindense(full_imgs,full_masks, patch_h,patch_w, N_patches, inside=True):
    if (N_patches%full_imgs.shape[0] != 0):
        print ("N_patches: plase enter a multiple of 20")
        exit()
    assert (len(full_imgs.shape) == 4 and len(full_masks.shape) == 4)  #4D arrays
    #assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  #check the channel is 1 or 3
    assert (full_masks.shape[1] == 1)   #masks only black and white
    assert (full_imgs.shape[2] == full_masks.shape[2] and full_imgs.shape[3] == full_masks.shape[3])
    patches = np.empty((2*N_patches,full_imgs.shape[1],patch_h,patch_w))
    patches_masks = np.empty((2*N_patches,full_masks.shape[1],patch_h,patch_w))
    #patches_thin_mask=np.empty((N_patches,full_masks.shape[1],patch_h,patch_w))
    img_h = full_imgs.shape[2]  #height of the full image
    img_w = full_imgs.shape[3] #width of the full image
    # (0,0) in the center of the image
    patch_per_img = int(N_patches / full_imgs.shape[0])  #N_patches equally divided in the full images
    thin_patch_per_img=patch_per_img
    print ("patches per full image: " +str(patch_per_img))
    iter_tot = 0   #iter over the total numbe rof patches (N_patches)
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
            patches[iter_tot] = patch
            patches_masks[iter_tot] = patch_mask
    
            iter_tot += 1   #total
            k += 1  #per full_img
        thin_vessel=full_masks[i,0,:,:]-opening(full_masks[i,0,:,:],square(2))
        h_list, w_list=np.where(thin_vessel==1)
        print(h_list.shape[0])
        q=0
        while q<thin_patch_per_img:
            thin_index=random.randint(0,h_list.shape[0]-1)
            x_center = w_list[thin_index]
            if x_center <int(patch_w/2) or x_center > (img_w-int(patch_w/2)):
                continue
            # print "x_center " +str(x_center)
            y_center = h_list[thin_index]
            if y_center < int(patch_h/2) or y_center > (img_h-int(patch_h/2)):
                continue
            # print "y_center " +str(y_center)
            patch = full_imgs[i,:,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]
            patch_mask = full_masks[i,:,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]

            patches[iter_tot] = patch
            patches_masks[iter_tot] = patch_mask  
            iter_tot += 1   #total
            q+=1
    return patches, patches_masks #,patches_thin_mask
def extract_random_loss_mask(full_imgs,full_masks,dilation_masks, patch_h,patch_w, N_patches, inside=True):
    if (N_patches%full_imgs.shape[0] != 0):
        print ("N_patches: plase enter a multiple of 20")
        exit()
    assert (len(full_imgs.shape) == 4 and len(full_masks.shape) == 4)  #4D arrays
    #assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  #check the channel is 1 or 3
    assert (full_masks.shape[1] == 1)   #masks only black and white
    assert (full_imgs.shape[2] == full_masks.shape[2] and full_imgs.shape[3] == full_masks.shape[3])
    patches = np.empty((N_patches,full_imgs.shape[1],patch_h,patch_w))
    patches_masks = np.empty((N_patches,full_masks.shape[1],patch_h,patch_w))
    patches_thin_mask=np.empty((N_patches,full_masks.shape[1],patch_h,patch_w))
    img_h = full_imgs.shape[2]  #height of the full image
    img_w = full_imgs.shape[3] #width of the full image
    # (0,0) in the center of the image
    patch_per_img = int(N_patches / full_imgs.shape[0])  #N_patches equally divided in the full images
    
    print ("patches per full image: " +str(patch_per_img))
    iter_tot = 0   #iter over the total numbe rof patches (N_patches)
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
            patch_thin_mask=dilation_masks[i,:,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]
            patches[iter_tot] = patch
            patches_masks[iter_tot] = patch_mask
            patches_thin_mask[iter_tot]=patch_thin_mask
    
            iter_tot += 1   #total
            k += 1  #per full_img
    return patches, patches_masks ,patches_thin_mask

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


#Divide all the full_imgs in pacthes
def extract_ordered(full_imgs, patch_h, patch_w):
    assert (len(full_imgs.shape) == 4)  #4D arrays
    assert (full_imgs.shape[1] == 1 or full_imgs.shape[1] == 3)  #check the channel is 1 or 3
    img_h = full_imgs.shape[2]  #height of the full image
    img_w = full_imgs.shape[3] #width of the full image
    N_patches_h = int(img_h/patch_h) #round to lowest int
    if (img_h%patch_h != 0):
        print ("warning: " +str(N_patches_h) +" patches in height, with about " +str(img_h%patch_h) +" pixels left over")
    N_patches_w = int(img_w/patch_w) #round to lowest int
    if (img_w%patch_w != 0):
        print ("warning: " +str(N_patches_w) +" patches in width, with about " +str(img_w%patch_w) +" pixels left over")
    print ("number of patches per image: " +str(N_patches_h*N_patches_w))
    N_patches_tot = (N_patches_h*N_patches_w)*full_imgs.shape[0]
    patches = np.empty((N_patches_tot,full_imgs.shape[1],patch_h,patch_w))

    iter_tot = 0   #iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  #loop over the full images
        for h in range(N_patches_h):
            for w in range(N_patches_w):
                patch = full_imgs[i,:,h*patch_h:(h*patch_h)+patch_h,w*patch_w:(w*patch_w)+patch_w]
                patches[iter_tot] = patch
                iter_tot += 1   #total
    assert (iter_tot == N_patches_tot)
    return patches  #array with all the full_imgs divided in patches


def paint_border_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w):
    assert (len(full_imgs.shape)==4)  #4D arrays
    #assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  #check the channel is 1 or 3
    img_h = full_imgs.shape[2]  #height of the full image
    img_w = full_imgs.shape[3] #width of the full image
    leftover_h = (img_h-patch_h)%stride_h  #leftover on the h dim
    leftover_w = (img_w-patch_w)%stride_w  #leftover on the w dim
    if (leftover_h != 0):  #change dimension of img_h
        print ("\nthe side H is not compatible with the selected stride of " +str(stride_h))
        print ("img_h " +str(img_h) + ", patch_h " +str(patch_h) + ", stride_h " +str(stride_h))
        print ("(img_h - patch_h) MOD stride_h: " +str(leftover_h))
        print ("So the H dim will be padded with additional " +str(stride_h - leftover_h) + " pixels")
        tmp_full_imgs = np.zeros((full_imgs.shape[0],full_imgs.shape[1],img_h+(stride_h-leftover_h),img_w))
        tmp_full_imgs[0:full_imgs.shape[0],0:full_imgs.shape[1],0:img_h,0:img_w] = full_imgs
        full_imgs = tmp_full_imgs
    if (leftover_w != 0):   #change dimension of img_w
        print ("the side W is not compatible with the selected stride of " +str(stride_w))
        print ("img_w " +str(img_w) + ", patch_w " +str(patch_w) + ", stride_w " +str(stride_w))
        print ("(img_w - patch_w) MOD stride_w: " +str(leftover_w))
        print ("So the W dim will be padded with additional " +str(stride_w - leftover_w) + " pixels")
        tmp_full_imgs = np.zeros((full_imgs.shape[0],full_imgs.shape[1],full_imgs.shape[2],img_w+(stride_w - leftover_w)))
        tmp_full_imgs[0:full_imgs.shape[0],0:full_imgs.shape[1],0:full_imgs.shape[2],0:img_w] = full_imgs
        full_imgs = tmp_full_imgs
    print ("new full images shape: \n" +str(full_imgs.shape))
    return full_imgs

#Divide all the full_imgs in pacthes
def extract_ordered_overlap(full_imgs,patch_h, patch_w,stride_h,stride_w):
    assert (len(full_imgs.shape) == 4)  #4D arrays
    #assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  #check the channel is 1 or 3
    img_h = full_imgs.shape[2]  #height of the full image
    img_w = full_imgs.shape[3] #width of the full image
    assert ((img_h-patch_h)%stride_h == 0 and (img_w-patch_w)%stride_w == 0)
    N_patches_img = ((img_h-patch_h)//stride_h+1)*((img_w-patch_w)//stride_w+1)  #// --> division between integers
    N_patches_tot = N_patches_img*full_imgs.shape[0]
    print ("Number of patches on h : " +str(((img_h-patch_h)//stride_h+1)))
    print ("Number of patches on w : " +str(((img_w-patch_w)//stride_w+1)))
    print ("number of patches per image: " +str(N_patches_img) +", totally for this dataset: " +str(N_patches_tot))
    patches = np.empty((N_patches_tot,full_imgs.shape[1],patch_h,patch_w))
    iter_tot = 0   #iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  #loop over the full images
        for h in range((img_h-patch_h)//stride_h+1):
            for w in range((img_w-patch_w)//stride_w+1):
                patch = full_imgs[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]
                patches[iter_tot]=patch
                iter_tot += 1   #total
    assert (iter_tot == N_patches_tot)
    return patches  #array with all the full_imgs divided in patches


def recompone_overlap(preds, img_h, img_w, stride_h, stride_w):
    assert (len(preds.shape) == 4)  #4D arrays
    assert (preds.shape[1] == 1 or preds.shape[1] == 3)  #check the channel is 1 or 3
    patch_h = preds.shape[2]
    patch_w = preds.shape[3]
    N_patches_h = (img_h-patch_h)//stride_h+1   #img_h too  large ,48 too small
    N_patches_w = (img_w-patch_w)//stride_w+1
    N_patches_img = N_patches_h * N_patches_w
    print ("N_patches_h: "+str(N_patches_h))
    print ("N_patches_w: "+str(N_patches_w))
    print ("N_patches_img: "+str(N_patches_img))
    #assert (preds.shape[0]%N_patches_img == 0)
    N_full_imgs = preds.shape[0]//N_patches_img
    print ("According to the dimension inserted, there are " +str(N_full_imgs) +" full images (of " +str(img_h)+"x" +str(img_w) +" each)")
    full_prob = np.zeros((N_full_imgs,preds.shape[1],img_h,img_w))  #itialize to zero mega array with sum of Probabilities
    full_sum = np.zeros((N_full_imgs,preds.shape[1],img_h,img_w))

    k = 0 #iterator over all the patches
    for i in range(N_full_imgs):
        for h in range((img_h-patch_h)//stride_h+1):
            for w in range((img_w-patch_w)//stride_w+1):
                full_prob[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=preds[k]
                #full_prob[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]=np.maximum(full_prob[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w],preds[k])
                full_sum[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=1
                k += 1
    #assert(k == preds.shape[0])
    print(np.where(full_sum==0))
    assert(np.min(full_sum) >= 1.0)  #at least one
    final_avg = full_prob/full_sum
    #final_avg=full_prob
    print (final_avg.shape)
    #assert(np.max(final_avg) <= 1.0) #max value for a pixel is 1.0
    #assert(np.min(final_avg) >= 0.0) #min value for a pixel is 0.0
    return final_avg


#Recompone the full images with the patches
def recompone(data,N_h,N_w):
    assert (data.shape[1] == 1 or data.shape[1] == 3)  #check the channel is 1 or 3
    assert(len(data.shape) == 4)
    N_pacth_per_img = N_w*N_h
    #assert(data.shape[0]%N_pacth_per_img == 0)
    N_full_imgs = data.shape[0]/N_pacth_per_img
    patch_h = data.shape[2]
    patch_w = data.shape[3]
    N_pacth_per_img = N_w*N_h
    #define and start full recompone
    full_recomp = np.empty((int(N_full_imgs),int(data.shape[1]),int(N_h*patch_h),int(N_w*patch_w)))
    k = 0  #iter full img
    s = 0  #iter single patch
    while (s < data.shape[0]):
        #recompone one:
        single_recon = np.empty((data.shape[1],N_h*patch_h,N_w*patch_w))
        for h in range(N_h):
            for w in range(N_w):
                single_recon[:,h*patch_h:(h*patch_h)+patch_h,w*patch_w:(w*patch_w)+patch_w]=data[s]
                s += 1
        full_recomp[k]=single_recon
        k += 1
    assert (k == N_full_imgs)
    return full_recomp


#Extend the full images because patch divison is not exact
def paint_border(data,patch_h,patch_w):
    assert (len(data.shape) == 4)  #4D arrays
    assert (data.shape[1] == 1 or data.shape[1] == 3)  #check the channel is 1 or 3
    img_h = data.shape[2]
    img_w = data.shape[3]
    new_img_h = 0
    new_img_w = 0
    if (img_h%patch_h) == 0:
        new_img_h = img_h
    else:
        new_img_h = (int(img_h)/int(patch_h))*patch_h
    if (img_w%patch_w) == 0:
        new_img_w = img_w
    else:
        new_img_w = (int(img_w)/int(patch_w))*patch_w
    new_data = np.zeros((data.shape[0],data.shape[1],int(new_img_h),int(new_img_w)))
    new_data[:,:,0:img_h,0:img_w] = data[:,:,:,:]
    return new_data

def paint_border_test(data,patch_h,patch_w):
    assert (len(data.shape) == 4)  #4D arrays
    assert (data.shape[1] == 1 or data.shape[1] == 3)  #check the channel is 1 or 3
    img_h = data.shape[2]
    img_w = data.shape[3]
    new_img_h = 0
    new_img_w = 0
    if (img_h%patch_h) == 0:
        new_img_h = img_h
    else:
        new_img_h = ((int(img_h)/int(patch_h))+1)*patch_h
    if (img_w%patch_w) == 0:
        new_img_w = img_w
    else:
        new_img_w = ((int(img_w)/int(patch_w))+1)*patch_w
    new_data = np.zeros((data.shape[0],data.shape[1],int(new_img_h),int(new_img_w)))
    new_data[:,:,0:img_h,0:img_w] = data[:,:,:,:]
    return new_data

#return only the pixels contained in the FOV, for both images and masks
def pred_only_FOV(data_imgs,data_masks,original_imgs_border_masks):
    assert (len(data_imgs.shape) == 4 and len(data_masks.shape) == 4)  #4D arrays
    assert (data_imgs.shape[0] == data_masks.shape[0])
    assert (data_imgs.shape[2] == data_masks.shape[2])
    assert (data_imgs.shape[3] == data_masks.shape[3])
    assert (data_imgs.shape[1] == 1 and data_masks.shape[1] == 1)  #check the channel is 1
    height = data_imgs.shape[2]
    width = data_imgs.shape[3]
    new_pred_imgs = []
    new_pred_masks = []
    for i in range(data_imgs.shape[0]):  #loop over the full images
        for x in range(width):
            for y in range(height):
                if inside_FOV_DRIVE(i,x,y,original_imgs_border_masks)==True:
                    new_pred_imgs.append(data_imgs[i,:,y,x])
                    new_pred_masks.append(data_masks[i,:,y,x])
    new_pred_imgs = np.asarray(new_pred_imgs)
    new_pred_masks = np.asarray(new_pred_masks)
    return new_pred_imgs, new_pred_masks

#function to set to black everything outside the FOV, in a full image
def kill_border(data, original_imgs_border_masks):
    assert (len(data.shape) == 4)  #4D arrays
    assert (data.shape[1] == 1 or data.shape[1] == 3)  #check the channel is 1 or 3
    height = data.shape[2]
    width = data.shape[3]
    for i in range(data.shape[0]):  #loop over the full images
        for x in range(width):
            for y in range(height):
                if inside_FOV_DRIVE(i,x,y,original_imgs_border_masks) == False:
                    data[i,:,y,x] = 0.0


def inside_FOV_DRIVE(i, x, y, DRIVE_masks):
    assert (len(DRIVE_masks.shape) == 4)  #4D arrays
    assert (DRIVE_masks.shape[1] == 1)  #DRIVE masks is black and white
    # DRIVE_masks = DRIVE_masks/255.  #NOOO!! otherwise with float numbers takes forever!!

    if (x >= DRIVE_masks.shape[3] or y >= DRIVE_masks.shape[2]): #my image bigger than the original
        return False

    if (DRIVE_masks[i,0,y,x] > 0):  #0==black pixels
        # print DRIVE_masks[i,0,y,x]  #verify it is working right
        return True
    else:
        return False
