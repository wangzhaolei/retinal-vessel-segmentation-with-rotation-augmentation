import os
import h5py
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics

smooth=1.
'''
for iii in range(test_num_images):
       print(iii)

       pos_index = np.int32(np.where(imgs_mask_test[iii]==1))
       pred_pos_index = np.array(np.where(pred[iii]>.5))
       insect_index = np.int32(np.where(np.logical_and(pred[iii]>.5, imgs_mask_test[iii]==1)))
       pred_mask = np.zeros([imgs_mask_test[iii].shape[0], imgs_mask_test[iii].shape[1], 3], dtype=np.float32)
       pred_mask[pos_index[0], pos_index[1], 0] = 1.
       pred_mask[pred_pos_index[0], pred_pos_index[1], ...] = 1.
       pred_mask[insect_index[0], insect_index[1], 0] = 0.
       pred_mask[insect_index[0], insect_index[1], 1] = 1.
       pred_mask[insect_index[0], insect_index[1], 2] = 0.
       seg_img = np.copy(imgs_test[iii])
       seg_img[imgs_mask_test[iii]==1] = np.max(seg_img)

       imshow(imgs_test[iii,:,:,0],seg_img[:,:,0],pred_mask,pred[iii,:,:,0],title=['test image','ground truth','prediction','heat map'])
       plt.savefig(os.path.join(pred_dir, str(iii) + '_pred.png'))'''

def rgb2d(rgb):
    r,g,b=rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]
    array2d=2*r+3*g+b
    return array2d
def two2rgb(array2d):
    pred_mask = np.zeros([array2d.shape[0], array2d.shape[1], 3])

    red_index=np.where(array2d==2)
    blue_index=np.where(array2d==1)
    white_index=np.where(array2d==6)

    pred_mask[red_index[0], red_index[1], 0] = 1 ##red
    pred_mask[blue_index[0],blue_index[1],2] =1 ###blue
    pred_mask[white_index[0],white_index[1],:]=1 ###white
    return pred_mask

def error_image(pred,gt,th):
    """
    :param pred:shape is (patch_height,patch_width)
    :param gt:(patch_height,patch_width)
    :return: pred_mask (patch_height,patch_width,3)
    """
    pos_index = np.int32(np.where(gt == 1))
    pred_pos_index = np.array(np.where(pred >= th))
    insect_index = np.int32(np.where(np.logical_and(pred >= th, gt == 1)))
    pred_mask = np.zeros([gt.shape[0], gt.shape[1], 3], dtype=np.float32)
    pred_mask[pos_index[0], pos_index[1], 0] = 1. ###red(1,0,0) is

    pred_mask[pred_pos_index[0], pred_pos_index[1], 2] = 1.  ###blue(0,0,1)

    #pred_mask[pred_pos_index[0], pred_pos_index[1], ...] = 1.
    #pred_mask[insect_index[0], insect_index[1], 0] = 0.
    #pred_mask[insect_index[0], insect_index[1], 1] = 1.
    #pred_mask[insect_index[0], insect_index[1], 2] = 0.

    pred_mask[insect_index[0], insect_index[1], ...] = 1.###white(1,1,1)
    #pred_mask[pred_pos_index[0], pred_pos_index[1], 0] = 0.
    #pred_mask[pred_pos_index[0], pred_pos_index[1], 1] = 0.


    return pred_mask
def evaluation(y_true, y_pred):
    count_TP = 0
    count_TN = 0
    count_P = 0
    count_T = 0
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    dice_coef=(2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

    count_TP = np.sum(((y_true_f == 1)*(y_pred_f==1))) + count_TP
    count_TN = np.sum((y_true_f+y_pred_f)==0)+count_TN
    count_P = np.sum((y_pred_f==1)) + count_P
    count_T = np.sum((y_true_f==1)) + count_T
    count_FP=count_P-count_TP

    precision=count_TP/(count_P+1e-4)
    recall=count_TP/(count_T+1e-4)
    acc=(count_TP+count_TN)/(count_TN+count_P+count_T-count_TP)
    iou=count_TP/(count_P+count_T-count_TP)
    spec=count_TN/(count_TN+count_FP)

    return acc,precision,recall,spec,iou,dice_coef
def computeAUC(gt,pred,pos_label=1):
    gt = np.array(gt).flatten()
    pred=np.array(pred).flatten()
    fpr,tpr,thresholds=metrics.roc_curve(gt,pred,pos_label=pos_label)
    return metrics.auc(fpr,tpr)
def thin_recall(thin_gt,pred,thresh):
    count_gt_thin = np.sum((thin_gt == 1))
    pred_all = pred > thresh
    thin_TP = np.sum(pred_all[thin_gt==1]*thin_gt[thin_gt==1])
    recall = thin_TP / (count_gt_thin+1e-4)
    return recall
def imshow(*args,**kwargs):
    title=kwargs.get('title','')
    cmap=kwargs.get('cmap','gray')
    if len(args) == 0:
        raise ValueError("No images given to imshow")
    elif len(args) == 1:
        plt.title(title)
        plt.imshow(args[0], interpolation='none')
    else:
        n=len(args)
        if type(title) == str:
            title = [title]*n
        cmap = [cmap]*n
        plt.figure(figsize = (n*10,40))
        for i in range(n):
            plt.subplot(2,2,i+1)
            plt.title(title[i])
            plt.imshow(args[i],cmap[i])
 #plt.show()
def load_hdf5(infile):
    with h5py.File(infile,"r") as f:  #"with" close the file after its nested commands
        return f["image"][()]

def write_hdf5(arr,outfile):
    with h5py.File(outfile,"w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)

#convert RGB image in black and white
def rgb2gray(rgb):
    assert (len(rgb.shape) == 4)  #4D arrays
    assert (rgb.shape[1] == 3)
    #bn_imgs=rgb[:,1,:,:]  #only use the green channel
    bn_imgs = rgb[:,0,:,:]*0.299 + rgb[:,1,:,:]*0.587 + rgb[:,2,:,:]*0.114
    bn_imgs = np.reshape(bn_imgs,(rgb.shape[0],1,rgb.shape[2],rgb.shape[3]))
    return bn_imgs

#group a set of images row per columns
def group_images(data,per_row):
    assert data.shape[0]%per_row == 0
    assert (data.shape[1]==1 or data.shape[1]==3)
    data = np.transpose(data,(0,2,3,1))  #corect format for imshow
    all_stripe = []
    for i in range(int(data.shape[0]/per_row)):
        stripe = data[i*per_row]
        for k in range(i*per_row+1, i*per_row+per_row):
            stripe = np.concatenate((stripe,data[k]),axis=1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    for i in range(1,len(all_stripe)):
        totimg = np.concatenate((totimg,all_stripe[i]),axis=0)
    return totimg


#visualize image (as PIL image, NOT as matplotlib!)
def visualize(data,filename):
    assert (len(data.shape) == 3) #height*width*channels
    img = None
    if data.shape[2] == 1:  #in case it is black and white
        data = np.reshape(data,(data.shape[0],data.shape[1]))
    if np.max(data) > 1:
        img = Image.fromarray(data.astype(np.uint8))   #the image is already 0-255
    else:
        img = Image.fromarray((data*255).astype(np.uint8))  #the image is between 0-1
    img.save(filename + '.png')
    return img
def visualize_1(data,filename):
    assert (len(data.shape) == 3) #height*width*channels
    if data.shape[2] == 1:  #in case it is black and white
        data = np.reshape(data,(data.shape[0],data.shape[1]))
    plt.imshow(data)

    #plt.savefig(filename + '.png')


#prepare the mask in the right shape for the Unet
def masks_Unet(masks):
    assert (len(masks.shape) == 4)  #4D arrays
    assert (masks.shape[1] == 1)  #check the channel is 1
    im_h = masks.shape[2]
    im_w = masks.shape[3]
    masks = np.reshape(masks,(masks.shape[0],im_h*im_w))
    new_masks = np.empty((masks.shape[0],im_h*im_w,2))
    for i in range(masks.shape[0]):
        for j in range(im_h*im_w):
            if  masks[i,j] == 0:
                new_masks[i,j,0] = 1
                new_masks[i,j,1] = 0
            else:
                new_masks[i,j,0] = 0
                new_masks[i,j,1] = 1
    return new_masks


def pred_to_imgs(pred, patch_height, patch_width, mode="original"):
    assert (len(pred.shape) == 3)  #3D array: (Npatches,height*width,2)
    assert (pred.shape[2] == 2 )  #check the classes are 2
    pred_images = np.empty((pred.shape[0],pred.shape[1]))  #(Npatches,height*width)
    if mode == "original":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                pred_images[i,pix] = pred[i,pix,1]
    elif mode == "threshold":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                if pred[i,pix,1] >= 0.5:
                    pred_images[i,pix] = 1
                else:
                    pred_images[i,pix] = 0
    else:
        print ("mode "+str(mode)+" not recognized, it can be 'original' or 'threshold'")
        exit()
    pred_images = np.reshape(pred_images,(pred_images.shape[0],1, patch_height, patch_width))
    return pred_images

