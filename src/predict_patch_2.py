###################################################
#   Script to
#   - Calculate prediction of the test dataset
#   - Calculate the parameters to evaluate the prediction
##################################################
import numpy as np
import configparser
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from skimage.morphology import opening, rectangle, square
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
import sys

sys.path.insert(0, './lib/')
from help_functions import *
from unet_model import UNet, UNet_cat, UNet_level4_our
from extract_patches import recompone
from extract_patches import recompone_overlap
from extract_patches import paint_border
from extract_patches import kill_border
from extract_patches import pred_only_FOV
from extract_patches import get_data_testing_test
from extract_patches import get_data_testing_overlap
from pre_processing import my_PreProc
import argparse


def thin_recall(thin_gt, pred, thresh):
    count_gt_thin = np.sum((thin_gt == 1))
    pred_all = pred >= thresh
    thin_TP = np.sum(pred_all[thin_gt == 1] * thin_gt[thin_gt == 1])
    recall = thin_TP / count_gt_thin
    return recall


def test(experiment_path, test_epoch):
    # ========= CONFIG FILE TO READ FROM =======
    config = configparser.RawConfigParser()
    config.read('./' + experiment_path + '/' + experiment_path + '_config.txt')
    # ===========================================
    # run the training on invariant or local
    path_data = config.get('data paths', 'path_local')
    model = config.get('training settings', 'model')
    # original test images (for FOV selection)
    DRIVE_test_imgs_original = path_data + config.get('data paths', 'test_imgs_original')
    test_imgs_orig = load_hdf5(DRIVE_test_imgs_original)
    full_img_height = test_imgs_orig.shape[2]
    full_img_width = test_imgs_orig.shape[3]
    # the border masks provided by the DRIVE
    DRIVE_test_border_masks = path_data + config.get('data paths', 'test_border_masks')
    test_border_masks = load_hdf5(DRIVE_test_border_masks)
    # dimension of the patches
    patch_height = int(config.get('data attributes', 'patch_height'))
    patch_width = int(config.get('data attributes', 'patch_width'))
    # the stride in case output with average
    stride_height = int(config.get('testing settings', 'stride_height'))
    stride_width = int(config.get('testing settings', 'stride_width'))
    assert (stride_height < patch_height and stride_width < patch_width)
    # model name
    name_experiment = config.get('experiment name', 'name')
    path_experiment = './' + name_experiment + '/'
    # N full images to be predicted
    Imgs_to_test = int(config.get('testing settings', 'full_images_to_test'))
    # Grouping of the predicted images
    N_visual = int(config.get('testing settings', 'N_group_visual'))
    # ====== average mode ===========
    average_mode = config.getboolean('testing settings', 'average_mode')
    #N_subimgs = int(config.get('training settings', 'N_subimgs'))
    #batch_size = int(config.get('training settings', 'batch_size'))
    #epoch_size = N_subimgs // (batch_size)
    # #ground truth
    # gtruth= path_data + config.get('data paths', 'test_groundTruth')
    # img_truth= load_hdf5(gtruth)
    # visualize(group_images(test_imgs_orig[0:20,:,:,:],5),'original')#.show()
    # visualize(group_images(test_border_masks[0:20,:,:,:],5),'borders')#.show()
    # visualize(group_images(img_truth[0:20,:,:,:],5),'gtruth')#.show()

    # ============ Load the data and divide in patches
    patches_imgs_test = None
    new_height = None
    new_width = None
    masks_test = None
    patches_masks_test = None

    if average_mode == True:
        patches_imgs_test, new_height, new_width, masks_test= get_data_testing_overlap(
            DRIVE_test_imgs_original = DRIVE_test_imgs_original, #original'DRIVE_datasets_training_testing/test_hard_masks.npy'
            DRIVE_test_groudTruth = path_data + config.get('data paths', 'test_groundTruth'),  #masks
            Imgs_to_test = int(config.get('testing settings', 'full_images_to_test')),
            patch_height = patch_height,
            patch_width = patch_width,
            stride_height = stride_height,
            stride_width = stride_width)
    else:
        patches_imgs_test, patches_masks_test = get_data_testing_test(
            DRIVE_test_imgs_original = DRIVE_test_imgs_original,  #original
            DRIVE_test_groudTruth = path_data + config.get('data paths', 'test_groundTruth'),  #masks
            Imgs_to_test = int(config.get('testing settings', 'full_images_to_test')),
            patch_height = patch_height,
            patch_width = patch_width
        )
    #np.save(path_experiment + 'test_patches.npy', patches_imgs_test)
    #visualize(group_images(patches_imgs_test,100),'./'+name_experiment+'/'+"test_patches")

    # ================ Run the prediction of the patches ==================================
    best_last = config.get('testing settings', 'best_last')
    # Load the saved model
    if model == 'UNet':
        net = UNet(n_channels=1, n_classes=2)
    elif model == 'UNet_cat':
        net = UNet_cat(n_channels=1, n_classes=2)
    else:
        net = UNet_level4_our(n_channels=1, n_classes=2)
    # load data
    test_data = data.TensorDataset(torch.tensor(patches_imgs_test),torch.zeros(patches_imgs_test.shape[0]))
    test_loader = data.DataLoader(test_data, batch_size=1, pin_memory=True, shuffle=False)
    trained_model = path_experiment + 'DRIVE_' + str(test_epoch) + 'epoch.pth'
    print(trained_model)
    # trained_model= path_experiment+'DRIVE_unet2_B'+str(60*epoch_size)+'.pth'
    net.load_state_dict(torch.load(trained_model))
    net.eval()
    print('Finished loading model :' + trained_model)
    net = net.cuda()
    cudnn.benchmark = True
    # Calculate the predictions
    predictions_out = np.empty((patches_imgs_test.shape[0],patch_height*patch_width,2))
    for i_batch, (images, targets) in enumerate(test_loader):
        images = Variable(images.float().cuda())
        out1= net(images)

        pred = out1.permute(0,2,3,1)

        pred = F.softmax(pred, dim=-1)

        pred = pred.data.view(-1,patch_height*patch_width,2)

        predictions_out[i_batch] = pred

    # ===== Convert the prediction arrays in corresponding images
    pred_patches_out = pred_to_imgs(predictions_out, patch_height, patch_width, "original")
    #np.save(path_experiment + 'pred_patches_' + str(test_epoch) + "_epoch" + '.npy', pred_patches_out)
    #visualize(group_images(pred_patches_out,100),'./'+name_experiment+'/'+"pred_patches")


    #========== Elaborate and visualize the predicted images ====================
    pred_imgs_out = None
    orig_imgs = None
    gtruth_masks = None
    if average_mode == True:
        pred_imgs_out = recompone_overlap(pred_patches_out,new_height,new_width, stride_height, stride_width)
        orig_imgs = my_PreProc(test_imgs_orig[0:pred_imgs_out.shape[0],:,:,:])    #originals
        gtruth_masks = masks_test  #ground truth masks
    else:
        pred_imgs_out = recompone(pred_patches_out,10,9)       # predictions
        orig_imgs = recompone(patches_imgs_test,10,9)  # originals
        gtruth_masks = recompone(patches_masks_test,10,9)  #masks

    # apply the DRIVE masks on the repdictions #set everything outside the FOV to zero!!
    # DRIVE MASK  #only for visualization
    kill_border(pred_imgs_out, test_border_masks)
    # back to original dimensions
    orig_imgs = orig_imgs[:,:,0:full_img_height,0:full_img_width]
    pred_imgs_out = pred_imgs_out[:, :, 0:full_img_height, 0:full_img_width]
    gtruth_masks = gtruth_masks[:, :, 0:full_img_height, 0:full_img_width]

    print ("Orig imgs shape: "+str(orig_imgs.shape))
    print("pred imgs shape: " + str(pred_imgs_out.shape))
    print("Gtruth imgs shape: " + str(gtruth_masks.shape))
    np.save(path_experiment + 'pred_img_' + str(test_epoch) + "_epoch" + '.npy',pred_imgs_out)
    # visualize(group_images(orig_imgs,N_visual),path_experiment+"all_originals")#.show()
    if average_mode == True:
        visualize(group_images(pred_imgs_out, N_visual),
                  path_experiment + "all_predictions_" + str(test_epoch) + "thresh_epoch")
    else:
        visualize(group_images(pred_imgs_out, N_visual),
                  path_experiment + "all_predictions_" + str(test_epoch) + "epoch_no_average")
    visualize(group_images(gtruth_masks, N_visual), path_experiment + "all_groundTruths")

    # visualize results comparing mask and prediction:
    # assert (orig_imgs.shape[0] == pred_imgs_out.shape[0] and orig_imgs.shape[0] == gtruth_masks.shape[0])
    # N_predicted = orig_imgs.shape[0]
    # group = N_visual
    # assert (N_predicted%group == 0)
    

    # ====== Evaluate the results
    print("\n\n========  Evaluate the results =======================")
   
    # predictions only inside the FOV
    y_scores, y_true = pred_only_FOV(pred_imgs_out, gtruth_masks, test_border_masks)  # returns data only inside the FOV
    '''
    print("Calculating results only inside the FOV:")
    print("y scores pixels: " + str(
        y_scores.shape[0]) + " (radius 270: 270*270*3.14==228906), including background around retina: " + str(
        pred_imgs_out.shape[0] * pred_imgs_out.shape[2] * pred_imgs_out.shape[3]) + " (584*565==329960)")
    print("y true pixels: " + str(
        y_true.shape[0]) + " (radius 270: 270*270*3.14==228906), including background around retina: " + str(
        gtruth_masks.shape[2] * gtruth_masks.shape[3] * gtruth_masks.shape[0]) + " (584*565==329960)")
    '''
    # Area under the ROC curve
    fpr, tpr, thresholds = roc_curve((y_true), y_scores)
    AUC_ROC = roc_auc_score(y_true, y_scores)
    # test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
    print("\nArea under the ROC curve: " + str(AUC_ROC))
    rOc_curve = plt.figure()
    plt.plot(fpr, tpr, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
    plt.title('ROC curve')
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")
    plt.savefig(path_experiment + "ROC.png")

    # Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
    recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
    AUC_prec_rec = np.trapz(precision, recall)
    print("\nArea under Precision-Recall curve: " + str(AUC_prec_rec))
    prec_rec_curve = plt.figure()
    plt.plot(recall, precision, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
    plt.title('Precision - Recall curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower right")
    plt.savefig(path_experiment + "Precision_recall.png")

    # Confusion matrix
    threshold_confusion = 0.5
    print("\nConfusion matrix:  Custom threshold (for positive) of " + str(threshold_confusion))
    y_pred = np.empty((y_scores.shape[0]))
    for i in range(y_scores.shape[0]):
        if y_scores[i] >= threshold_confusion:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    confusion = confusion_matrix(y_true, y_pred)
    print(confusion)
    accuracy = 0
    if float(np.sum(confusion)) != 0:
        accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
    print("Global Accuracy: " + str(accuracy))
    specificity = 0
    if float(confusion[0, 0] + confusion[0, 1]) != 0:
        specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
    print("Specificity: " + str(specificity))
    sensitivity = 0
    if float(confusion[1, 1] + confusion[1, 0]) != 0:
        sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
    print("Sensitivity: " + str(sensitivity))
    precision = 0
    if float(confusion[1, 1] + confusion[0, 1]) != 0:
        precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
    print("Precision: " + str(precision))

    # Jaccard similarity index
    jaccard_index = jaccard_similarity_score(y_true, y_pred, normalize=True)
    print("\nJaccard similarity score: " + str(jaccard_index))

    # F1 score
    F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
    print("\nF1 score (F-measure): " + str(F1_score))
    ####evaluate the thin vessels
    thin_3pixel_recall_indivi = []
    thin_3pixel_auc_roc = []
    for j in range(pred_imgs_out.shape[0]):
        thick3=opening(gtruth_masks[j, 0, :, :], square(3))
        thin_gt = gtruth_masks[j, 0, :, :] - thick3
        
        thin_pred=pred_imgs_out[j, 0, :, :]
        
        thin_pred[thick3==1]=0
        thin_3pixel_recall_indivi.append(round(thin_recall(thin_gt, pred_imgs_out[j, 0, :, :], thresh=0.5), 4))
        thin_3pixel_auc_roc.append(round(roc_auc_score(thin_gt.flatten(), thin_pred.flatten()), 4))
    thin_2pixel_recall_indivi = []
    thin_2pixel_auc_roc = []
    for j in range(pred_imgs_out.shape[0]):
        thick=opening(gtruth_masks[j, 0, :, :], square(2))
        thin_gt = gtruth_masks[j, 0, :, :] - thick
        #thin_gt_only=thin_gt[thin_gt==1]
        #print(thin_gt_only)
        thin_pred=pred_imgs_out[j, 0, :, :]
        #thin_pred=thin_pred[thin_gt==1]
        thin_pred[thick==1]=0
        thin_2pixel_recall_indivi.append(round(thin_recall(thin_gt, pred_imgs_out[j, 0, :, :], thresh=0.5), 4))
        thin_2pixel_auc_roc.append(round(roc_auc_score(thin_gt.flatten(), thin_pred.flatten()), 4))
    
    #print("thin 2vessel recall:", thin_2pixel_recall_indivi)
    #print('thin 2vessel auc score', thin_2pixel_auc_roc)
    # Save the results
    with open(path_experiment + 'test_performances_all_epochs.txt', mode='a') as f:
        f.write("\n\n" + path_experiment + " test epoch:" + str(test_epoch)
                + '\naverage mode is:' + str(average_mode)
                + "\nArea under the ROC curve: %.4f" % (AUC_ROC)
                + "\nArea under Precision-Recall curve: %.4f" % (AUC_prec_rec)
                + "\nJaccard similarity score: %.4f" % (jaccard_index)
                + "\nF1 score (F-measure): %.4f" % (F1_score)
                + "\nConfusion matrix:"
                + str(confusion)
                + "\nACCURACY: %.4f" % (accuracy)
                + "\nSENSITIVITY: %.4f" % (sensitivity)
                + "\nSPECIFICITY: %.4f" % (specificity)
                + "\nPRECISION: %.4f" % (precision)
                + "\nthin 2vessels recall indivi:\n" + str(thin_2pixel_recall_indivi)
                + "\nthin 2vessels recall mean:%.4f" % (np.mean(thin_2pixel_recall_indivi))
                + "\nthin 2vessels auc indivi:\n" + str(thin_2pixel_auc_roc)
                + "\nthin 2vessels auc score mean:%.4f" % (np.mean(thin_2pixel_auc_roc))
                + "\nthin 3vessels recall indivi:\n" + str(thin_3pixel_recall_indivi)
                + "\nthin 3vessels recall mean:%.4f" % (np.mean(thin_3pixel_recall_indivi))
                + "\nthin 3vessels auc indivi:\n" + str(thin_3pixel_auc_roc)
                + "\nthin 3vessels auc score mean:%.4f" % (np.mean(thin_3pixel_auc_roc))
                )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='retina vessel segmentation')
    parser.add_argument('--testdir', type=str, help='Checkpoint state_dict file to predict')
    parser.add_argument('--start', type=int, help='start test')
    args = parser.parse_args()
    for i in range(args.start,61,1):
        test(experiment_path=args.testdir, test_epoch=i)



'''
    for i in range(int(N_predicted/group)):
        orig_stripe = group_images(orig_imgs[i*group:(i*group)+group,:,:,:],group)
        masks_stripe = group_images(gtruth_masks[i*group:(i*group)+group,:,:,:],group)
        #pred_stripe_media= group_images(pred_imgs_media[i*group:(i*group)+group,:,:,:],group)
        pred_stripe_out= group_images(pred_imgs_out[i*group:(i*group)+group,:,:,:],group)
        total_img = np.concatenate((orig_stripe,masks_stripe,pred_stripe_out),axis=0)
        visualize(total_img,path_experiment+name_experiment +"pred_test_set"+str(i))#.show()
    '''
