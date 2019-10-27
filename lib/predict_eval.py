import os

import h5py
import numpy as np
from skimage.morphology import opening, rectangle, square
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from extract_patches import pred_only_FOV
# ====== Evaluate the results
def thin_recall(thin_gt, pred, thresh):
    count_gt_thin = np.sum((thin_gt == 1))
    pred_all = pred >= thresh
    thin_TP = np.sum(pred_all[thin_gt == 1] * thin_gt[thin_gt == 1])
    recall = thin_TP / count_gt_thin
    return recall
def evaluate_predict(pred_imgs_out,gtruth_masks,test_border_masks):
    print("\n\n========  Evaluate the results =======================")
   
    # predictions only inside the FOV
    y_scores, y_true = pred_only_FOV(pred_imgs_out, gtruth_masks, test_border_masks)  # returns data only inside the FOV
   
    # Area under the ROC curve
    fpr, tpr, thresholds = roc_curve((y_true), y_scores)
    AUC_ROC = roc_auc_score(y_true, y_scores)
    # test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
    print("\nArea under the ROC curve: " + str(AUC_ROC))
    '''
    rOc_curve = plt.figure()
    plt.plot(fpr, tpr, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
    plt.title('ROC curve')
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")
    plt.savefig(path_experiment + "ROC.png")
    '''

    # Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
    recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
    AUC_prec_rec = np.trapz(precision, recall)
    print("\nArea under Precision-Recall curve: " + str(AUC_prec_rec))
    '''
    prec_rec_curve = plt.figure()
    plt.plot(recall, precision, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
    plt.title('Precision - Recall curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower right")
    plt.savefig(path_experiment + "Precision_recall.png")
    '''
    # Confusion matrix
    threshold_confusion = 0.5
    print("\nConfusion matrix:  Custom threshold (for positive) of " + str(threshold_confusion))
    y_scores[y_scores>= threshold_confusion]=1
    y_scores[y_scores< threshold_confusion] = 0
    y_pred=y_scores
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
    return AUC_ROC,AUC_prec_rec,accuracy,specificity,sensitivity,precision,jaccard_index,F1_score
def thin_eval(pred_imgs_out,gtruth_masks):
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
    return thin_3pixel_recall_indivi,thin_3pixel_auc_roc,thin_2pixel_recall_indivi,thin_2pixel_auc_roc




