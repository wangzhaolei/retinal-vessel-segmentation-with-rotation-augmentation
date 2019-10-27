###################################################
#   Script to:
#   - Load the images and extract the patches
#   - define the training
##################################################
import numpy as np
import argparse
import configparser
import sys
sys.path.insert(0, './lib/')
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
# if torch.cuda.is_available():
#    torch.set_default_tensor_type('torch.cuda.FloatTensor')
from tensorboardX import SummaryWriter
from help_functions import *
from loaddata import DRIVE_train_dataset
from unet_model import UNet, UNet_cat_with_side,UNet_side_loss, UNet_level4_our, UNet_cat
from myloss import CrossEntropyLoss2D


# ========
parser = argparse.ArgumentParser(description='retina vessel segmentation')
parser.add_argument('--configfile', type=str, help='Checkpoint state_dict file to resume training from')

args = parser.parse_args()
# ========= Load settings from Config file
global_config = args.configfile
config = configparser.RawConfigParser()
config.read(global_config)
# path to the datasets
path_data = config.get('data paths', 'path_local')
# Experiment name
name_experiment = config.get('experiment name', 'name')

print("copy the configuration file in the results folder")
os.system('cp ' + global_config + ' ./' + name_experiment + '/' + name_experiment + '_config.txt')
# training settings

resume = config.getboolean('training settings', 'resume')
model = config.get('training settings', 'model')
optim_select = config.get('training settings', 'optim_select')
vessel_weight = float(config.get('training settings', 'vessel_weight'))
#N_subimgs = int(config.get('training settings', 'N_subimgs'))
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))
start_iter = int(config.get('training settings', 'start_iter'))  # Resume training at this iter
num_workers = int(config.get('training settings', 'num_workers'))  # Number of workers used in dataloading
learning_rate = float(config.get('training settings', 'learning_rate'))  # help=initial learning rate
momentum = float(config.get('training settings', 'momentum'))  # help=Momentum value for optim
weight_decay = float(config.get('training settings', 'weight_decay'))  # help= Weight decay for SGD
gamma = float(config.get('training settings', 'gamma'))  # help= Gamma update for SGD
filename = './' + name_experiment + '/train_monitor.txt'
# DRIVE_train_imgs_original=path_data+'train_RGB_imgs.npy',
os.system('mkdir -p ' + './' + name_experiment + '/mylog')

writer = SummaryWriter('./' + name_experiment + '/mylog')




def train():
    print('Loading the dataset...')
    print('Training ' + model + ' on DRIVE patches')
    # ============ Load the data 
    dataset = DRIVE_train_dataset(root=config.get('training settings', 'train_data_dir'),
                                  train_file=config.get('training settings', 'train_file'),
                                 flip=config.getboolean('training settings', 'flip'))
    print('all patches:',len(dataset))
    # ======== define the dataloader

    train_loader = data.DataLoader(dataset, batch_size=batch_size,pin_memory=True, shuffle=True)
    '''
    patches_imgs_test, patches_masks_test = get_data_testing(
        DRIVE_test_imgs_original=path_data + config.get('data paths', 'test_imgs_original'),  # original
        DRIVE_test_groudTruth=path_data + config.get('data paths', 'test_groundTruth'),  # masks
        Imgs_to_test=int(config.get('testing settings', 'full_images_to_test')),
        patch_height=int(config.get('data attributes', 'patch_height')),
        patch_width=int(config.get('data attributes', 'patch_width'))
    )
    visualize(group_images(patches_imgs_test[0:40, :, :, :], 5),
              './' + name_experiment + '/' + "sample_test_imgs")  # .show()
    visualize(group_images(patches_masks_test[0:40, :, :, :], 5),
              './' + name_experiment + '/' + "sample_test_masks")  # .show()
    test_data = data.TensorDataset(torch.tensor(patches_imgs_test),
                                   torch.tensor(patches_masks_test))
    test_loader = data.DataLoader(test_data, batch_size=30, pin_memory=True, shuffle=True)
    '''
    if model == 'UNet':
        ssd_net = UNet(n_channels=1, n_classes=2)
    elif model == 'UNet_side_loss':
        ssd_net = UNet_side_loss(n_channels=1, n_classes=2)
    elif model == 'UNet_level4_our':
        ssd_net = UNet_level4_our(n_channels=1, n_classes=2)
    elif model == 'UNet_cat':
        ssd_net = UNet_cat(n_channels=1, n_classes=2)
    else:
        ssd_net = UNet_multichannel(n_channels=1, n_classes=2)

    net = ssd_net
    #dummy_input = Variable(torch.rand(1, 1, 48, 48))
    #writer.add_graph(net,dummy_input)
    
    net = torch.nn.DataParallel(ssd_net)#,devices_ids=[0,1,2])
    cudnn.benchmark = True
    net = net.cuda()
    '''
    if resume == True:
        ssd_net.load_state_dict(torch.load('./drive_train_with_64_adam_lr3/DRIVE_50epoch.pth'))
    
    if resume == False:
        print('Initializing weights...')

        ssd_net.inc.apply(weights_init)
        ssd_net.down1.apply(weights_init)
        ssd_net.down2.apply(weights_init)
        ssd_net.up1.apply(weights_init)
        ssd_net.up2.apply(weights_init)
        ssd_net.outc.apply(weights_init)
    '''
    #############################
    if optim_select == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # criterion = FocalLoss(class_num=2, alpha=None, gamma=2, size_average=True)
    DRIVE_weight = torch.cuda.FloatTensor([1.0, vessel_weight])
    criterion = CrossEntropyLoss2D(weights=DRIVE_weight)
    # criterion=dilation_loss()
    #criterion=CrossEntropyLoss2D(weights=None)
    # criterion=thin_mid_vessel_loss(thin_weight=80,mid_weight=9)#18
    net.train()
    step_index = 0

    epoch_size = len(dataset) // batch_size  #epoch 0 is the 1 epoch trained
    for epoch in range(start_iter, N_epochs + 1):
        train_loss = 0
        #acc = 0
        #precision = 0
        #recall = 0
        #spec = 0
        #dice_coef = 0
        #AUC = 0
        for i_batch, (images, targets) in enumerate(train_loader):
            iteration = epoch*epoch_size+i_batch
            # 50-90 the eval don't change so the 0.0001 is so small(35,50,90)
            # 90-150 don't change hhhh,so 1e-4 and 1e-5 is too small
            # no adjust learning rate is so shock ,but the where is the change node
            if epoch in (3,6,9):
                step_index += 1
                adjust_learning_rate(optimizer, gamma, step_index,epoch)

            images = Variable(images.float().cuda())
            targets = Variable(targets.long().cuda())  # long
            # dilation_mask=Variable(dilation_mask.long().cuda())

            # forward
            # t0 = time.time()
            out= net(images)
            # backprop
            optimizer.zero_grad()
            loss = criterion(out.permute(0, 2, 3, 1).contiguous().view(-1, 2), targets.view(-1))
            
            
            loss.backward()
            optimizer.step()
            # t1 = time.time()
            train_loss += loss.item()
            writer.add_scalar('train/batch_loss', loss.item(),iteration)
            ori=F.softmax(out,dim=1)
            #pre = torch.ge(ori.data[:, 1, :, :], 0.5)
            #acc_, precision_, recall_, spec_, _, dice_coef_ = evaluation(targets.data, pre)
            #AUC_ = computeAUC(targets.data,ori.data[:,1,:,:], pos_label=1)

            #acc += acc_
            #precision += precision_
            #recall += recall_
            #spec += spec_
            #dice_coef += dice_coef_
            #AUC += AUC_

        print('epoch' + repr(epoch) + '|| Loss: %.4f ||' % (loss.item()))
        #if epoch % 10 == 0:
        print('Saving state, epoch:', epoch)
        torch.save(ssd_net.state_dict(), name_experiment + '/' + 'DRIVE_' + repr(epoch) + 'epoch.pth')
        '''
        if epoch % 30 == 0:
            visualize(group_images(images.data, 1), './' + name_experiment + '/' + "train_imgs_" + str(epoch))
            visualize(group_images(targets.data, 1), './' + name_experiment + '/' + "train_masks_" + str(epoch))
            visualize(group_images(torch.unsqueeze(ori.data[:, 1, :, :], 1), 1),
                      './' + name_experiment + '/' + "train_pred_" + str(epoch))
        '''     
        writer.add_scalar('train/loss', train_loss / epoch_size, epoch)
        '''
        writer.add_scalar('train/acc', acc / epoch_size, epoch)
        writer.add_scalar('train/AUC', AUC / epoch_size, epoch)
        writer.add_scalar('train/precision', precision / epoch_size, epoch)
        writer.add_scalar('train/recall', recall / epoch_size, epoch)
        writer.add_scalar('train/specificity', spec / epoch_size, epoch)
        writer.add_scalar('train/dice_score', dice_coef / epoch_size, epoch)
        '''
        '''
        # for test
        #if epoch % 5 == 0:
        test_loss = []
        test_acc = []
        test_AUC = []
        test_precision = []
        test_recall = []
        test_spec = []
        test_dice = []
        for test_batch, (test_images, test_targets) in enumerate(test_loader):
            test_images = Variable(test_images.float().cuda())
            test_targets = Variable(test_targets.long().cuda())  # long

            test_out,s1,s2,s3,s4 = net(test_images)
            testloss = criterion_cl(test_out.permute(0, 2, 3, 1).contiguous().view(-1, 2), test_targets.view(-1))+\
            criterion(s1.permute(0, 2, 3, 1).contiguous().view(-1, 2), test_targets.view(-1))+\
            criterion(s2.permute(0, 2, 3, 1).contiguous().view(-1, 2), test_targets.view(-1))+\
            criterion(s3.permute(0, 2, 3, 1).contiguous().view(-1, 2), test_targets.view(-1))+\
            criterion(s4.permute(0, 2, 3, 1).contiguous().view(-1, 2), test_targets.view(-1))
            test_loss.append(testloss.item())
            
            test_ori=F.softmax(test_out,dim=1)
            test_pre = torch.ge(test_ori.data[:, 1, :, :], 0.5)
            acc_te, precision_te, recall_te, spec_te, _, dice_te = evaluation(test_targets.data, test_pre)
            
            AUC_te = computeAUC(test_targets.data, test_ori.data[:,1,:,:], pos_label=1)

            test_acc.append(acc_te)
            test_AUC.append(AUC_te)
            test_precision.append(precision_te)
            test_recall.append(recall_te)
            test_spec.append(spec_te)
            test_dice.append(dice_te)

        print('\nepoch ' + repr(epoch) + ' || test Loss: %.4f || \n' % (np.mean(test_loss)))
        writer.add_scalar('test/loss', np.mean(test_loss), epoch)
        writer.add_scalar('test/acc', np.mean(test_acc), epoch)
        writer.add_scalar('test/AUC', np.mean(test_AUC), epoch)
        writer.add_scalar('test/precision', np.mean(test_precision), epoch)
        writer.add_scalar('test/recall', np.mean(test_recall), epoch)
        writer.add_scalar('test/specificity', np.mean(test_spec), epoch)
        writer.add_scalar('test/dice_score', np.mean(test_dice), epoch)
        if epoch % 15 == 0:
            visualize(group_images(test_images.data, 5), './' + name_experiment + '/' + "test_imgs_" + str(epoch))
            visualize(group_images(test_targets.data, 5), './' + name_experiment + '/' + "test_masks_" + str(epoch))
            visualize(group_images(torch.unsqueeze(test_out.data[:, 0, :, :], 1), 5),
                      './' + name_experiment + '/' + "test_pred_" + str(epoch))
        '''


def adjust_learning_rate(optimizer, gamma, step,epoch):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    #lr = learning_rate * (gamma ** (step))
    #lr=1e-4*(-2/3*epoch+7)
    if epoch==3:
        lr=5*1e-4
    elif epoch==6:
        lr=3*1e-4
    else:
        lr=1e-4
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        xavier(m.weight.data)
        # m.bias.data.zero_()


def constant_init(m):
    if isinstance(m, nn.Conv2d):
        init.constant_(m.weight.data, 0.01)
        m.bias.data.zero_()


def gaussian_init(m):
    if isinstance(m, nn.Conv2d):
        init.normal_(m.weight.data, mean=0, std=0.001)
        m.bias.data.zero_()


if __name__ == '__main__':
    train()



