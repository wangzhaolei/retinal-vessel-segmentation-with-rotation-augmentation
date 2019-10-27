# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class soft(nn.Module):
    def __init__(self,size_average=True):
        super(soft, self).__init__()
        self.size_average = size_average

    def forward(self, inputs, targets):
        #inputs=inputs.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        #targets=targets.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        #N = inputs.size(0)
        #C = inputs.size(1)
        P = F.softmax(inputs,dim=1)

        soft_targets=F.softmax(targets,dim=1) ###N*C

        batch_loss = (-P.log()*soft_targets).sum(1).view(-1,1)
    
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class dilation_loss(nn.Module):
    """docstring for thin_mid_vessel_loss
    outputs's shape (N*H*W,2) 2 is classnumbers
    targets's shape (N*H*W)
    thin_mask's shape (N*H*W) is the thin vessel mask

    """
    def __init__(self):
        super(dilation_loss, self).__init__()
    def forward(self,outputs,targets,dilation_mask):
        N = outputs.size(0)  #all pixels
        C = outputs.size(1)  #number of class
        P = F.softmax(outputs,dim=1) #(N,C)

        class_mask = outputs.data.new(N, C).fill_(0)  #(N,C)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        ####thin mask seperate the vessel
        P_thin=P[dilation_mask==1]   #(-1,2)
        target_thin=class_mask[dilation_mask==1] #(-1,2)

        #####for thin
        #print('P_thin shape:',P_thin.size())
        #print("target_thin shape:",target_thin.size())
        if P_thin.size()[0]==0:
            thin_loss=0
        else:
            probs_thin = (P_thin*target_thin).sum(1)  #(-1)
            thin_loss = (-probs_thin.log()).sum()

        loss = thin_loss/N
        
        return loss
class thin_vessel_loss(nn.Module):
    """docstring for thin_mid_vessel_loss
    outputs's shape (N*H*W,2) 2 is classnumbers
    targets's shape (N*H*W)
    thin_mask's shape (N*H*W) is the thin vessel mask

    """
    def __init__(self,thin_weight):
        super(thin_vessel_loss, self).__init__()
        self.thin_weight=thin_weight
       
    def forward(self,outputs,targets,thin_mask):
        N = outputs.size(0)  #all pixels
        C = outputs.size(1)  #number of class
        P = F.softmax(outputs,dim=1) #(N,C)

        class_mask = outputs.data.new(N, C).fill_(0)  #(N,C)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        ####thin mask seperate the vessel
        P_thin=P[thin_mask==1]   #(-1,2)
        target_thin=class_mask[thin_mask==1] #(-1,2)

        P_mid_Back=P[thin_mask==0]  #(-1,2)
        target_mid=class_mask[thin_mask==0] #(-1,2)
        #assert ((P_thin.size()[0]+P_mid_Back.size()[0])==N)
        #assert ((target_thin.size()[0]+target_mid.size()[0])==N)
        #print('1',P_thin.size()[0])
        #print('2',P_mid_Back.size()[0])
        #print('3',target_thin.size()[0])
        #print('4',target_mid.size()[0])
        #print(target_thin)
        '''
        if outputs.is_cuda and not self.thin_weight.is_cuda:
            self.thin_weight = self.thin_weight.cuda()
        thin = self.thin_weight[targets[thin_mask==1].data.view(-1)]#(-1)
        '''
        #print(thin.size())
        #print(thin)

        #####for thin
        #print('P_thin shape:',P_thin.size())
        #print("target_thin shape:",target_thin.size())
        if P_thin.size()[0]==0:
            thin_loss=0
        else:
            probs_thin = (P_thin*target_thin).sum(1)  #(-1)
            thin_loss = (-self.thin_weight*(probs_thin.log())).sum()

        #####for mid
        probs_mid = (P_mid_Back*target_mid).sum(1)
        mid_loss = -(probs_mid.log())

        loss = (thin_loss+mid_loss.sum())/N
        
        return loss

class thin_mid_vessel_loss(nn.Module):
    """docstring for thin_mid_vessel_loss
    outputs's shape (N*H*W,2) 2 is classnumbers
    targets's shape (N*H*W)
    thin_mask's shape (N*H*W) is the thin vessel mask

    """
    def __init__(self,thin_weight,mid_weight):
        super(thin_mid_vessel_loss, self).__init__()
        self.thin_weight=Variable(torch.Tensor([1,thin_weight]))
        self.mid_weight=Variable(torch.Tensor([1,mid_weight]))
    def forward(self,outputs,targets,thin_mask):
        N = outputs.size(0)  #all pixels
        C = outputs.size(1)  #number of class
        P = F.softmax(outputs,dim=1) #(N,C)

        class_mask = outputs.data.new(N, C).fill_(0)  #(N,C)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        ####thin mask seperate the vessel
        P_thin=P[thin_mask==1]   #(-1,2)
        target_thin=class_mask[thin_mask==1] #(-1,2)

        P_mid_Back=P[thin_mask==0]  #(-1,2)
        target_mid=class_mask[thin_mask==0] #(-1,2)

        if outputs.is_cuda and not self.thin_weight.is_cuda and not self.mid_weight.is_cuda:
            self.thin_weight = self.thin_weight.cuda()
            self.mid_weight=self.mid_weight.cuda()
        thin = self.thin_weight[targets[thin_mask==1].data.view(-1)]#(-1)
        mid=self.mid_weight[targets[thin_mask==0].data.view(-1)]    #(-1)

        #####for thin
        #print('P_thin shape:',P_thin.size())
        #print("target_thin shape:",target_thin.size())
        if P_thin.size()[0]==0:
            thin_loss=0
        else:
            probs_thin = (P_thin*target_thin).sum(1)  #(-1)
            thin_loss = (-thin*(probs_thin.log())).sum()

        #####for mid
        probs_mid = (P_mid_Back*target_mid).sum(1)
        mid_loss = -mid*(probs_mid.log())

        loss = (thin_loss+mid_loss.sum())/N
        
        return loss



class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, outputs, targets):
        smooth = 1.
        outputs=F.softmax(outputs,dim=-1)
        iflat = outputs[:,1].view(-1)
        intersection = (iflat * targets).sum()
        return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + targets.sum() + smooth))
    
class CrossEntropyLoss2D(nn.Module):
    def __init__(self, weights,size_average=True):
        super(CrossEntropyLoss2D, self).__init__()
        self.nll_loss_2d = nn.NLLLoss(weight=weights,size_average=size_average)

    def forward(self, outputs, targets):
        return self.nll_loss_2d(F.log_softmax(outputs,dim=-1), targets)

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classi?ed examples (p > .5), 
                                   putting more focus on hard, misclassi?ed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    return y[labels]            # [N,D]

