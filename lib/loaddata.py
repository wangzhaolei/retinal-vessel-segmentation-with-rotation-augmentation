import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import numpy as np
import os.path
def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x
class DRIVE_train_dataset(data.Dataset):
    """
    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,train_file,flip):
        
        self._annopath = osp.join('%s','masks', '%s.npy')
        self._imgpath = osp.join('%s','imgs', '%s.npy')
        self.flip=flip
        self.ids = list()
        for line in open(root+'/'+train_file):
            self.ids.append((root, line.rstrip('\n')))

    def __len__(self):
        #print(len(self.ids))
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        
        target = np.load(self._annopath % img_id)#(1,48,48)
        img =np.load(self._imgpath % img_id)     #(1,48,48)
        #height, width= img.shape
        if self.flip==True:
            if np.random.random() < 0.5:
                img = flip_axis(img, 1)
                target = flip_axis(target, 1)
            if np.random.random() < 0.5:
                img = flip_axis(img, 0)
                target = flip_axis(target, 0)
        #target=target[np.newaxis,:,:] #(1,48,48)
        #img=img[np.newaxis,:,:]

        return torch.from_numpy(img.copy()),torch.from_numpy(target.copy())
        # return torch.from_numpy(img), target, height, width

class DRIVE_test_dataset(data.Dataset):
    """
    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root):
        
        self._imgpath = osp.join('%s','imgs', '%s.npy')
        self.ids = list()
        for line in open(root+'/test_ids.txt'):
            self.ids.append((root, line.rstrip('\n')))

    def __len__(self):
        #print(len(self.ids))
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]

        img =np.load(self._imgpath % img_id)     #(48,48)
        #height, width= img.shape

        img=img[np.newaxis,:,:]

        return torch.from_numpy(img.copy())
        # return torch.from_numpy(img), target, height, width
