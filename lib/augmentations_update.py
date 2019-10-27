import numpy as np
import types
from numpy import random
import scipy.ndimage as ndi
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def apply_transform(x,transform_matrix,channel_axis=0,fill_mode='nearest',cval=0.):
    """
    Apply the image transformation specified by a matrix.
    Arguments:
    x: 2D numpy array, single image.
    transform_matrix: Numpy array specifying the geometric transformation.
    channel_axis: Index of axis for channels in the input tensor.
    fill_mode: Points outside the boundaries of the input are filled according to the given mode
    (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
    cval: Value used for points outside the boundaries
    of the input if `mode='constant'`.
    Returns:
    The transformed version of the input.
    """
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [
    ndi.interpolation.affine_transform(x_channel,final_affine_matrix,final_offset,
        order=1,mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x

class random_shear(object):
    def __init__(self,intensity):
        self.intensity=intensity
    def __call__(self,x, y, row_index=0, col_index=1, channel_index=2,fill_mode='nearest', cval=0.):
        shear = np.random.uniform(-self.intensity, self.intensity)
        shear_matrix = np.array([[1, -np.sin(shear), 0],[0, np.cos(shear), 0],[0, 0, 1]])
        h, w = x.shape[row_index], x.shape[col_index]
        transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
        x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
        y = apply_transform(y, transform_matrix, channel_index, fill_mode, cval)
        return x, y

class elastic_transform(object):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
             Convolutional Neural Networks applied to Visual Document Analysis", in
             Proc. of the International Conference on Document Analysis and
             Recognition, 2003.
         Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    def __init__(self,alpha, sigma):
        self.alpha=alpha
        self.sigma=sigma

    def __call__(self,image, mask, alpha_affine=None, random_state=None):

        shape = image[:,:,0].shape
        if random_state is None:
            random_state = np.random.RandomState(None)
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
        xn = np.ndarray((584, 565, 3), dtype=np.uint8)
        yn = np.ndarray((584, 565, 1))
        for i in range(3):
            xn[..., i] = map_coordinates(image[..., i], indices, order=1, mode='reflect').reshape(shape)
        yn[..., 0] = map_coordinates(mask[..., 0], indices, order=1, mode='reflect').reshape(shape)
        return xn,yn

class random_rotation(object):
    def __init__(self,rg):
        self.rg=rg
        
    def __call__(self,image,target,row_index=0,col_index=1,channel_index=2,
        fill_mode='nearest',cval=0.):

        theta = np.pi / 180 * np.random.uniform(-self.rg, self.rg)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],[0, 0, 1]])

        h, w = image.shape[row_index], image.shape[col_index]
        transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
        image = apply_transform(image, transform_matrix, channel_index, fill_mode, cval)
        target = apply_transform(target, transform_matrix, channel_index, fill_mode, cval)
        return image, target
class random_zoom(object):
    def __init__(self, zoom_range):
        self.zoom_range = zoom_range

    def __call__(self,image, target, row_index=0, col_index=1, channel_index=2,
        fill_mode='nearest', cval=0.):
        if len(self.zoom_range) != 2:
            raise Exception('zoom_range should be a tuple or list of two floats. '
                'Received arg: ', self.zoom_range)

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx,zy = 1, 1
        else:
            zx,zy= np.random.uniform(self.zoom_range[0], self.zoom_range[1],2)


        zoom_matrix = np.array([[zx, 0, 0],[0, zy, 0],[0, 0, 1]])
        h, w = image.shape[row_index], image.shape[col_index]
        transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
        image = apply_transform(image, transform_matrix, channel_index, fill_mode, cval)
        target = apply_transform(target, transform_matrix, channel_index, fill_mode, cval)
        return image,target
class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target=None):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target

class ConvertFromInts(object):
    def __call__(self, image, target=None):
        return image.astype(np.float32), target

class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, target=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), target

class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, target=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, target

class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, target=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, target

            # max trails (50)
            for _ in range(50):
                current_image = image
                current_target=target

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]
                current_target = current_target[rect[1]:rect[3], rect[0]:rect[2]]



                return current_image, current_target

class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image,target):
        if random.randint(2):
            return image,target

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        #expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        expand_target = np.zeros(
            (int(height*ratio), int(width*ratio)),
            dtype=target.dtype)
        expand_target[int(top):int(top + height),
                     int(left):int(left + width)] =target
        target = expand_target

        return image,target

class RandomMirror(object):
    def __call__(self, image, target):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            target=target[:,::-1]
        return image, target
class Randomflip(object):
    def __call__(self,image,target):
        if random.randint(2):
            image=image[::-1,...]
            target=target[::-1,...]
        return image,target

class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class Augmentation(object):
    def __init__(self,rg,zoom_range):
        self.rg=rg
        self.zoom_range=zoom_range
        #self.intensity=intensity
        #self.alpha = alpha
        #self.sigma = sigma
        self.augment = Compose([
#            ConvertFromInts(),
            random_rotation(self.rg),
            random_zoom(self.zoom_range),
###            random_shear(self.intensity)
###            elastic_transform(self.alpha,self.sigma)
#            Randomflip(),
#            RandomSampleCrop(),
#            RandomMirror(),
#            SubtractMeans(self.mean)
        ])

    def __call__(self, img, target):
        return self.augment(img, target)


#if self.transform is not None:
#    img, target = self.transform(img, target)