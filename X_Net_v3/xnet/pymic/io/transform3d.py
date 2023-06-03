# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import json
import random
import numpy as np
import skimage

from scipy import ndimage
from pymic.util.image_process import *
from pymic.Toolkit.deform import *

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple/list or int): Desired output size. 
            If tuple/list, output_size should in the format of [D, H, W] or [H, W].
            Channel number is kept the same as the input. If D is None, the input image
            is only reslcaled in 2D.
            If int, the smallest axis is matched to output_size keeping 
            aspect ratio the same.
    """

    def __init__(self, output_size, inverse = True):
        assert isinstance(output_size, (int, list, tuple))
        self.output_size = output_size
        self.inverse = inverse

    def __call__(self, sample):
        image_ct = sample['image_ct']
        image_mr = sample['image_mr']
        input_shape = image_ct.shape
        input_dim   = len(input_shape) - 1

        if isinstance(self.output_size, (list, tuple)):
            output_size = self.output_size
            if(output_size[0] is None):
                output_size[0] = input_shape[1]
            assert(len(output_size) == input_dim)
        else:
            min_edge = min(input_shape[1:])
            output_size = [self.output_size * input_shape[i+1] / min_edge for \
                            i in range(input_dim)]
        scale = [(output_size[i] + 0.0)/input_shape[1:][i] for i in range(input_dim)]
        scale = [1.0] + scale
        image_ct = ndimage.interpolation.zoom(image_ct, scale, order = 1)
        image_mr = ndimage.interpolation.zoom(image_mr, scale, order = 1)

        sample['image_ct'] = image_ct
        sample['image_mr'] = image_mr
        sample['Rescale_origin_shape'] = json.dumps(input_shape)
        if('label_ct' in sample):
            label_ct = sample['label_ct']
            label_ct = ndimage.interpolation.zoom(label_ct, scale, order = 0)
            sample['label_ct'] = label_ct
            label_mr = sample['label_mr']
            label_mr = ndimage.interpolation.zoom(label_mr, scale, order = 0)
            sample['label_mr'] = label_mr
        
        return sample

    def inverse_transform_for_prediction(self, sample):
        ''' rescale sample['predict'] (5D or 4D) to the original spatial shape.
         assume batch size is 1, otherwise scale may be different for 
         different elemenets in the batch.

        origin_shape is a 4D or 3D vector as saved in __call__().'''
        origin_shape = json.loads(sample['Rescale_origin_shape'][0])
        origin_dim   = len(origin_shape) - 1
        predict_ct = sample['predict_ct']
        predict_mr = sample['predict_mr']
        input_shape = predict_ct.shape
        scale = [(origin_shape[1:][i] + 0.0)/input_shape[2:][i] for \
                i in range(origin_dim)]
        scale = [1.0, 1.0] + scale

        output_predict = ndimage.interpolation.zoom(predict_ct, scale, order = 1)
        sample['predict_ct'] = output_predict
        output_predict = ndimage.interpolation.zoom(predict_mr, scale, order = 1)
        sample['predict_mr'] = output_predict
        return sample

class RandomScale(object):
    """Rescale the image according the given scale.

    Args:
        scale interval:uaually between [0.8,1.2]
    """

    def __init__(self, interrange):
        self.interrange = interrange

    def __call__(self, sample):
        image_ct = sample['image_ct']
        image_mr = sample['image_mr']
        array_image_ct = np.squeeze(image_ct)
        array_image_mr = np.squeeze(image_mr)
        scale = np.random.uniform(low=self.interrange[0], high=self.interrange[1], size=None)
        coords = create_zero_centered_coordinate_mesh(array_image_ct.shape)
        coords = scale_coords(coords, scale)
        for d in range(len(array_image_ct.shape)):
            ctr = int(np.round(array_image_ct.shape[d] / 2.))
            coords[d] += ctr
        ret_ct = map_coordinates(array_image_ct, coords, order=1, \
                mode='constant', cval=0.0).astype(array_image_ct.dtype)
        for d in range(len(array_image_mr.shape)):
            ctr = int(np.round(array_image_mr.shape[d] / 2.))
            coords[d] += ctr
        ret_mr = map_coordinates(array_image_mr, coords, order=1, \
                mode='constant', cval=0.0).astype(array_image_mr.dtype)
        image_ct = np.expand_dims(ret_ct, axis = 0)
        image_mr = np.expand_dims(ret_mr, axis = 0)
        sample['image_ct'] = image_ct
        sample['image_mr'] = image_ct

        if('label_ct' in sample):
            label = sample['label_ct']
            label = np.squeeze(label)
            label = map_coordinates(label, coords, order=3, \
                mode='constant', cval=0.0).astype(label.dtype)
            label = np.expand_dims(label, axis = 0)
            sample['label_ct'] = label
            label = sample['label_mr']
            label = np.squeeze(label)
            label = map_coordinates(label, coords, order=3, \
                mode='constant', cval=0.0).astype(label.dtype)
            label = np.expand_dims(label, axis = 0)
            sample['label_mr'] = label
        
        return sample

class RandomFlip(object):
    """
    random flip the image (shape [C, D, H, W] or [C, H, W]) 
    Args:
        flip_depth (bool) : random flip along depth axis or not, only used for 3D images
        flip_height (bool): random flip along height axis or not
        flip_width (bool) : random flip along width axis or not
    """
    def __init__(self, flip_depth, flip_height, flip_width, inverse):
        self.flip_depth  = flip_depth
        self.flip_height = flip_height
        self.flip_width  = flip_width
        self.inverse = inverse

    def __call__(self, sample):
        image_ct = sample['image_ct']
        image_mr = sample['image_mr']
        input_shape = image_ct.shape
        input_dim = len(input_shape) - 1
        flip_axis = []
        if(self.flip_width):
            if(random.random() > 0.5):
                flip_axis.append(-1)
        if(self.flip_height):
            if(random.random() > 0.5):
                flip_axis.append(-2)
        if(input_dim == 3 and self.flip_depth):
            if(random.random() > 0.5):
                flip_axis.append(-3)
        if(len(flip_axis) > 0):
            # use .copy() to avoid negative strides of numpy array
            # current pytorch does not support negative strides
            sample['image_ct'] = np.flip(image_ct, flip_axis).copy()
            sample['image_mr'] = np.flip(image_mr, flip_axis).copy()
        
        sample['RandomFlip_Param'] = json.dumps(flip_axis)
        if len(flip_axis) > 0:
            sample['label_ct'] = np.flip(sample['label_ct'] , flip_axis).copy()
            sample['label_mr'] = np.flip(sample['label_mr'] , flip_axis).copy()
        
        return sample

    def  inverse_transform_for_prediction(self, sample):
        ''' flip sample['predict'] (5D or 4D) to the original direction.
         assume batch size is 1, otherwise flip parameter may be different for 
         different elemenets in the batch.

        flip_axis is a list as saved in __call__().'''
        flip_axis = json.loads(sample['RandomFlip_Param'][0]) 
        if(len(flip_axis) > 0):
            sample['predict_ct']  = np.flip(sample['predict_ct'] , flip_axis)
            sample['predict_mr']  = np.flip(sample['predict_mr'] , flip_axis)
        return sample

class RandomRotate(object):
    """
    random rotate the image (shape [C, D, H, W] or [C, H, W]) 
    Args:
        angle_range_d (tuple/list/None) : rorate angle range along depth axis (degree),
               only used for 3D images
        angle_range_h (tuple/list/None) : rorate angle range along height axis (degree)
        angle_range_w (tuple/list/None) : rorate angle range along width axis (degree)
    """
    def __init__(self, angle_range_d, angle_range_h, angle_range_w, inverse):
        self.angle_range_d  = angle_range_d
        self.angle_range_h  = angle_range_h
        self.angle_range_w  = angle_range_w
        self.inverse = inverse

    def __apply_transformation(self, image, transform_param_list, order = 1):
        """
        apply rotation transformation to an ND image
        Args:
            image (nd array): the input nd image
            transform_param_list (list): a list of roration angle and axes
            order (int): interpolation order
        """
        for angle, axes in transform_param_list:
            image = ndimage.rotate(image, angle, axes, reshape = False, order = order)
        return image

    def __call__(self, sample):
        image_ct = sample['image_ct']
        image_mr = sample['image_mr']
        input_shape = image_ct.shape
        input_dim = len(input_shape) - 1
        
        transform_param_list = []
        if(self.angle_range_d is not None):
            angle_d = np.random.uniform(self.angle_range_d[0], self.angle_range_d[1])
            transform_param_list.append([angle_d, (-1, -2)])
        if(input_dim == 3):
            if(self.angle_range_h is not None):
                angle_h = np.random.uniform(self.angle_range_h[0], self.angle_range_h[1])
                transform_param_list.append([angle_h, (-1, -3)])
            if(self.angle_range_w is not None):
                angle_w = np.random.uniform(self.angle_range_w[0], self.angle_range_w[1])
                transform_param_list.append([angle_w, (-2, -3)])
        assert(len(transform_param_list) > 0)

        sample['image_ct'] = self.__apply_transformation(image_ct, transform_param_list, 1)
        sample['image_mr'] = self.__apply_transformation(image_mr, transform_param_list, 1)
        sample['RandomRotate_Param'] = json.dumps(transform_param_list)
        if('label_ct' in sample ):
            sample['label_ct'] = self.__apply_transformation(sample['label_ct'] , 
                                transform_param_list, 0)
        if('label_mr' in sample ):
            sample['label_mr'] = self.__apply_transformation(sample['label_mr'] , 
                                transform_param_list, 0)
        return sample

    def  inverse_transform_for_prediction(self, sample):
        ''' rorate sample['predict'] (5D or 4D) to the original direction.
        assume batch size is 1, otherwise rotate parameter may be different for 
        different elemenets in the batch.

        transform_param_list is a list as saved in __call__().'''
        # get the paramters for invers transformation
        transform_param_list = json.loads(sample['RandomRotate_Param'][0]) 
        transform_param_list.reverse()
        for i in range(len(transform_param_list)):
            transform_param_list[i][0] = - transform_param_list[i][0]
        sample['predict_ct'] = self.__apply_transformation(sample['predict_ct'] , 
                                transform_param_list, 1)
        sample['predict_mr'] = self.__apply_transformation(sample['predict_mr'] , 
                                transform_param_list, 1)
        return sample

class Pad(object):
    """
    Pad the image (shape [C, D, H, W] or [C, H, W]) to an new spatial shape, 
    the real output size will be max(image_size, output_size)
    Args:
       output_size (tuple/list): the size along each spatial axis. 
       
    """
    def __init__(self, output_size, inverse = True):
        self.output_size = output_size
        self.inverse = inverse


    def __call__(self, sample):
        image_ct = sample['image_ct']
        image_mr = sample['image_mr']
        input_shape = image_ct.shape
        input_dim = len(input_shape) - 1
        assert(len(self.output_size) == input_dim)
        margin = [max(0, self.output_size[i] - input_shape[1+i]) \
            for i in range(input_dim)]

        margin_lower = [int(margin[i] / 2) for i in range(input_dim)]
        margin_upper = [margin[i] - margin_lower[i] for i in range(input_dim)]
        pad = [(margin_lower[i], margin_upper[i]) for  i in range(input_dim)]
        pad = tuple([(0, 0)] + pad)
        if(max(margin) > 0):
            image_ct = np.pad(image_ct, pad, 'reflect')   
            image_mr = np.pad(image_mr, pad, 'reflect')   

        sample['image_ct'] = image_ct
        sample['image_mr'] = image_mr
        sample['Pad_Param'] = json.dumps((margin_lower, margin_upper))
        
        if('label_ct' in sample):
            label_ct = sample['label_ct']
            label_mr = sample['label_mr']
            if(max(margin) > 0):
                label_ct = np.pad(label_ct, pad, 'reflect')
                label_mr = np.pad(label_mr, pad, 'reflect')
            sample['label_ct'] = label_ct
            sample['label_mr'] = label_mr
        
        return sample
    
    def inverse_transform_for_prediction(self, sample):
        ''' crop sample['predict'] (5D or 4D) to the original spatial shape.
         assume batch size is 1, otherwise scale may be different for 
         different elemenets in the batch.

        origin_shape is a 4D or 3D vector as saved in __call__().'''
        # raise ValueError("not implemented")
        params = json.loads(sample['Pad_Param'][0]) 
        margin_lower = params[0]
        margin_upper = params[1]
        predict_ct = sample['predict_ct']
        predict_mr = sample['predict_mr']
        predict_shape = predict_ct.shape
        crop_min = [0, 0] + margin_lower
        crop_max = [predict_shape[2:][i] - margin_upper[i] \
            for i in range(len(margin_lower))]
        crop_max = list(predict_shape[:2]) + crop_max

        output_predict_ct = crop_ND_volume_with_bounding_box(predict_ct, crop_min, crop_max)
        output_predict_mr = crop_ND_volume_with_bounding_box(predict_mr, crop_min, crop_max)
        sample['predict_ct'] = output_predict_ct
        sample['predict_mr'] = output_predict_mr
        return sample

class CropWithBoundingBox(object):
    """Crop the image (shape [C, D, H, W] or [C, H, W]) based on bounding box

    Args:
        start (None or tuple/list): The start index along each spatial axis.
            if None, calculate the start index automatically so that 
            the cropped region is centered at the non-zero region.
        output_size (None or tuple/list): Desired spatial output size.
            if None, set it as the size of bounding box of non-zero region 
    """

    def __init__(self, start, output_size, inverse = True):
        self.start = start
        self.output_size = output_size
        self.inverse = inverse


        
    def __call__(self, sample):
        image_ct = sample['image_ct']
        image_mr = sample['image_mr']
        input_shape = image_ct.shape
        input_dim   = len(input_shape) - 1
        bb_min, bb_max = get_ND_bounding_box(image)
        bb_min, bb_max = bb_min[1:], bb_max[1:]

        if(self.start is None):
            if(self.output_size is None):
                crop_min = bb_min, crop_max = bb_max
            else:
                assert(len(self.output_size) == input_dim)
                crop_min = [int((bb_min[i] + bb_max[i] + 1)/2) - int(self.output_size[i]/2) \
                    for i in range(input_dim)]
                crop_min = [max(0, crop_min[i]) for i in range(input_dim)]
                crop_max = [crop_min[i] + self.output_size[i] for i in range(input_dim)]
        else:
            assert(len(self.start) == input_dim)
            crop_min = self.start
            if(self.output_size is None):
                assert(len(self.output_size) == input_dim)
                crop_max = [crop_min[i] + bb_max[i] - bb_min[i] \
                    for i in range(input_dim)]
            else:
                crop_max =  [crop_min[i] + self.output_size[i] for i in range(input_dim)]
        crop_min = [0] + crop_min
        crop_max = list(input_shape[0:1]) + crop_max
        image_ct = crop_ND_volume_with_bounding_box(image_ct, crop_min, crop_max)
        image_mr = crop_ND_volume_with_bounding_box(image_mr, crop_min, crop_max)
        
        sample['image_ct'] = image_ct
        sample['image_mr'] = image_mr
        sample['CropWithBoundingBox_Param'] = json.dumps((input_shape, crop_min, crop_max))
        if('label_ct' in sample):
            label_ct = sample['label_ct']
            label_mr = sample['label_mr']
            crop_max[0] = label.shape[0]
            label_ct = crop_ND_volume_with_bounding_box(label_ct, crop_min, crop_max)
            label_mr = crop_ND_volume_with_bounding_box(label_mr, crop_min, crop_max)
            sample['label_ct'] = label_ct
            sample['label_mr'] = label_mr

        return sample

    def inverse_transform_for_prediction(self, sample):
        ''' rescale sample['predict'] (5D or 4D) to the original spatial shape.
         assume batch size is 1, otherwise scale may be different for 
         different elemenets in the batch.

        origin_shape is a 4D or 3D vector as saved in __call__().'''
        params = json.loads(sample['CropWithBoundingBox_Param'][0]) 
        origin_shape = params[0]
        crop_min     = params[1]
        crop_max     = params[2]
        predict_ct = sample['predict_ct']
        predict_mr = sample['predict_mr']
        origin_shape   = list(predict_ct.shape[:2]) + origin_shape[1:]
        output_predict_ct = np.zeros(origin_shape, predict_ct.dtype)
        output_predict_mr = np.zeros(origin_shape, predict_mr.dtype)
        crop_min = [0, 0] + crop_min[1:]
        crop_max_ct = list(predict_ct.shape[:2]) + crop_max[1:]
        crop_max_mr = list(predict_mr.shape[:2]) + crop_max[1:]
        output_predict_ct = set_ND_volume_roi_with_bounding_box_range(output_predict_ct,
            crop_min, crop_max, predict_ct)
        output_predict_mr = set_ND_volume_roi_with_bounding_box_range(output_predict_mr,
            crop_min, crop_max, predict_mr)

        sample['predict_ct'] = output_predict_ct
        sample['predict_mr'] = output_predict_mr
        return sample

class RandomCrop(object):
    """Randomly crop the input image (shape [C, D, H, W] or [C, H, W]) 

    Args:
        output_size (tuple or list): Desired output size [D, H, W] or [H, W].
            the output channel is the same as the input channel.
    """

    def __init__(self, output_size, inverse = True):
        assert isinstance(output_size, (list, tuple))
        self.output_size = output_size
        self.inverse = inverse

    def __call__(self, sample):
        image_ct = sample['image_ct']
        image_mr = sample['image_mr']
        input_shape = image_ct.shape
        input_dim   = len(input_shape) - 1

        assert(input_dim == len(self.output_size))
        crop_margin = [input_shape[i + 1] - self.output_size[i]\
            for i in range(input_dim)]
        crop_min = [random.randint(0, item) for item in crop_margin]
        crop_max = [crop_min[i] + self.output_size[i] \
            for i in range(input_dim)]
        crop_min = [0] + crop_min
        crop_max = list(input_shape[0:1]) + crop_max
        image_ct = crop_ND_volume_with_bounding_box(image_ct, crop_min, crop_max)
        image_mr = crop_ND_volume_with_bounding_box(image_mr, crop_min, crop_max)
       
        sample['image_ct'] = image_ct
        sample['image_mr'] = image_mr
        sample['RandomCrop_Param'] = json.dumps((input_shape, crop_min, crop_max))
        if('label_ct' in sample):
            label_ct = sample['label_ct']
            crop_max[0] = label_ct.shape[0]
            label_ct = crop_ND_volume_with_bounding_box(label_ct, crop_min, crop_max)
            sample['label_ct'] = label_ct
            label_mr = sample['label_mr']
            crop_max[0] = label_mr.shape[0]
            label_mr = crop_ND_volume_with_bounding_box(label_mr, crop_min, crop_max)
            sample['label_mr'] = label_mr
        return sample

    def inverse_transform_for_prediction(self, sample):
        ''' rescale sample['predict'] (5D or 4D) to the original spatial shape.
         assume batch size is 1, otherwise scale may be different for 
         different elemenets in the batch.

        origin_shape is a 4D or 3D vector as saved in __call__().'''
        params = json.loads(sample['RandomCrop_Param'][0]) 
        origin_shape = params[0]
        crop_min     = params[1]
        crop_max     = params[2]
        predict_ct = sample['predict_ct']
        predict_mr = sample['predict_mr']
        origin_shape   = list(predict_ct.shape[:2]) + origin_shape[1:]
        output_predict_ct = np.zeros(origin_shape, predict_ct.dtype)
        output_predict_mr = np.zeros(origin_shape, predict_mr.dtype)
        crop_min = [0, 0] + crop_min[1:]
        crop_max = list(predict_ct.shape[:2]) + crop_max[1:]
        output_predict_ct = set_ND_volume_roi_with_bounding_box_range(output_predict_ct,
            crop_min, crop_max, predict_ct)
        output_predict_mr = set_ND_volume_roi_with_bounding_box_range(output_predict_mr,
            crop_min, crop_max, predict_mr)

        sample['predict_ct'] = output_predict_ct
        sample['predict_mr'] = output_predict_mr
        return sample


class ChannelWiseGammaCorrection(object):
    """
    apply gamma correction to each channel
    """
    def __init__(self, gamma_min, gamma_max, inverse = False):
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.inverse = inverse
    
    def __call__(self, sample):
        image_ct = sample['image_ct']
        image_mr = sample['image_mr']
        for chn in range(image_ct.shape[0]):
            gamma_c = random.random() * (self.gamma_max - self.gamma_min) + self.gamma_min
            img_c = image_ct[chn]
            v_min = img_c.min()
            v_max = img_c.max()
            img_c = (img_c - v_min)/(v_max - v_min)
            img_c = np.power(img_c, gamma_c)*(v_max - v_min) + v_min
            image_ct[chn] = img_c

        sample['image_ct'] = image_ct
        
        for chn in range(image_mr.shape[0]):
            gamma_c = random.random() * (self.gamma_max - self.gamma_min) + self.gamma_min
            img_c = image_mr[chn]
            v_min = img_c.min()
            v_max = img_c.max()
            img_c = (img_c - v_min)/(v_max - v_min)
            img_c = np.power(img_c, gamma_c)*(v_max - v_min) + v_min
            image_mr[chn] = img_c

        sample['image_mr'] = image_mr
        return sample
    
    def inverse_transform_for_prediction(self, sample):
        raise(ValueError("not implemented"))

class ChannelWiseNormalize(object):
    """Nomralize the image (shape [C, D, H, W] or [C, H, W]) for each channel

    Args:
        mean (None or tuple/list): The mean values along each channel.
        std  (None or tuple/list): The std values along each channel.
            if mean and std are None, calculate them from non-zero region
        zero_to_random (bool): If true, replace zero values with random values.
    """
    def __init__(self, mean, std, zero_to_random = False, inverse = False):
        self.mean = mean
        self.std  = std
        self.zero_to_random = zero_to_random
        self.inverse = inverse

    def __call__(self, sample):
        image= sample['image']
        mask = image[0] > 0
        for chn in range(image.shape[0]):
            if(self.mean is None and self.std is None):
                pixels = image[chn][mask > 0]
                chn_mean = pixels.mean()
                chn_std  = pixels.std()
            else:
                chn_mean = self.mean[chn]
                chn_std  = self.std[chn]
            chn_norm = (image[chn] - chn_mean)/chn_std
            if(self.zero_to_random):
                chn_random = np.random.normal(0, 1, size = chn_norm.shape)
                chn_norm[mask == 0] = chn_random[mask == 0]
            image[chn] = chn_norm

        sample['image'] = image
        return sample

    def inverse_transform_for_prediction(self, sample):
        raise(ValueError("not implemented"))

class ChannelWiseThreshold(object):
    """Threshold the image (shape [C, D, H, W] or [C, H, W]) for each channel

    Args:
        threshold (tuple/list): The threshold value along each channel.
    """
    def __init__(self, threshold, inverse = False):
        self.threshold = threshold
        self.inverse = inverse

    def __call__(self, sample):
        image= sample['image']
        for chn in range(image.shape[0]):
            mask = np.asarray(image[chn] > self.threshold[chn], image.dtype)
            image[chn] = mask * image[chn]

        sample['image'] = image
        return sample

    def inverse_transform_for_prediction(self, sample):
        raise(ValueError("not implemented"))

class LabelConvert(object):
    """ Convert a list of labels to another list
    Args:
        source_list (tuple/list): A list of labels to be converted
        target_list (tuple/list): The target label list
    """
    def __init__(self, source_list, target_list, inverse = False):
        self.source_list = source_list
        self.target_list = target_list
        self.inverse = inverse
        assert(len(source_list) == len(target_list))
    
    def __call__(self, sample):
        label_ct = sample['label_ct']
        label_mr = sample['label_mr']
        label_converted_ct = convert_label(label_ct, self.source_list, self.target_list)
        label_converted_mr = convert_label(label_mr, self.source_list, self.target_list)
        sample['label_ct'] = label_converted_ct
        sample['label_mr'] = label_converted_mr
        return sample
    
    def inverse_transform_for_prediction(self, sample):
        raise(ValueError("not implemented"))

class RegionSwop(object):
    """
    Swop a subregion randomly between two images and their corresponding label
    Args:
        axes: the list of possible specifed spatial axis for swop, 
              if None, then it is all the spatial axes
        prob: the possibility of use region swop

    """
    def __init__(self, spatial_axes = None, probility = 0.5, inverse = False):
        self.axes = spatial_axes
        self.prob = probility
        self.inverse = inverse
    
    def __call__(self, sample):
        # image shape is like [B, C, D, H, W]
        img_ct = sample['image_ct']
        img_mr = sample['image_mr']
        [B, C, D, H, W] = img_ct.shape
        if(B < 2):
            return sample
        swop_flag = random.random() < self.prob
        if(swop_flag):
            swop_axis = random.sample(self.axes, 1)[0]
            ratio = random.random()
            roi_min = [0, 0, 0, 0]
            if(swop_axis == 0):
                d = int(D*ratio)
                roi_max = [C, d, H, W]
            elif(swop_axis == 1):
                h = int(H*ratio)
                roi_max = [C, D, h, W]
            else:
                w = int(W*ratio)
                roi_max = [C, D, H, w]
            img_sub0_ct = crop_ND_volume_with_bounding_box(img_ct[0], roi_min, roi_max)
            img_sub1_ct = crop_ND_volume_with_bounding_box(img_ct[1], roi_min, roi_max)
            img_sub0_mr = crop_ND_volume_with_bounding_box(img_mr[0], roi_min, roi_max)
            img_sub1_mr = crop_ND_volume_with_bounding_box(img_mr[1], roi_min, roi_max)
            img_ct[0] = set_ND_volume_roi_with_bounding_box_range(img_ct[0], roi_min, roi_max, img_sub1_ct)
            img_ct[1] = set_ND_volume_roi_with_bounding_box_range(img_ct[1], roi_min, roi_max, img_sub0_ct)
            img_mr[0] = set_ND_volume_roi_with_bounding_box_range(img_mr[0], roi_min, roi_max, img_sub1_mr)
            img_mr[1] = set_ND_volume_roi_with_bounding_box_range(img_mr[1], roi_min, roi_max, img_sub0_mr)
            sample['image_ct'] = img_ct
            sample['image_mr'] = img_mr
            if('label_ct' in sample):
                label_ct = sample['label_ct']
                roi_max[0] = label_ct.shape[1]
                lab_sub0_ct = crop_ND_volume_with_bounding_box(label_ct[0], roi_min, roi_max)
                lab_sub1_ct = crop_ND_volume_with_bounding_box(label_ct[1], roi_min, roi_max)
                label_ct[0] = set_ND_volume_roi_with_bounding_box_range(label_ct[0], roi_min, roi_max, lab_sub1_ct)
                label_ct[1] = set_ND_volume_roi_with_bounding_box_range(label_ct[1], roi_min, roi_max, lab_sub0_ct)
                sample['label_ct'] = label_ct
                label_mr = sample['label_mr']
                roi_max[0] = label_mr.shape[1]
                lab_sub0_mr = crop_ND_volume_with_bounding_box(label_mr[0], roi_min, roi_max)
                lab_sub1_mr = crop_ND_volume_with_bounding_box(label-mr[1], roi_min, roi_max)
                label_mr[0] = set_ND_volume_roi_with_bounding_box_range(label_mr[0], roi_min, roi_max, lab_sub1_mr)
                label_mr[1] = set_ND_volume_roi_with_bounding_box_range(label_mr[1], roi_min, roi_max, lab_sub0_mr)
                sample['label_mr'] = label_mr
        return sample

class Nooperating(object):
    """Do nothing
    """

    def __init__(self, donothing, inverse = False):
        self.donothing = donothing
        self.inverse = inverse

    def __call__(self, sample):
        image_ct = sample['image_ct']
        image_mr = sample['image_mr']
        sample['image_ct'] = image_ct
        sample['image_mr'] = image_mr
        if('label_ct' in sample):
            label_ct = sample['label_ct']
            label_mr = sample['label_mr']
            sample['label_ct'] = label_ct
            sample['label_mr'] = label_mr
        return sample
    

    
def get_transform(name, params):
    if (name == "CropWithBoundingBox"):
        start = params["CropWithBoundingBox_start".lower()]
        output_size = params["CropWithBoundingBox_output_size".lower()]
        inverse = params["CropWithBoundingBox_inverse".lower()]
        return CropWithBoundingBox(start, output_size, inverse)

    elif(name == "Rescale"):
        output_size = params["Rescale_output_size".lower()]
        inverse     = params["Rescale_inverse".lower()]
        return Rescale(output_size, inverse)
    
    elif(name == "RandomScale"):
        interrange = params["RandomScale_interrange".lower()]
        return RandomScale(interrange)

    elif(name == "Pad"):
        output_size = params["Pad_output_size".lower()]
        inverse = params["Pad_inverse".lower()]
        return Pad(output_size, inverse)

    elif(name == 'RandomCrop'): # HERE
        output_size = params['RandomCrop_output_size'.lower()]
        inverse     = params['RandomCrop_inverse'.lower()]
        return RandomCrop(output_size, inverse)
        
    elif(name == "RandomFlip"): # HERE
        flip_depth  = params["RandomFlip_flip_depth".lower()]
        flip_height = params["RandomFlip_flip_height".lower()]
        flip_width  = params["RandomFlip_flip_width".lower()]
        inverse     = params["RandomFlip_inverse".lower()]
        return RandomFlip(flip_depth, flip_height, flip_width, inverse)

    elif(name == "RandomRotate"):
        angle_range_d = params["RandomRotate_angle_range_d".lower()]
        angle_range_h = params["RandomRotate_angle_range_h".lower()]
        angle_range_w = params["RandomRotate_angle_range_w".lower()]
        inverse = params["RandomRotate_inverse".lower()]
        return RandomRotate(angle_range_d, angle_range_h, angle_range_w, inverse)


    elif(name == "ChannelWiseGammaCorrection"):
        gamma_min = params['ChannelWiseGammaCorrection_gamma_min'.lower()]
        gamma_max = params['ChannelWiseGammaCorrection_gamma_max'.lower()]
        inverse = params['ChannelWiseGammaCorrection_inverse'.lower()]
        return ChannelWiseGammaCorrection(gamma_min, gamma_max, inverse)

    elif (name == 'ChannelWiseNormalize'):
        mean = params['ChannelWiseNormalize_mean'.lower()]
        std  = params['ChannelWiseNormalize_std'.lower()]
        zero_to_random = params['ChannelWiseNormalize_zero_to_random'.lower()]
        inverse = params['ChannelWiseNormalize_inverse'.lower()]
        return ChannelWiseNormalize(mean, std, zero_to_random, inverse)
    
    elif(name == 'ChannelWiseThreshold'):
        threshold = params['ChannelWiseThreshold_threshold'.lower()]
        inverse   = params['ChannelWiseThreshold_inverse'.lower()]
        return ChannelWiseThreshold(threshold, inverse)

    elif(name == 'LabelConvert'):
        source_list = params['LabelConvert_source_list'.lower()]
        target_list = params['LabelConvert_target_list'.lower()]
        inverse = params['LabelConvert_inverse'.lower()]
        return LabelConvert(source_list, target_list, inverse)
    

    elif(name == 'RegionSwop'):
        spatial_axes = params['RegionSwop_spatial_axes'.lower()]
        prob    = params['RegionSwop_probability'.lower()]
        inverse = params['RegionSwop_inverse'.lower()]
        return RegionSwop(spatial_axes, prob, inverse)

    elif(name == 'Donothing'):
        donothing = params['Donothing_do'.lower()]
        return Nooperating(donothing)

    else:
        raise ValueError("undefined transform :{0:}".format(name))
