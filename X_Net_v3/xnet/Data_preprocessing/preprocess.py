
from __future__ import absolute_import, print_function
import nibabel
import numpy as np
import os.path
import os
from scipy import ndimage as ndi
from scipy import ndimage
from skimage.measure import label as lb
from skimage.measure import regionprops
from skimage import morphology
from skimage.filters import roberts

def save_array_as_nifty_volume(data, filename, transpose=True):
    """
    save a numpy array as nifty image
    inputs:
        data: a numpy array with shape [Channel, Depth, Height, Width]
        filename: the ouput file name
    outputs: None
    """
    if transpose:
        data = data.transpose(2, 1, 0)
    img = nibabel.Nifti1Image(data, None)
    nibabel.save(img, filename)


def load_origin_nifty_volume_as_array(filename):
    """
    load nifty image into numpy array, and transpose it based on the [z,y,x] axis order
    The output array shape is like [Depth, Height, Width]
    inputs:
        filename: the input file name, should be *.nii or *.nii.gz
    outputs:
        data: a numpy data array
        zoomfactor:
    """
    img = nibabel.load(filename)
    pixelspacing = img.header.get_zooms()
    zoomfactor = list(pixelspacing)
    zoomfactor.reverse()
    data = img.get_data()
    data = data.transpose(2, 1, 0)
#     print(data.shape)

    return data, zoomfactor

def zoom_data(file, mode='img', zoom_factor=[1,1,1], class_number=0):
    """
    对数据进行插值并储存，
    :param data_root: 数据所在上层目录
    :param save_root: 存储的顶层目录
    :zoom_factor:   缩放倍数
    :return:
    """

    if mode =='label':
        intfile = np.int16(file)
        #zoom_file = np.int16(resize_Multi_label_to_given_shape(intfile, zoom_factor, class_number, order=2))
        zoom_file = ndimage.interpolation.zoom(file, zoom_factor, order=0)
    elif mode == 'img':
        zoom_file = ndimage.interpolation.zoom(file, zoom_factor, order=1)
    else:
        KeyError('please choose img or label mode')
    return zoom_file

def crop(file, bound):
    '''
    :param file: z, x, y
    :param bound: [min,max]
    :return:
    '''

    cropfile = file[max(bound[0][0], 0):min(bound[1][0], file.shape[0]), bound[0][1]:bound[1][1], bound[0][2]:bound[1][2]]
    return cropfile

def img_normalized(file, upthresh=0, downthresh=0, norm=True, thresh=True):
    """
    :param file: np array
    :param upthresh:
    :param downthresh:
    :param norm: norm or not
    :return:
    """
    if thresh:
        assert upthresh > downthresh
        file[np.where(file > upthresh)] = upthresh
        file[np.where(file < downthresh)] = downthresh
    if norm:
        file = (file-downthresh)/(upthresh-downthresh)
    return file

def get_bound_coordinate(file, pad=[0,0,0]):
    '''
    输出array非0区域的各维度上下界坐标+-pad
    :param file: groundtruth图,
    :param pad: 各维度扩充的大小
    :return: bound: [min,max]
    '''
    nonzeropoint = np.asarray(np.nonzero(file))   # 得到非0点坐标,输出为一个3*n的array，3代表3个维度，n代表n个非0点在对应维度上的坐标
    maxpoint = np.max(nonzeropoint, 1).tolist()
    minpoint = np.min(nonzeropoint, 1).tolist()
    for i in range(len(pad)):
        maxpoint[i] = maxpoint[i]+pad[i]
        minpoint[i] = minpoint[i]-pad[i]
    return [minpoint, maxpoint]

def get_segmented_body1(img, window_max=250, window_min=-150, window_length=0, show_body=False, znumber=0):
    '''
    将身体与外部分离出来
    '''

    mask = []

    if znumber < 40:
        radius = [13, 6]
    else:
        radius = [6, 8]


    '''
    Step 1: Convert into a binary image.二值化,为确保所定阈值通过大多数
    '''
    threshold = -600
    binary = np.where(img > threshold, 1.0, 0.0)  # threshold the image

    '''
    Step 2: Remove the blobs connected to the border of the image.
            清除边界
    '''

    '''
    Step 3: Erosion operation with a disk of radius 2. This operation is
    seperate the lung nodules attached to the blood vessels.
    腐蚀操作，以2mm为半径去除
    '''
    binary = morphology.erosion(binary, np.ones([radius[0], radius[0]]))

    '''
    Step 4: Closure operation with a disk of radius 10. This operation is
    to keep nodules attached to the lung wall.闭合运算
    '''
    binary = morphology.dilation(binary, np.ones([radius[1], radius[1]]))
    '''
    Step 5: Label the image.连通区域标记
    '''
    label_image = lb(binary)

    '''
    Step 6: Keep the labels with the largest area.保留最大区域
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 1:
        for region in regionprops(label_image):
            if region.area < areas[-1]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0

    '''
    Step 7: Fill in the small holes inside the binary mask .孔洞填充
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    '''
    Step 8: show the input image.
    '''

    '''
    Step 9: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    img[get_high_vals] = 0
    mask.append(binary)

    img[img > (window_max + window_length)] = window_max + window_length
    img[img < (window_min - window_length)] = window_min - window_length
    img = (img - window_min) / (window_max - window_min)
    img[get_high_vals] = 0

    return img, binary

def preprocessing(img_ct, label_ct, img_mr, label_mr, pixelspace_ct, pixelspace_mr, pad=[0,0,0],cropmode='small'):
    '''
    data preprocessing
    '''

    zoomfactor = [pixelspace_ct[0]/3, pixelspace_ct[1], pixelspace_ct[2]]
    img_ct = zoom_data(img_ct, mode='img', zoom_factor=zoomfactor)
    img_mr = zoom_data(img_mr, mode='img', zoom_factor=zoomfactor)
    label_ct = zoom_data(label_ct, mode='label', zoom_factor=zoomfactor)
    label_mr = zoom_data(label_mr, mode='label', zoom_factor=zoomfactor)
    # Le cropping est déjà fait héhé
    # ishape_ct = np.copy(img_ct)
    # ishape_ct = ishape_ct.shape
    # ishape_mr = np.copy(img_mr)
    # ishape_mr = ishape_mr.shape
    # a_ct = np.copy(img_ct)
    # a_mr = np.copy(img_mr)
    # np.where(a < -900,a,0)
    # e,_ = get_segmented_body1(a[0])
    # e = np.expand_dims(e,axis=0)
    # for i in range(1,a.shape[0]):
    #     c,_ = get_segmented_body1(a[i])
    #     d = np.expand_dims(c,axis=0)
    #     e = np.concatenate((e,d), 0)

    # bound = get_bound_coordinate(e, pad=pad)# [min,max]
    # newbound = [[],[]]
    # if cropmode == 'small':
    #     newbound[0] = [bound[0][0]+50,bound[0][1]+(bound[1][1]-bound[1][0])//10,bound[0][2]+(bound[1][2]-bound[0][2])//3+5]
    #     newbound[1] = [bound[1][0],bound[1][1]-(bound[1][1]-bound[1][0])//3,bound[1][2]-(bound[1][2]-bound[0][2])//3-5]
    # elif cropmode == 'middle':
    #     newbound[0] = [bound[0][0]+35,bound[0][1]+(bound[1][1]-bound[1][0])//10-5,bound[0][2]+(bound[1][2]-bound[0][2]-5)//3+5]
    #     newbound[1] = [bound[1][0],bound[1][1]-(bound[1][1]-bound[1][0])//3+10,bound[1][2]-(bound[1][2]-bound[0][2])//3]
    # else:
    #     newbound[0] = [bound[0][0]+16,bound[0][1],bound[0][2]]
    #     newbound[1] = [bound[1][0],bound[1][1],bound[1][2]]
    # newbound = [newbound[0], newbound[1]]

    # cropimg = crop(img, newbound)
    # cropimg = img_normalized(cropimg, upthresh=700, downthresh=-200, norm=True, thresh=True)
    # croplabel = crop(label, newbound)

    # if cropimg.shape[0] % 16 != 0:
    #     i = cropimg.shape[0] // 16
    #     i = i * 16
    #     cropimg = cropimg[:i,:,:]
    #     croplabel = croplabel[:i,:,:]

    # if cropimg.shape[1] % 16 != 0:
    #     i = cropimg.shape[1] // 16
    #     i = i * 16
    #     cropimg = cropimg[:,:i,:]
    #     croplabel = croplabel[:,:i,:]

    # if cropimg.shape[2] % 16 != 0:
    #     i = cropimg.shape[2] // 16
    #     i = i * 16
    #     cropimg = cropimg[:,:,:i]
    #     croplabel = croplabel[:,:,:i]

    return img_ct, label_ct, img_mr, label_mr

if __name__ == '__main__':
    '''
    Put your own path here
    '''
    imgroot_ct = '/auto/home/users/n/b/nboulang/X_Net_v3/xnet/alldataset/origindata/data_ct'
    imgroot_mr = '/auto/home/users/n/b/nboulang/X_Net_v3/xnet/alldataset/origindata/data_mr'
    labelroot_ct = '/auto/home/users/n/b/nboulang/X_Net_v3/xnet/alldataset/origindata/label_ct'
    labelroot_mr = '/auto/home/users/n/b/nboulang/X_Net_v3/xnet/alldataset/origindata/label_mr'

    # saveroot_small_ct = '/auto/home/users/n/b/nboulang/X_Net_v3/xnet/alldataset/small_scale/'
    # smalldatanewroot_ct = saveroot_small_ct+'data_ct'
    # smalllabelnewroot_ct = saveroot_small_ct+'label_ct'
    # saveroot_small_mr = '/auto/home/users/n/b/nboulang/X_Net_v3/xnet/alldataset/small_scale/'
    # smalldatanewroot_mr = saveroot_small_mr+'data_mr'
    # smalllabelnewroot_mr = saveroot_small_mr+'label_mr'

    saveroot_middle_ct = '/auto/home/users/n/b/nboulang/X_Net_v3/xnet/alldataset/middle_scale/'
    middledatanewroot_ct = saveroot_middle_ct+'data_ct'
    middlelabelnewroot_ct = saveroot_middle_ct+'label_ct'
    saveroot_middle_mr = '/auto/home/users/n/b/nboulang/X_Net_v3/xnet/alldataset/middle_scale/'
    middledatanewroot_mr = saveroot_middle_mr+'data_mr'
    middlelabelnewroot_mr = saveroot_middle_mr+'label_mr'

    # saveroot_large_ct = '/auto/home/users/n/b/nboulang/X_Net_v3/xnet/alldataset/large_scale/'
    # largedatanewroot_ct = saveroot_large_ct+'data_ct'
    # largelabelnewroot_ct = saveroot_large_ct+'label_ct'
    # saveroot_large_mr = '/auto/home/users/n/b/nboulang/X_Net_v3/xnet/alldataset/large_scale/'
    # largedatanewroot_mr = saveroot_large_mr+'data_mr'
    # largelabelnewroot_mr = saveroot_large_mr+'label_mr'

    all_imgname_ct = os.listdir(imgroot_ct)
    all_imgname_mr = os.listdir(imgroot_mr)

    for i in range(len(all_imgname_ct)):
        imgname_ct = all_imgname_ct[i]
        imgname_mr = all_imgname_mr[i]
        
        print('imgname_ct is ',imgname_ct)
        print('imgname_mr is ',imgname_mr)

        labelname_ct = imgname_ct.replace('ta', 'ta_seg')
        labelname_mr = imgname_mr.replace('ta', 'ta_seg')
        imgpath_ct = os.path.join(imgroot_ct, imgname_ct)
        imgpath_mr = os.path.join(imgroot_mr, imgname_mr)
        labelpath_ct = os.path.join(labelroot_ct, labelname_ct)
        labelpath_mr = os.path.join(labelroot_mr, labelname_mr)
        # imgnewpath_small_ct = os.path.join(smalldatanewroot_ct, imgname_ct)
        # imgnewpath_small_mr = os.path.join(smalldatanewroot_mr, imgname_mr)
        # labelnewpath_small_ct = os.path.join(smalllabelnewroot_ct, labelname_ct)
        # labelnewpath_small_mr = os.path.join(smalllabelnewroot_mr, labelname_mr)

        imgnewpath_middle_ct = os.path.join(middledatanewroot_ct, imgname_ct)
        imgnewpath_middle_mr = os.path.join(middledatanewroot_mr, imgname_mr)
        labelnewpath_middle_ct = os.path.join(middlelabelnewroot_ct, labelname_ct)
        labelnewpath_middle_mr = os.path.join(middlelabelnewroot_mr, labelname_mr)

        # imgnewpath_large_ct = os.path.join(largedatanewroot_ct, imgname_ct)
        # imgnewpath_large_mr = os.path.join(largedatanewroot_mr, imgname_mr)
        # labelnewpath_large_ct = os.path.join(largelabelnewroot_ct, labelname_ct)
        # labelnewpath_large_mr = os.path.join(largelabelnewroot_mr, labelname_mr)

        img_ct, pixelspace_ct = load_origin_nifty_volume_as_array(imgpath_ct)
        img_mr, pixelspace_mr = load_origin_nifty_volume_as_array(imgpath_mr)
        label_ct, _ = load_origin_nifty_volume_as_array(labelpath_ct)
        label_mr, _ = load_origin_nifty_volume_as_array(labelpath_mr)

        # cropdata_small_ct, crop_label_small_ct, cropdata_small_mr, crop_label_small_mr = preprocessing(img_ct, label_ct, img_mr, label_mr, pixelspace_mr, pixelspace_ct, cropmode = 'small')
        # save_array_as_nifty_volume(crop_label_small_ct, labelnewpath_small_ct)
        # save_array_as_nifty_volume(crop_label_small_mr, labelnewpath_small_mr)
        # save_array_as_nifty_volume(cropdata_small_ct,imgnewpath_small_ct)
        # save_array_as_nifty_volume(cropdata_small_mr,imgnewpath_small_mr)

        cropdata_middle_ct, crop_label_middle_ct, cropdata_middle_mr, crop_label_middle_mr = preprocessing(img_ct, label_ct, img_mr, label_mr, pixelspace_mr, pixelspace_ct, cropmode = 'middle')  # TODO
        save_array_as_nifty_volume(crop_label_middle_ct, labelnewpath_middle_ct)
        save_array_as_nifty_volume(crop_label_middle_mr, labelnewpath_middle_mr)
        save_array_as_nifty_volume(cropdata_middle_ct,imgnewpath_middle_ct)
        save_array_as_nifty_volume(cropdata_middle_mr,imgnewpath_middle_mr)

        # cropdata_large_ct, crop_label_large_ct, cropdata_large_mr, crop_label_large_mr = preprocessing(img_ct, label_ct, img_mr, label_mr, pixelspace_mr, pixelspace_ct, cropmode = 'large')  # TODO
        # save_array_as_nifty_volume(crop_label_large_ct, labelnewpath_large_ct)
        # save_array_as_nifty_volume(crop_label_large_mr, labelnewpath_large_mr)
        # save_array_as_nifty_volume(cropdata_large_ct,imgnewpath_large_mr)
        # save_array_as_nifty_volume(cropdata_large_ct,imgnewpath_large_mr)