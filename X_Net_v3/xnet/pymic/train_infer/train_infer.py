# -*- coding: utf-8 -*-
from __future__ import print_function, division


import os
import sys
import time
import scipy
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import sys
print(sys.path)


# Files imports
sys.path.append('/auto/home/users/n/b/nboulang/X_Net_v3/xnet/')
sys.path.insert(0, "/auto/home/users/n/b/nboulang/X_Net_v3/xnet/")

from datetime import datetime
from tensorboardX import SummaryWriter
from pymic.io.image_read_write import save_nd_array_as_image
from pymic.io.nifty_dataset import NiftyDataset
from pymic.io.transform3d import get_transform
from pymic.train_infer.net_factory import get_network
from pymic.train_infer.infer_func import volume_infer, volume_infer_by_patch
from pymic.train_infer.loss import *
from pymic.train_infer.get_optimizer import get_optimiser
from pymic.util.image_process import convert_label
from pymic.util.parse_config import parse_config
import matplotlib.pyplot as plt
from PIL import Image


class TrainInferAgent():
    def __init__(self, config, stage = 'train'):
        self.config = config
        self.stage  = stage
        assert(stage in ['train', 'inference', 'test'])

    def __create_dataset(self):
        root_dir  = self.config['dataset']['root_dir']
        train_csv = self.config['dataset'].get('train_csv', None)
        valid_csv = self.config['dataset'].get('valid_csv', None)
        test_csv  = self.config['dataset'].get('test_csv', None)
        modal_num = self.config['dataset']['modal_num']
        if(self.stage == 'train'):
            transform_names = self.config['dataset']['train_transform']
            validtransform_names = self.config['dataset']['valid_transform']
            self.validtransform_list = [get_transform(name, self.config['dataset']) \
            for name in validtransform_names if name != 'RegionSwop']
        else:
            transform_names = self.config['dataset']['test_transform']
        self.transform_list = [get_transform(name, self.config['dataset']) \
            for name in transform_names if name != 'RegionSwop']
        if('RegionSwop' in transform_names):
            self.region_swop = get_transform('RegionSwop', self.config['dataset']) 
        else:
            self.region_swop = None
        if(self.stage == 'train'):
            train_dataset = NiftyDataset(root_dir=root_dir,
                                csv_file  = train_csv,
                                modal_num = modal_num,
                                with_label= True,
                                transform = transforms.Compose(self.transform_list))
            valid_dataset = NiftyDataset(root_dir=root_dir,
                                csv_file  = valid_csv,
                                modal_num = modal_num,
                                with_label= True,
                                transform = transforms.Compose(self.validtransform_list))
            batch_size = self.config['training']['batch_size']
            
            self.train_loader = torch.utils.data.DataLoader(train_dataset,  # TODO
                batch_size = batch_size, shuffle=True, num_workers=batch_size * 2)
            self.valid_loader = torch.utils.data.DataLoader(valid_dataset, 
                batch_size = 1, shuffle=False, num_workers=batch_size * 2)
        else:
            test_dataset = NiftyDataset(root_dir=root_dir,
                                csv_file  = test_csv,
                                modal_num = modal_num,
                                with_label= False,
                                transform = transforms.Compose(self.transform_list))
            batch_size = 1
            self.test_loder = torch.utils.data.DataLoader(test_dataset, 
                batch_size=batch_size, shuffle=False, num_workers=batch_size)

    def __create_network(self):
        self.net = get_network(self.config['network'])
        # self.net = nn.DataParallel(self.net, device_ids=[0,1])
        self.net.double()

    def __create_optimizer(self):
        self.optimizer = get_optimiser(self.config['training']['optimizer'],
                self.net.parameters(), 
                self.config['training'])
        last_iter = -1
        if(self.checkpoint is not None):
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            last_iter = self.checkpoint['iteration'] - 1
        self.schedule = optim.lr_scheduler.MultiStepLR(self.optimizer,
                self.config['training']['lr_milestones'],
                self.config['training']['lr_gamma'],
                last_epoch = last_iter)

    def __train(self):
        device = torch.device(self.config['training']['device_name'])
        self.net.to(device)

        summ_writer = SummaryWriter(self.config['training']['summary_dir'])
        chpt_prefx  = self.config['training']['checkpoint_prefix']
        loss_func   = self.config['training']['loss_function']
        iter_start  = self.config['training']['iter_start']
        iter_max    = self.config['training']['iter_max']
        iter_valid  = self.config['training']['iter_valid']
        iter_save   = self.config['training']['iter_save']
        class_num   = self.config['network']['class_num']

        if(iter_start > 0):
            checkpoint_file = "{0:}_{1:}.pt".format(chpt_prefx, iter_start)
            self.checkpoint = torch.load(checkpoint_file)
            assert(self.checkpoint['iteration'] == iter_start)
            self.net.load_state_dict(self.checkpoint['model_state_dict'])
        else:
            self.checkpoint = None
        self.__create_optimizer()

        train_loss      = 0
        train_dice_list_ct = []
        train_dice_list_mr = []
        loss_obj = SegmentationLossCalculator(loss_func)
        trainIter = iter(self.train_loader)
        print("{0:} training start".format(str(datetime.now())[:-7]))
        for it in range(iter_start, iter_max):
            try:
                data = next(trainIter)
            except StopIteration:
                trainIter = iter(self.train_loader)
                data = next(trainIter)
            if(self.region_swop is not None):
                data = self.region_swop(data)
            # get the inputs
            inputs_ct, labels_ct= data['image_ct'].double(), data['label_ct']
            inputs_mr, labels_mr= data['image_mr'].double(), data['label_mr']
        
            # # for debug
            # for i in range(inputs.shape[0]):
            #     image_i = inputs[i][0]
            #     label_i = labels[i][0]
            #     image_name = "temp/image_{0:}_{1:}.nii.gz".format(it, i)
            #     label_name = "temp/label_{0:}_{1:}.nii.gz".format(it, i)
            #     save_nd_array_as_image(image_i, image_name, reference_name = None)
            #     save_nd_array_as_image(label_i, label_name, reference_name = None)
            # continue
            inputs_ct, labels_ct= inputs_ct.to(device), labels_ct.to(device) #move to gpu
            inputs_mr, labels_mr= inputs_mr.to(device), labels_mr.to(device) #move to gpu
           

            # zero the parameter gradients
            self.optimizer.zero_grad()
            self.schedule.step()
                
            # forward + backward + optimize
            outputs_ct,  outputs_mr = self.net(inputs_ct, inputs_mr)
            soft_y_ct  = get_soft_label(labels_ct, class_num)
            soft_y_mr  = get_soft_label(labels_mr, class_num)
            loss_ct    = loss_obj.get_loss(outputs_ct, soft_y_ct)
            loss_mr    = loss_obj.get_loss(outputs_mr, soft_y_mr)
            loss    = loss_ct + loss_mr
            loss.backward()
            self.optimizer.step()

            # get dice evaluation for each class
            outputs_argmax_ct = torch.argmax(outputs_ct, dim = 1, keepdim = True)
            outputs_argmax_mr = torch.argmax(outputs_mr, dim = 1, keepdim = True)
            soft_out_ct  = get_soft_label(outputs_argmax_ct, class_num)
            soft_out_mr  = get_soft_label(outputs_argmax_mr, class_num)
            dice_list_ct = get_classwise_dice(soft_out_ct, soft_y_ct)
            dice_list_mr = get_classwise_dice(soft_out_mr, soft_y_mr)
            train_dice_list_ct.append(dice_list_ct.cpu().numpy())
            train_dice_list_mr.append(dice_list_mr.cpu().numpy())

            # evaluate performance on validation set
            train_loss = train_loss + loss.item()
            if (it % iter_valid == iter_valid - 1):
                train_avg_loss = train_loss / iter_valid
                train_cls_dice_ct = np.asarray(train_dice_list_ct).mean(axis = 0)
                train_cls_dice_mr = np.asarray(train_dice_list_mr).mean(axis = 0)
                train_avg_dice_ct = train_cls_dice_ct.mean()
                train_avg_dice_mr = train_cls_dice_mr.mean()
                train_loss = 0.0
                train_dice_list_ct = []
                train_dice_list_mr = []

                valid_loss = 0.0
                valid_dice_list_ct = []
                valid_dice_list_mr = []
                with torch.no_grad():
                    for data in self.valid_loader:
                        inputs_ct, labels_ct= data['image_ct'].double(), data['label_ct']
                        inputs_mr, labels_mr= data['image_mr'].double(), data['label_mr']
                        inputs_ct, labels_ct= inputs_ct.to(device), labels_ct.to(device)
                        inputs_mr, labels_mr= inputs_mr.to(device), labels_mr.to(device)
                        outputs_ct,  outputs_mr = self.net(inputs_ct, inputs_mr)
                        soft_y_ct  = get_soft_label(labels_ct, class_num)
                        soft_y_mr  = get_soft_label(labels_mr, class_num)
                        loss_ct    = loss_obj.get_loss(outputs_ct, soft_y_ct)
                        loss_mr    = loss_obj.get_loss(outputs_mr, soft_y_mr)
                        loss    = (loss_ct + loss_mr) / 2
                        valid_loss = valid_loss + loss.item()

                        outputs_argmax_ct = torch.argmax(outputs_ct, dim = 1, keepdim = True)
                        outputs_argmax_mr = torch.argmax(outputs_mr, dim = 1, keepdim = True)
                        soft_out_ct  = get_soft_label(outputs_argmax_ct, class_num)
                        soft_out_mr  = get_soft_label(outputs_argmax_mr, class_num)
                        dice_list_ct = get_classwise_dice(soft_out_ct, soft_y_ct)
                        dice_list_mr = get_classwise_dice(soft_out_mr, soft_y_mr)
                        valid_dice_list_ct.append(dice_list_ct.cpu().numpy())
                        valid_dice_list_mr.append(dice_list_mr.cpu().numpy())

                valid_avg_loss = valid_loss / len(self.valid_loader)
                valid_cls_dice_ct = np.asarray(valid_dice_list_ct).mean(axis = 0)
                valid_cls_dice_mr = np.asarray(valid_dice_list_mr).mean(axis = 0)
                valid_avg_dice_ct = valid_cls_dice_ct.mean()
                valid_avg_dice_mr = valid_cls_dice_mr.mean()
                loss_scalers = {'train': train_avg_loss, 'valid': valid_avg_loss}
                summ_writer.add_scalars('loss', loss_scalers, it + 1)
                dice_scalers_ct = {'train_ct': train_avg_dice_ct, 'valid_ct': valid_avg_dice_ct}
                dice_scalers_mr = {'train_mr': train_avg_dice_mr, 'valid_mr': valid_avg_dice_mr}
                summ_writer.add_scalars('class_avg_dice_ct', dice_scalers_ct, it + 1)
                summ_writer.add_scalars('class_avg_dice_mr', dice_scalers_mr, it + 1)
                print('train cls dice CT', train_cls_dice_ct.shape, train_cls_dice_ct)
                print('train cls dice MR', train_cls_dice_mr.shape, train_cls_dice_mr)
                print('valid cls dice CT', valid_cls_dice_ct.shape, valid_cls_dice_ct)
                print('valid cls dice MR', valid_cls_dice_mr.shape, valid_cls_dice_mr)
                for c in range(class_num):
                    dice_scalars_ct = {'train':train_cls_dice_ct[c], 'valid':valid_cls_dice_ct[c]}
                    dice_scalars_mr = {'train':train_cls_dice_mr[c], 'valid':valid_cls_dice_mr[c]}
                    summ_writer.add_scalars('class_{0:}_dice'.format(c), dice_scalars_ct, it + 1)
                    summ_writer.add_scalars('class_{0:}_dice'.format(c), dice_scalars_mr, it + 1)
                
                print("{0:} it {1:}, loss {2:.4f}, {3:.4f}".format(
                    str(datetime.now())[:-7], it + 1, train_avg_loss, valid_avg_loss))
            if (it % iter_save ==  iter_save - 1):
                save_dict = {'iteration': it + 1,
                             'model_state_dict': self.net.state_dict(),
                             'optimizer_state_dict': self.optimizer.state_dict()}
                save_name = "{0:}_{1:}.pt".format(chpt_prefx, it + 1)
                torch.save(save_dict, save_name)    
        summ_writer.close()
    
    def __infer(self):
        device = torch.device(self.config['testing']['device_name'])
        self.net.to(device)
        # laod network parameters and set the network as evaluation mode
        self.checkpoint = torch.load(self.config['testing']['checkpoint_name'])
        self.net.load_state_dict(self.checkpoint['model_state_dict'])
        self.net.train()

        output_dir   = self.config['testing']['output_dir']
        save_probability = self.config['testing']['save_probability']
        label_source = self.config['testing']['label_source']
        label_target = self.config['testing']['label_target']
        class_num    = self.config['network']['class_num']
        mini_batch_size     = self.config['testing']['mini_batch_size']
        mini_patch_inshape  = self.config['testing']['mini_patch_shape']
        mini_patch_outshape = None
        # automatically infer outupt shape
        if(mini_patch_inshape is not None):
            patch_inshape = [1, self.config['dataset']['modal_num']] + mini_patch_inshape
            testx = np.random.random(patch_inshape)
            testx = torch.from_numpy(testx)
            testx = torch.tensor(testx)
            testx = testx.to(device)
            testy = self.net(testx)
            testy = testy.detach().cpu().numpy()
            mini_patch_outshape = testy.shape[2:]
            print('mini patch in shape', mini_patch_inshape)
            print('mini patch out shape', mini_patch_outshape)
        start_time = time.time()
        with torch.no_grad():
            for data in self.test_loder:
                images_ct = data['image_ct'].double()
                images_mr = data['image_mr'].double()
                names  = data['names']
                print(names[0])
                data['predict_ct'], data['predict_mr'] = volume_infer(images_ct, images_mr, self.net, device, class_num, 
                    mini_batch_size, mini_patch_inshape, mini_patch_outshape)
            

                for i in reversed(range(len(self.transform_list))):
                    if (self.transform_list[i].inverse):
                        data = self.transform_list[i].inverse_transform_for_prediction(data) 
                output_ct = np.argmax(data['predict_ct'][0], axis = 0)
                output_mr = np.argmax(data['predict_mr'][0], axis = 0)
                output_ct = np.asarray(output_ct, np.uint8)
                output_mr = np.asarray(output_mr, np.uint8)

                if((label_source is not None) and (label_target is not None)):
                    output_ct = convert_label(output_ct, label_source, label_target)
                    output_mr = convert_label(output_mr, label_source, label_target)
                # save the output and (optionally) probability predictions
                root_dir  = self.config['dataset']['root_dir']
                save_name = names[0].split('/')[-1]
                save_name = "{0:}/{1:}".format(output_dir, save_name)
                save_nd_array_as_image(output_ct, save_name, root_dir + names[0])
                save_nd_array_as_image(output_mr, save_name, root_dir + names[0])

                # CT
                if(save_probability):
                    save_name_split = save_name.split('.')
                    if('.nii.gz' in save_name):
                        save_prefix = '.'.join(save_name_split[:-2])
                        save_format = 'nii.gz'
                    else:
                        save_prefix = '.'.join(save_name_split[:-1])
                        save_format = save_name_split[-1]
                    prob = scipy.special.softmax(data['predict_ct'][0],axis = 0)
                    class_num = prob.shape[0]
                    for c in range(0, class_num):
                        temp_prob = prob[c]
                        prob_save_name = "{0:}_prob_{1:}.{2:}".format(save_prefix, c, save_format)
                        save_nd_array_as_image(temp_prob, prob_save_name, root_dir + '/CT_' + names[0])
                # MR  
                if(save_probability):
                    save_name_split = save_name.split('.')
                    if('.nii.gz' in save_name):
                        save_prefix = '.'.join(save_name_split[:-2])
                        save_format = 'nii.gz'
                    else:
                        save_prefix = '.'.join(save_name_split[:-1])
                        save_format = save_name_split[-1]
                    prob = scipy.special.softmax(data['predict_mr'][0],axis = 0)
                    class_num = prob.shape[0]
                    for c in range(0, class_num):
                        temp_prob = prob[c]
                        prob_save_name = "{0:}_prob_{1:}.{2:}".format(save_prefix, c, save_format)
                        save_nd_array_as_image(temp_prob, prob_save_name, root_dir + '/MR_' + names[0])

        avg_time = (time.time() - start_time) / len(self.test_loder)
        print("average testing time {0:}".format(avg_time))

    def run(self):
        agent.__create_dataset()
        agent.__create_network()
        if(self.stage == 'train'):
            self.__train()
        else:
            self.__infer()

if __name__ == "__main__":
    if(len(sys.argv) < 3):
        print('Number of arguments should be 3. e.g.')
        print(sys.argv)
        print('    python train_infer.py train config.cfg')
        exit()
    stage    = str(sys.argv[1])
    cfg_file = str(sys.argv[2])
    config   = parse_config(cfg_file)
    agent    = TrainInferAgent(config, stage)
    agent.run()

