import os
import sys
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from PIL.Image import CUBIC
from time import time

from opts import parse_opts
from model import generate_model
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop, TemporalCenterCrop
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger, get_hms
from train import train_epoch
from validation import val_epoch
import test

if __name__ == '__main__':
    opt = parse_opts()

    torch.manual_seed(opt.manual_seed)

    model, parameters = generate_model(opt)
    #print(model)
    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()

    if opt.bayesian:
        from models.BayesianLayers.BBBlayers import GaussianVariationalInference
        criterion = GaussianVariationalInference(criterion)

    #if opt.no_mean_norm and not opt.std_norm:
    if True:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    if not opt.no_train:
        assert opt.train_crop in ['random', 'corner', 'center', 'rescale']
        assert opt.train_temporal_crop in ['center', 'random']
        if opt.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(
                opt.scales, opt.sample_size, crop_positions=['c'])
        elif opt.train_crop == 'rescale':
            #crop_method = transforms.Resize((opt.sample_size, opt.sample_size), interpolation=CUBIC)
            crop_method = Scale(opt.sample_size, interpolation=CUBIC)
        spatial_transform = Compose([
            crop_method,
            RandomHorizontalFlip(),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = TemporalCenterCrop(opt.sample_duration)
        if opt.train_temporal_crop == 'random':
            temporal_transform = TemporalRandomCrop(opt.sample_duration)
        target_transform = ClassLabel()
        training_data = get_training_set(opt, spatial_transform,
                                         temporal_transform, target_transform)
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)
        train_logger = Logger(
            opt.result_path_logs+'_train.log',
            ['epoch', 'loss', 'acc', 'lr'], bool(opt.resume_path))
        train_batch_logger = Logger(
            opt.result_path_logs+'_train_batch.log',
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'], bool(opt.resume_path))

        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening

        if opt.optimizer == 'sgd':
            optimizer = optim.SGD(
                parameters,
                lr=opt.learning_rate,
                momentum=opt.momentum,
                dampening=dampening,
                weight_decay=opt.weight_decay,
                nesterov=opt.nesterov)
        elif opt.optimizer == 'adam' or opt.optimizer == 'amsgrad':
            optimizer = optim.Adam(
                parameters,
                lr=opt.learning_rate,
                weight_decay=opt.weight_decay,
                amsgrad=(opt.optimizer=='amsgrad'))
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=opt.lr_patience)
        del training_data, target_transform, temporal_transform, spatial_transform, parameters, crop_method
    if not opt.no_val:
        if opt.train_crop == 'rescale':
            spatial_transform = Compose([
                #transforms.Resize((opt.sample_size, opt.sample_size), interpolation=CUBIC),
                Scale(opt.sample_size, interpolation=CUBIC),
                ToTensor(opt.norm_value), norm_method
                ])
        else:
            spatial_transform = Compose([
                Scale(opt.sample_size),
                CenterCrop(opt.sample_size),
                ToTensor(opt.norm_value), norm_method
            ])
        assert opt.val_temporal_crop in ['loop', 'random', 'center']
        temporal_transform = LoopPadding(opt.sample_duration)
        if opt.val_temporal_crop == 'center':
            temporal_transform = TemporalCenterCrop(opt.sample_duration)
        elif opt.val_temporal_crop == 'random':
            temporal_transform = TemporalRandomCrop(opt.sample_duration)
        target_transform = ClassLabel()
        validation_data = get_validation_set(
            opt, spatial_transform, temporal_transform, target_transform)
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        val_logger = Logger(
            opt.result_path_logs+'_val.log', ['epoch', 'loss', 'acc', 'acc_mean', 'acc_vote'], bool(opt.resume_path))
        uncertainty_logger = Logger(
            opt.result_path_logs+'_uncertainty.log', ['epoch',
            'epistemic', 'aleatoric', 'random_param_mean', 'random_param_log_alpha',
            'total_param_mean', 'total_param_log_alpha'], bool(opt.resume_path))
        del validation_data, target_transform, temporal_transform, spatial_transform

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt.learning_rate
        del checkpoint

    start_time = time()
    print('run')
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            train_epoch(i, train_loader, model, criterion, optimizer, opt,
                        train_logger, train_batch_logger)
        if not opt.no_val:
            validation_loss = val_epoch(i, val_loader, model, criterion, opt,
                                        val_logger, uncertainty_logger)

        if not opt.no_train and not opt.no_val:
            scheduler.step(validation_loss)

        elapsed_time = time() - start_time
        print('| Elapsed time : %d:%02d:%02d' %(get_hms(elapsed_time)))

    if opt.test:
        spatial_transform = Compose([
            Scale(int(opt.sample_size / opt.scale_in_test)),
            CornerCrop(opt.sample_size, opt.crop_position_in_test),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = LoopPadding(opt.sample_duration)
        target_transform = VideoID()

        test_data = get_test_set(opt, spatial_transform, temporal_transform,
                                 target_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        test.test(test_loader, model, opt, test_data.class_names)
