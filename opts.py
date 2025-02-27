import argparse
import json
import os

from mean import get_mean, get_std
from constants import *


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--oth', action='store_true')
    parser.add_argument(
        '--dataset',
        default='kinetics',
        type=str,
        help='Used dataset (activitynet | kinetics | ucf101 | hmdb51 | jhmdb | ucfsports)')
    parser.add_argument(
        '--root_path',
        default='/root/data/ActivityNet',
        type=str,
        help='Root directory path of data')
    parser.add_argument(
        '--video_path',
        default='video_kinetics_jpg',
        type=str,
        help='Directory path of Videos')
    parser.add_argument(
        '--annotation_path',
        default='kinetics.json',
        type=str,
        help='Annotation file path')
    parser.add_argument(
        '--result_path',
        default='',
        type=str,
        help='Result directory path')
    parser.add_argument(
        '--n_classes',
        default=400,
        type=int,
        help=
        'Number of classes (activitynet: 200, kinetics: 400, ucf101: 101, hmdb51: 51)'
    )
    parser.add_argument(
        '--n_finetune_classes',
        default=400,
        type=int,
        help=
        'Number of classes for fine-tuning. n_classes is set to the number when pretraining.'
    )
    parser.add_argument(
        '--sample_size',
        default=112,
        type=int,
        help='Height and width of inputs')
    parser.add_argument(
        '--sample_duration',
        default=16,
        type=int,
        help='Temporal duration of inputs')
    parser.add_argument(
        '--initial_scale',
        default=1.0,
        type=float,
        help='Initial scale for multiscale cropping')
    parser.add_argument(
        '--n_scales',
        default=5,
        type=int,
        help='Number of scales for multiscale cropping')
    parser.add_argument(
        '--scale_step',
        default=0.84089641525,
        type=float,
        help='Scale step for multiscale cropping')
    parser.add_argument(
        '--train_crop',
        default='corner',
        type=str,
        help=
        'Spatial cropping method in training. random is uniform. corner is selection from 4 corners and 1 center.  (random | corner | center)'
    )
    parser.add_argument(
        '--train_temporal_crop', '--ttc',
        default='random',
        type=str,
        help='Temporal cropping. random or center of sample.')
    parser.add_argument(
        '--val_temporal_crop', '--vtc',
        default='loop',
        type=str,
        help='Temporal cropping in validation set. Random, center or looppading')
    parser.add_argument(
        '--learning_rate', '--lr',
        default=0.1,
        type=float,
        help=
        'Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument(
        '--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument(
        '--weight_decay', default=1e-3, type=float, help='Weight Decay')
    parser.add_argument(
        '--mean_dataset',
        default='activitynet',
        type=str,
        help=
        'dataset for mean values of mean subtraction (activitynet | kinetics)')
    parser.add_argument(
        '--no_mean_norm',
        action='store_true',
        help='If true, inputs are not normalized by mean.')
    parser.set_defaults(no_mean_norm=False)
    parser.add_argument(
        '--std_norm',
        action='store_true',
        help='If true, inputs are normalized by standard deviation.')
    parser.set_defaults(std_norm=False)
    parser.add_argument(
        '--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument(
        '--optimizer', '--opt',
        default='adam',
        const='adam',
        nargs='?',
        choices=['sgd', 'adam', 'amsgrad'],
        help='( sgd | adam | amsgrad |')
    parser.add_argument(
        '--lr_patience', '--lr_p',
        default=10,
        type=int,
        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.'
    )
    parser.add_argument(
        '--batch_size', '--bs', default=128, type=int, help='Batch Size')
    parser.add_argument(
        '--n_epochs',
        default=200,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--begin_epoch',
        default=1,
        type=int,
        help=
        'Training begins at this epoch. Previous trained model indicated by resume_path is loaded.'
    )
    parser.add_argument('--plot_cm', default=25, type=int, help='Every how many epochs print a confusion matrix')
    parser.add_argument(
        '--n_val_samples',
        default=3,
        type=int,
        help='Number of validation samples for each activity')
    parser.add_argument(
        '--n_tra_samples',
        default=1,
        type=int,
        help='Number of training samples for each activity')
    parser.add_argument(
        '--resume_path',
        default='',
        type=str,
        help='Save data (.pth) of previous training')
    parser.add_argument(
        '--pretrain_path', default='', type=str, help='Pretrained model (.pth)')
    parser.add_argument(
        '--ft_begin_index',
        default=0,
        type=int,
        help='Begin block index of fine-tuning')
    parser.add_argument(
        '--no_train',
        action='store_true',
        help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument(
        '--no_val',
        action='store_true',
        help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)
    parser.add_argument(
        '--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test=False)
    parser.add_argument(
        '--test_subset',
        default='val',
        type=str,
        help='Used subset in test (val | test)')
    parser.add_argument(
        '--scale_in_test',
        default=1.0,
        type=float,
        help='Spatial scale in test')
    parser.add_argument(
        '--crop_position_in_test',
        default='c',
        type=str,
        help='Cropping method (c | tl | tr | bl | br) in test')
    parser.add_argument(
        '--no_softmax_in_test',
        action='store_true',
        help='If true, output for each clip is not normalized using softmax.')
    parser.set_defaults(no_softmax_in_test=False)
    parser.add_argument(
        '--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument(
        '--n_threads',
        default=4,
        type=int,
        help='Number of threads for multi-thread loading')
    parser.add_argument(
        '--checkpoint', '--ckpt',
        default=10,
        type=int,
        help='Trained model is saved at every this epochs.')
    parser.add_argument(
        '--keep_n_checkpoints', '--keep_n_ckpt',
        default=5,
        type=int,
        help='Number of checkpoints to keep.')
    parser.add_argument(
        '--no_hflip',
        action='store_true',
        help='If true holizontal flipping is not performed.')
    parser.set_defaults(no_hflip=False)
    parser.add_argument(
        '--norm_value',
        default=1,
        type=int,
        help=
        'If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument(
        '--model',
        default='resnet',
        type=str,
        help='(resnet | preresnet | wideresnet | resnext | densenet | BBBresnet | ')
    parser.add_argument(
        '--model_depth',
        default=18,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--wide_resnet_k', default=2, type=int, help='Wide resnet k')
    parser.add_argument(
        '--resnext_cardinality',
        default=32,
        type=int,
        help='ResNeXt cardinality')
    parser.add_argument(
        '--manual_seed', default=1, type=int, help='Manually set random seed')
    # Extra options
    parser.add_argument(
        '--split',
        default=1,
        type=int,
        help='Split. Dataset: ( HMDB51 )')
    #parser.add_argument(
    #    '--cross_validation', action='store_true', help='Set true to test using cross validation')
    #parser.add_argument(
    #    '--cross_validation_type',
    #    default='loo',
    #    type=str,
    #    help='Type of Cross Validation: ( loo ')#| 5fold | 10fold )')
    # Bayesian options
    parser.add_argument(
        '--bayesian',
        action='store_true',
        help='Set true if model is bayesian. It is set automatically.')
    parser.add_argument('--kl_calc', action='store_true', help='Calculate KL')
    parser.add_argument('--bias', action='store_true', help='Include bias')
    parser.add_argument('--q_logvar_init', default=-5, type=int, help='q_logvar_init')
    parser.add_argument('--num_samples', default=10, type=int, help='Number of samples')
    parser.add_argument('--beta_type',
        default="Blundell",
        type=str,
        help='Beta type ( Bloendell | Soenderby | Standard | ')
    parser.add_argument('--img_channels', '--imgc', default=3, type=int, choices=[3, 1], help='(3 or 1) -> (rgb or bw)')
    parser.add_argument('--dropout_rate', '--dp', default=0.2, type=float, help='Dropout rate in dropout_resnet model')
    #parser.add_argument('--q_std_init',
    #    #default=5,
    #    type=float,
    #    help='Posterior Standard Deviation initial value => logvar is calculated automatically (log(std**2))')

    args = parser.parse_args()

    # Constants
    args.root_path = ROOT_PATH[args.dataset]
    if(os.path.isdir('/var/scratch/delariva/')):
        args.root_path = '/var/scratch/delariva/%s'%args.root_path

    args.video_path = VIDEO_PATH[args.dataset]
    try: args.annotation_path = ANNOTATION_PATH[args.dataset]%args.split
    except: args.annotation_path = ANNOTATION_PATH[args.dataset]
    args.n_classes = NUM_CLASSES[args.dataset]
    args.bayesian = BAYESIAN[args.model]
    if not args.bayesian: args.num_samples = 1
    args.result_path_logs = args.result_path or create_results_dir_name(args)
    args.result_path = args.result_path_logs + os.sep

    # Build opts
    if args.root_path != '':
        args.video_path = os.path.join(args.root_path, args.video_path)
        args.annotation_path = os.path.join(args.root_path, args.annotation_path)
        args.checkpoints_path = os.path.join(args.root_path, args.result_path)
        if not os.path.isdir(args.result_path):
            os.makedirs(args.result_path)
        if not os.path.isdir(args.checkpoints_path):
            os.makedirs(args.checkpoints_path)
        if args.resume_path:
            args.resume_path = os.path.join(args.checkpoints_path, args.resume_path)
        if args.pretrain_path:
            args.pretrain_path = os.path.join(args.root_path, args.pretrain_path)
    args.scales = [args.initial_scale]
    for i in range(1, args.n_scales):
        args.scales.append(args.scales[-1] * args.scale_step)
    args.arch = '{}-{}'.format(args.model, args.model_depth)
    args.mean = get_mean(args.norm_value, dataset=args.mean_dataset)
    args.std = get_std(args.norm_value)
    args.labels = load_labels(args.annotation_path)
    print(args)
    with open(args.result_path_logs+'_opts.json', 'w') as opt_file:
        json.dump(vars(args), opt_file)

    return args
