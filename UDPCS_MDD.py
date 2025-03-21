import random
import time
import warnings
import argparse
import os.path as osp
import shutil
import copy
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

import utils
from tllib.alignment.mdd import ClassificationMarginDisparityDiscrepancy \
    as MarginDisparityDiscrepancy, ImageClassifier
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
from tllib.utils.analysis import collect_feature, tsne, a_distance
import numpy as np
from tllib.self_training.pseudo_label import ConfidenceBasedSelfTrainingLoss
from tllib.vision.transforms import MultipleApply

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    print("################")
    print("UDPCS")
    print("################")

    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)
    train_source_transform = utils.get_train_transform(args.train_resizing, scale=args.scale, ratio=args.ratio,
                                                       random_horizontal_flip=not args.no_hflip,
                                                       random_color_jitter=False, resize_size=args.resize_size,
                                                       norm_mean=args.norm_mean, norm_std=args.norm_std)
    
    weak_augment = utils.get_train_transform(args.train_resizing, scale=args.scale, ratio=args.ratio,
                                             random_horizontal_flip=not args.no_hflip,
                                             random_color_jitter=False, resize_size=args.resize_size,
                                             norm_mean=args.norm_mean, norm_std=args.norm_std)
    strong_augment = utils.get_train_transform(args.train_resizing, scale=args.scale, ratio=args.ratio,
                                               random_horizontal_flip=not args.no_hflip,
                                               random_color_jitter=False, resize_size=args.resize_size,
                                               norm_mean=args.norm_mean, norm_std=args.norm_std,
                                               auto_augment=args.auto_augment)
    train_target_transform = MultipleApply([weak_augment, strong_augment])

    train_source_dataset, train_target_dataset, val_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source, args.target, train_source_transform, val_transform, train_target_transform=train_target_transform)

    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    target_pll_loader = DataLoader(train_target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                 width=args.bottleneck_dim, pool_layer=pool_layer).to(device)

    optimizer = SGD(classifier.get_parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # Load checkpoint from latest_UDPCS
    checkpoint = torch.load(logger.get_checkpoint_path('latest')[:-4]+'_UDPCS.pth', map_location='cpu')
    classifier.load_state_dict(checkpoint)

    best_acc1 = 0.
    pll = True

    for epoch in range(15, 18):  # Training for 3 more epochs (15 to 17)
        print("PLL: ", pll)
        
        if pll:
            train_pll(target_pll_loader, classifier, optimizer, lr_scheduler, epoch, pll, args)
        
        acc1 = utils.validate(val_loader, classifier, args, device)

        # Save latest checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest')[:-4]+'_UDPCS.pth')
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest')[:-4]+'_UDPCS.pth', logger.get_checkpoint_path('best')[:-4]+'_UDPCS.pth')
        best_acc1 = max(acc1, best_acc1)
        print("Best Accuracy PLL: "+str(best_acc1))

    print("best_acc1 = {:3.1f}".format(best_acc1))

    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')[:-4]+'_UDPCS.pth'))
    acc1 = utils.validate(test_loader, classifier, args, device)
    print("test_acc1_PLL = {:3.1f}".format(acc1))

    logger.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MDD for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--no-hflip', action='store_true', help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+', default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+', default=(0.229, 0.224, 0.225), help='normalization std')
    parser.add_argument('--auto-augment', default='rand-m10-n2-mstd2', type=str,
                        help='AutoAugment policy (default: rand-m10-n2-mstd2)')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=1024, type=int)
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('--margin', type=float, default=4., help="margin gamma")
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.004, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.0002, type=float)
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='confidence threshold')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='mdd',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)
