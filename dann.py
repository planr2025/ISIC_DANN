"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import sys
import random
import tqdm
import time
import warnings
import argparse
import shutil
import os.path as osp
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils
from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.alignment.dann import DomainAdversarialLoss, ImageClassifier
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
from tllib.utils.analysis import collect_feature, tsne, a_distance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    train_transform = utils.get_train_transform(args.train_resizing, scale=args.scale, ratio=args.ratio,
                                                random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)
    # print("train_transform: ", train_transform)
    # print("val_transform: ", val_transform)

    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source, args.target, train_transform, val_transform)
        
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    
    if hasattr(backbone, 'fc') and isinstance(backbone.fc, nn.Linear):
        out_features = backbone.fc.in_features
    elif hasattr(backbone, 'inplanes'):
        out_features = backbone.inplanes  # This works for ResNet
    # else:
    #     dummy_input = torch.randn(1, 3, 224, 224).to(device)
    #     backbone.eval()
    #     with torch.no_grad():
    #         out_features = backbone(dummy_input).shape[1]  # Extract dynamically
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    if hasattr(backbone, 'fc'):
        backbone.fc = nn.Identity()  # Remove the final classification layer
    print(f"Extracted Feature Dim: {out_features}") 
    pool_layer =  nn.AdaptiveAvgPool2d(output_size=(1, 1))
    if args.arch == 'resnet50':
        in_features = 2048  # ResNet50 outputs 2048-dimensional features
    else:
        in_features = 512  # Default case

    classifier = ImageClassifier(
    backbone, num_classes, bottleneck_dim=512,  # Set to 512
    pool_layer=pool_layer,
    head=nn.Sequential(
        nn.Linear(in_features, 512),  # Ensure input is 512
        nn.ReLU(),
        nn.Linear(512, num_classes)
    )
    ).to(device)

    domain_discri = DomainDiscriminator(in_feature=in_features, hidden_size=1024).to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters() + domain_discri.get_parameters(),
                    args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    initial_lr = args.lr  # Ensure this is set correctly
    optimizer.param_groups[0]['lr'] = initial_lr  # Manually set the initial learning rate

    lr_scheduler = LambdaLR(optimizer, lambda x: 1 if x == 0 else (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # define loss function
    domain_adv = DomainAdversarialLoss(domain_discri).to(device)

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        # feature_extractor = nn.Sequential(classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
        feature_extractor = nn.Sequential(
            classifier.backbone,
            classifier.pool_layer,
            nn.Flatten(),  # Converts (B, C, 1, 1) -> (B, C)
            classifier.bottleneck
        ).to(device)

        source_feature = collect_feature(train_source_loader, feature_extractor, device)
        target_feature = collect_feature(train_target_loader, feature_extractor, device)
        print("source_feature.shape: ", source_feature.shape)
        fc = nn.Sequential(
            nn.Linear(args.bottleneck_dim, args.bottleneck_dim // 2),  # Reduce dimension
            nn.ReLU(),  # Activation
            nn.Linear(args.bottleneck_dim // 2, num_classes)  # Final classification layer
        ).to(device)

        # Predict class labels using the classifier's final layer
        # source_preds = classifier.head(source_feature.to(device)).argmax(dim=1).cpu().numpy()
        # target_preds = classifier.head(target_feature.to(device)).argmax(dim=1).cpu().numpy()

        source_preds = fc(source_feature.to(device)).argmax(dim=1).cpu().numpy()
        target_preds = fc(target_feature.to(device)).argmax(dim=1).cpu().numpy()
      
        source_feature = F.normalize(source_feature, p=2, dim=1)
        target_feature = F.normalize(target_feature, p=2, dim=1)

  
        # plot t-SNE
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, f'{args.source[0]}_vs_{args.target[0]}_TSNE.pdf')
        tsne.visualize(
            source_feature, 
            target_feature, 
            source_labels=source_preds, 
            target_labels=target_preds, 
            filename=tSNE_filename, 
            source_label=args.source[0], 
            target_label=args.target[0]
        )
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        acc1, precision, recall, f1 = utils.validate(test_loader, classifier, args, device)
        
        print(f"Accuracy: {acc1:.4f}")
        # print(f"Precision: {precision:.4f}")
        # print(f"Recall: {recall:.4f}")
        # print(f"F1 Score: {f1:.4f}")

        return


    # start training
    best_acc1 = 0.
    for epoch in range(args.epochs):
        print("lr:", lr_scheduler.get_last_lr()[0])
        # train for one epoch
        train(train_source_iter, train_target_iter, classifier, domain_adv, optimizer,
              lr_scheduler, epoch, args)
        # lr_scheduler.step()
        # evaluate on validation set
        acc1, precision, recall, f1  = utils.validate(val_loader, classifier, args, device)

        print("acc1 = {:3.1f}".format(acc1))
        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1, precision, recall, f1  = utils.validate(test_loader, classifier, args, device)
    print("test_acc1 = {:3.1f}".format(acc1))

    logger.close()
import tqdm
import torch
import time
import torch.nn.functional as F

# def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
#           model: ImageClassifier, domain_adv: DomainAdversarialLoss, optimizer: SGD,
#           lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):

#     # Switch to train mode
#     model.train()
#     domain_adv.train()

#     end = time.time()

#     # Initialize tqdm progress bar (total = iterations per epoch / 100)
#     num_updates = args.iters_per_epoch // 100  # Update every 100 iterations
#     pbar = tqdm.tqdm(total=num_updates, desc=f"Epoch [{epoch}]", 
#                      leave=True, dynamic_ncols=True, position=0, ascii=True)

#     for i in range(args.iters_per_epoch):
#         data_s = next(train_source_iter)  # Get full tuple
#         data_t = next(train_target_iter)

#         # Unpack only the first 2 elements safely
#         x_s, labels_s = data_s[:2]  
#         x_t = data_t[0]   
#         x_s = x_s.to(device)
#         x_t = x_t.to(device)
#         labels_s = labels_s.to(device)

#         # Compute output
#         x = torch.cat((x_s, x_t), dim=0)
#         y, f = model(x)

#         if y.shape[0] >= 2:  # Ensure we have at least 2 samples
#             y_s, y_t = y.chunk(2, dim=0)
#         else:
#             y_s, y_t = y, None  # Avoid chunking if batch is too small

#         f_s, f_t = f.chunk(2, dim=0)

#         cls_loss = F.cross_entropy(y_s, labels_s)
#         transfer_loss = domain_adv(f_s, f_t)
#         domain_acc = domain_adv.domain_discriminator_accuracy
#         loss = cls_loss + transfer_loss * args.trade_off

#         cls_acc = accuracy(y_s, labels_s)[0]

#         # Compute gradient and do SGD step
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         # lr_scheduler.step()

#         # ✅ Update tqdm progress bar **only every 100 iterations**
#         if (i + 1) % 100 == 0:
#             pbar.set_postfix(Loss=f"{loss.item():.4f}", ClsAcc=f"{cls_acc:.2f}%", DomainAcc=f"{domain_acc:.2f}%")
#             pbar.update(1)  # Increase the progress bar by 1 step

         

#     pbar.close()  # Close the progress bar at the end of the epoch

def train(train_source_iter, train_target_iter, model, domain_adv, optimizer, lr_scheduler, epoch, args):
    model.train()
    domain_adv.train()

    # Initialize progress bar with more aggressive update settings
    pbar = tqdm.tqdm(
        total=args.iters_per_epoch,
        desc=f"Epoch [{epoch}]",
        leave=True,
        dynamic_ncols=True,
        position=0,
        ascii=True,
        mininterval=0.1,
        miniters=1,  # Force more frequent updates
        file=sys.stdout,  # Explicitly set output stream
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    )

    for i in range(args.iters_per_epoch):
        # Force synchronization if using GPU
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        data_s = next(train_source_iter)
        data_t = next(train_target_iter)

        x_s, labels_s = data_s[:2]
        x_t = data_t[0]
        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)

        # Compute output
        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x)

        if y.shape[0] >= 2:  # Ensure we have at least 2 samples
            y_s, y_t = y.chunk(2, dim=0)
        else:
            y_s, y_t = y, None  # Avoid chunking if batch is too small

        f_s, f_t = f.chunk(2, dim=0)

        # Use tqdm.write for any debug output
        if (i + 1) % 100 == 0:
            tqdm.tqdm.write(f"Source data: {x_s.shape} Labels: {labels_s}")
            tqdm.tqdm.write(f"Target data: {x_t.shape}")
        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = domain_adv(f_s, f_t)
        domain_acc = domain_adv.domain_discriminator_accuracy
        loss = cls_loss + transfer_loss * args.trade_off

        cls_acc = accuracy(y_s, labels_s)[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # Update progress bar
        # Update progress bar every 100 iterations
        if (i + 1) % 100 == 0 or (i + 1) == args.iters_per_epoch:
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'ClsAcc': f'{cls_acc:.2f}%',
                'DomainAcc': f'{domain_acc:.2f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
            pbar.update(100)  # Update by 100 steps
            
            # Debug info (won't interfere with progress bar)
            tqdm.write(f"\nStep {i+1}/{args.iters_per_epoch}:")
            tqdm.write(f"• Loss: {loss.item():.4f} (Cls: {cls_loss.item():.4f}, DA: {transfer_loss.item():.4f})")
            tqdm.write(f"• Batch: {x_s.shape[0]} src, {x_t.shape[0]} tgt samples")
            tqdm.write(f"• Mem: {torch.cuda.memory_allocated()/1024**2:.1f}MB used\n")

    pbar.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DANN for Unsupervised Domain Adaptation')
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
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=512, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
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
    parser.add_argument("--log", type=str, default='dann',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)