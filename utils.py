"""
@author: Junguang Jiang, Baixu Chen
@contact: JiangJunguang1123@outlook.com, cbx_99_hasta@outlook.com
"""
import sys
import os.path as osp
import time
from PIL import Image
import sklearn

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from timm.data.auto_augment import auto_augment_transform, rand_augment_transform

sys.path.append('../../..')
import tllib.vision.datasets as datasets
import tllib.vision.models as models
from tllib.vision.transforms import ResizeImage
from tllib.utils.metric import accuracy, ConfusionMatrix
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.vision.datasets.imagelist import MultipleDomainsDataset
from sklearn.metrics import precision_score, recall_score, f1_score


def get_model_names():
    return sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    ) + timm.list_models()


def get_model(model_name, pretrain=True):
    if model_name in models.__dict__:
        # load models from tllib.vision.models
        backbone = models.__dict__[model_name](pretrained=pretrain)
    else:
        # load models from pytorch-image-models
        backbone = timm.create_model(model_name, pretrained=pretrain)
        try:
            backbone.out_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0, '')
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
    return backbone


def get_dataset_names():
    return sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    ) + ['Digits']


def get_dataset(dataset_name, root, source, target, train_source_transform, val_transform, train_target_transform=None):
    if train_target_transform is None:
        train_target_transform = train_source_transform
    if dataset_name == "Digits":
        train_source_dataset = datasets.__dict__[source[0]](osp.join(root, source[0]), download=True,
                                                            transform=train_source_transform)
        train_target_dataset = datasets.__dict__[target[0]](osp.join(root, target[0]), download=True,
                                                            transform=train_target_transform)
        val_dataset = test_dataset = datasets.__dict__[target[0]](osp.join(root, target[0]), split='test',
                                                                  download=True, transform=val_transform)
        class_names = datasets.MNIST.get_classes()
        num_classes = len(class_names)
    elif dataset_name in datasets.__dict__:
        # load datasets from tllib.vision.datasets
        dataset = datasets.__dict__[dataset_name]
        # def concat_dataset(root, tasks, **kwargs):
        #     print(f"Tasks: {tasks}")
        #     print(f"Root: {root}")
        #     print(f"Kwargs: {kwargs}")
        #     return MultipleDomainsDataset([dataset(task=task, **kwargs) for task in tasks], tasks)

        def concat_dataset(tasks, start_idx, **kwargs):
            # return ConcatDataset([dataset(task=task, **kwargs) for task in tasks])
            return MultipleDomainsDataset([dataset(task=task, **kwargs) for task in tasks], tasks,
                                          domain_ids=list(range(start_idx, start_idx + len(tasks))))

        train_source_dataset = concat_dataset(root=root, tasks=source, download=True, transform=train_source_transform,
                                              start_idx=0)
        train_target_dataset = concat_dataset(root=root, tasks=target, download=True, transform=train_target_transform,
                                              start_idx=len(source))
        val_dataset = concat_dataset(root=root, tasks=target, download=True, transform=val_transform,
                                     start_idx=len(source))
        if dataset_name == 'DomainNet':
            test_dataset = concat_dataset(root=root, tasks=target, split='test', download=True, transform=val_transform,
                                          start_idx=len(source))
        else:
            test_dataset = val_dataset
        class_names = train_source_dataset.datasets[0].classes
        num_classes = len(class_names)
    else:
        raise NotImplementedError(dataset_name)
    return train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, class_names


from sklearn.metrics import precision_score, recall_score, f1_score

# def validate(val_loader, model, args, device):
#     batch_time = AverageMeter('Time', ':6.3f')
#     losses = AverageMeter('Loss', ':.4e')
#     top1 = AverageMeter('Acc@1', ':6.2f')
#     progress = ProgressMeter(
#         len(val_loader),
#         [batch_time, losses, top1],
#         prefix='Test: ')

#     # Switch to evaluation mode
#     model.eval()
#     if args.per_class_eval:
#         confmat = ConfusionMatrix(len(args.class_names))
#     else:
#         confmat = None

#     # Lists to store true and predicted labels
#     y_true = []
#     y_pred = []

#     with torch.no_grad():
#         end = time.time()
#         for i, data in enumerate(val_loader):
#             images, target = data[:2]
#             images = images.to(device)
#             target = target.to(device)
#             # print(f"Batch {i}: Images Shape - {images.shape}, Target Shape - {target.shape}")  # Debugging
#             # Compute output
#             output = model(images)
#             print(output)
#             if isinstance(output, tuple):  # If model outputs a tuple
#                 logits, _ = output  # Extract logits
#             else:
#                 logits = output
#                 logits = logits.squeeze()

#             # Compute loss
#             loss = F.cross_entropy(logits, target)

#             # Store predictions and ground truth
#             y_true.extend(target.cpu().numpy())  # Convert to NumPy
#             y_pred.extend(logits.argmax(dim=1).cpu().numpy())  # Get predicted class

#             # Measure accuracy and record loss
#             acc1, = accuracy(logits, target, topk=(1,))
#             if confmat:
#                 confmat.update(target, logits.argmax(1))
#             losses.update(loss.item(), images.size(0))
#             top1.update(acc1.item(), images.size(0))

#             # Measure elapsed time
#             batch_time.update(time.time() - end)
#             end = time.time()

#             if i % args.print_freq == 0:
#                 progress.display(i)

#         print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
#         if confmat:
#             print(confmat.format(args.class_names))

#     # Compute precision, recall, and F1-score
#     precision = precision_score(y_true, y_pred, average='macro')
#     recall = recall_score(y_true, y_pred, average='macro')
#     f1 = f1_score(y_true, y_pred, average='macro')

#     print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

#     return top1.avg, precision, recall, f1
from sklearn.metrics import accuracy_score

def validate(val_loader, model, args, device):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(val_loader), [batch_time, losses], prefix='Test:')

    # Switch to evaluation mode
    model.eval()
    if args.per_class_eval:
        confmat = ConfusionMatrix(len(args.class_names))
    else:
        confmat = None

    # Lists to store true and predicted labels
    y_true = []
    y_pred = []

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[:2]
            images = images.to(device)
            target = target.to(device)

            # Compute output
            output = model(images)
            if isinstance(output, tuple):
                logits, _ = output  # Extract logits
            else:
                logits = output
            
            # Ensure logits shape is correct
            logits = logits.view(images.size(0), -1)

            # Compute loss
            loss = F.cross_entropy(logits, target)

            # Store predictions and ground truth
            y_true.extend(target.cpu().numpy())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy())

            # Record loss
            losses.update(loss.item(), images.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # Compute overall accuracy
        acc1 = accuracy_score(y_true, y_pred) * 100  # Convert to percentage
        print(f' * Acc@1 {acc1:.3f}%')

        if confmat:
            print(confmat.format(args.class_names))

    # Compute precision, recall, and F1-score
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    return acc1, precision, recall, f1



def get_train_transform(resizing='default', scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), random_horizontal_flip=True,
                        random_color_jitter=False, resize_size=224, norm_mean=(0.485, 0.456, 0.406),
                        norm_std=(0.229, 0.224, 0.225), auto_augment=None):
    """
    resizing mode:
        - default: resize the image to 256 and take a random resized crop of size 224;
        - cen.crop: resize the image to 256 and take the center crop of size 224;
        - res: resize the image to 224;
    """
    transformed_img_size = 224
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomResizedCrop(224, scale=scale, ratio=ratio)
        ])
    elif resizing == 'cen.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224)
        ])
    elif resizing == 'ran.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomCrop(224)
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
        transformed_img_size = resize_size
    else:
        raise NotImplementedError(resizing)
    transforms = [transform]
    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if auto_augment:
        aa_params = dict(
            translate_const=int(transformed_img_size * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in norm_mean]),
            interpolation=Image.BILINEAR
        )
        if auto_augment.startswith('rand'):
            transforms.append(rand_augment_transform(auto_augment, aa_params))
        else:
            transforms.append(auto_augment_transform(auto_augment, aa_params))
    elif random_color_jitter:
        transforms.append(T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])
    return T.Compose(transforms)


def get_val_transform(resizing='default', resize_size=224,
                      norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
    """
    resizing mode:
        - default: resize the image to 256 and take the center crop of size 224;
        – res.: resize the image to 224
    """
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
    else:
        raise NotImplementedError(resizing)
    return T.Compose([
        transform,
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])


def empirical_risk_minimization(train_source_iter, model, optimizer, lr_scheduler, epoch, args, device):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)[:2]
        x_s = x_s.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y_s, f_s = model(x_s)

        cls_loss = F.cross_entropy(y_s, labels_s)
        loss = cls_loss

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
