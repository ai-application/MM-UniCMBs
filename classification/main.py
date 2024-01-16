import argparse
import datetime
import random

import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import clip
from pathlib import Path
from torch.utils.data import dataset
import torchvision.transforms as transforms
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
from engine import train_one_epoch, evaluate
from samplers import RASampler
import utils
import os

from models import CmbFormer_S, CmbFormer_B, resNet3D_18, resNet3D_50, resNet3D_101, uniformer_xxs, uniformer_xs, UniFormerv2_b16, UniFormerv2_114

from datasets import CMB_Dataset

def get_args_parser():
    parser = argparse.ArgumentParser('CMBs training script', add_help=False)
    parser.add_argument('--batch-size', default=32, type=int)

    parser.add_argument('--epochs', default=300, type=int)

    # Model parameters
    parser.add_argument('--input-size', default=64, type=int, help='images input size')

    # Model names
    parser.add_argument('--model-name', default='CmbFormer_S', type=str, help='Model Name: CmbFormer_S, CmbFormer_B,'
                                                                              'resNet3D_18, resNet3D_50, resNet3D_101, uniformer_xxs,'
                                                                              ' uniformer_xs, UniFormerv2_b16, UniFormerv2_114')
    # Text parameters
    parser.add_argument('--input-text', action="store_true", default=False, help='Texts input')

    # Text parameters
    parser.add_argument('--use-bert', action="store_true", default=False, help='Text BERT embedding or CLIP text embedding')

    # Text parameters
    parser.add_argument('--use-clip-image', action="store_true", default=False,
                        help='CLIP image embedding')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data-path', default='/home/box-train', type=str,
                        help='dataset path')

    parser.add_argument('--inat-category', default='name',
                        choices=[],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='error with multi-node if without this parameter. No effect?')
    return parser

def main(args):
    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    assert args.model_name in ['CmbFormer_S', 'CmbFormer_B', 'resNet3D_18', 'resNet3D_50', 'resNet3D_101', 'uniformer_xxs',
                               'uniformer_xs', 'UniFormerv2_b16', 'UniFormerv2_114']

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train = CMB_Dataset(args, os.path.join(args.data_path, 'train_images'), 'train', 'MINIST', batch_size=args.batch_size)
    dataset_val = CMB_Dataset(args, os.path.join(args.data_path, 'test_images'), 'valid', 'MINIST', batch_size=args.batch_size)

    args.nb_classes = dataset_train.get_class_nums()

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()

        sampler_train = RASampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)

        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=True
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=True
    )

    if args.model_name == 'CmbFormer_S':
        if args.use_bert:
                model = CmbFormer_S(num_classes = args.nb_classes, text_input = args.input_text, text_channel_in = 70,
                                bert_embedding = args.use_bert, clip_image_input = args.use_clip_image)
        else:
            model = CmbFormer_S(num_classes=args.nb_classes, text_input=args.input_text,
                                text_channel_in = 5, bert_embedding = args.use_bert, clip_image_input = args.use_clip_image)

    elif args.model_name == 'CmbFormer_B':
        if args.use_bert:
            model = CmbFormer_B(num_classes=args.nb_classes, text_input=args.input_text, text_channel_in = 70,
                                bert_embedding = args.use_bert, clip_image_input = args.use_clip_image)
        else:
            model = CmbFormer_B(num_classes=args.nb_classes, text_input=args.input_text, text_channel_in = 5,
                                bert_embedding = args.use_bert, clip_image_input = args.use_clip_image)

    elif args.model_name == 'resNet3D_18':
        model = resNet3D_18(num_classes=args.nb_classes)
    elif args.model_name == 'resNet3D_50':
        model = resNet3D_50(num_classes=args.nb_classes)
    elif args.model_name == 'resNet3D_101':
        model = resNet3D_101(num_classes=args.nb_classes)
    elif args.model_name == 'uniformer_xxs':
        model = uniformer_xxs(num_classes=args.nb_classes)
    elif args.model_name == 'uniformer_xs':
        model = uniformer_xs(num_classes=args.nb_classes)
    elif args.model_name == 'UniFormerv2_b16':
        model = UniFormerv2_b16(num_classes=args.nb_classes)
    elif args.model_name == 'UniFormerv2_114':
        model = UniFormerv2_114(num_classes=args.nb_classes)
    else:
        raise ValueError

    print(model)

    model.eval()

    if args.use_bert:
        flops = FlopCountAnalysis(model, (torch.rand(2, 3, 3, args.input_size, args.input_size), torch.rand(2, 70, 768), torch.rand(2, 3, 512)))
    else:
        flops = FlopCountAnalysis(model, (torch.rand(2, 3, 3, args.input_size, args.input_size), torch.rand(2, 5, 512), torch.rand(2, 3, 512)))

    #统计FLOPs，目前先注释掉
    print(flop_count_table(flops))

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    max_accuracy = 0.0
    output_dir = Path(args.output_dir)
    # ipdb.set_trace()
    if args.resume == '':
        tmp = f"{args.output_dir}/checkpoint.pth"
        if os.path.exists(tmp):
            args.resume = tmp
    flag = os.path.exists(args.resume)
    if args.resume and flag:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        if 'model' in checkpoint:
            model_without_ddp.load_state_dict(checkpoint['model'])
        else:
            model_without_ddp.load_state_dict(checkpoint)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch,
            args.clip_grad
        )

        lr_scheduler.step(epoch)

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                        'max_accuracy': max_accuracy,
                    }, checkpoint_path)
            if epoch % 10 == 0:
                utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                        'max_accuracy': max_accuracy,
                    }, f"{args.output_dir}/backup.pth")

        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])

        print('Max accuracy: {:.2f}%'.format(max_accuracy))
        if max_accuracy == test_stats["acc1"]:
            utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'max_accuracy': max_accuracy,
                }, f"{args.output_dir}/best.pth")
        train_f = {'train_{}'.format(k): v for k, v in train_stats.items()}
        test_f = {'test_{}'.format(k): v for k, v in test_stats.items()}

        log_stats = dict({'epoch': epoch,
                          'n_parameters': n_parameters}, **train_f, **test_f)

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CMBs training script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
