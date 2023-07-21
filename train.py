
import argparse
import time
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt

from src_files.data_loading.data_loader import create_data_loaders
from src_files.helper_functions.distributed import print_at_master, to_ddp, reduce_tensor, num_distrib, setup_distrib
from src_files.helper_functions.general_helper_functions import accuracy, AverageMeter, silence_PIL_warnings
from src_files.models import create_model
from src_files.loss_functions.losses import CrossEntropyLS
from torch.cuda.amp import GradScaler, autocast
from src_files.optimizers.create_optimizer import create_optimizer

parser = argparse.ArgumentParser(description='PyTorch ImageNet21K Single-label Training')
parser.add_argument('--data_path', default = '/content/newtraindata', type=str)
parser.add_argument('--lr', default=3e-4, type=float)
parser.add_argument('--model_name', default='resnet50')
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--image_size', default=224, type=int)
parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--label_smooth", default=0.2, type=float)


def main():
    # arguments
    args = parser.parse_args()

    # EXIF warning silent
    silence_PIL_warnings()

    # setup distributed
    setup_distrib(args)

    # Setup model
    model = create_model(args).cuda()
    model = to_ddp(model, args)

    # create optimizer
    optimizer = create_optimizer(model, args)

    # Data loading
    train_loader, val_loader = create_data_loaders(args)

    # Actuall Training
    history = train(model, train_loader, val_loader, optimizer, args)
    plot_history(history)

def train(model, train_loader, val_loader, optimizer, args):
    history = {'train_loss':[],'val_loss':[],'train_acc1': [], 'train_acc5': [], 'val_acc1': [], 'val_acc5': []}
    # set loss
    loss_fn = CrossEntropyLS(args.label_smooth)
    # set scheduler
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
                                        epochs=args.epochs, pct_start=0.1, cycle_momentum=False, div_factor=20)

    # set scalaer
    scaler = GradScaler()

    # training loop
    for epoch in range(args.epochs):
        if num_distrib() > 1:
            train_loader.sampler.set_epoch(epoch)

        # train epoch
        print_at_master("\nEpoch {}".format(epoch))
        epoch_start_time = time.time()

        train_loss_meter = AverageMeter()
        train_top1 = AverageMeter()
        train_top5 = AverageMeter()

        for i, (input, target) in enumerate(train_loader):
            with autocast():  # mixed precision
                output = model(input)
                loss = loss_fn(output, target)  # note - loss also in fp16
            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss_meter.update(loss.item(), input.size(0))
          
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            if num_distrib() > 1:
                acc1 = reduce_tensor(acc1, num_distrib())
                acc5 = reduce_tensor(acc5, num_distrib())
                torch.cuda.synchronize()
            train_top1.update(acc1.item(), input.size(0))
            train_top5.update(acc5.item(), input.size(0))

        history['train_acc1'].append(train_top1.avg)
        history['train_acc5'].append(train_top5.avg)
        history['train_loss'].append(train_loss_meter.avg)
        print_at_master("Epoch: {}/{}, Training Loss: {:.4f}".format(epoch + 1, args.epochs, train_loss_meter.avg))

        epoch_time = time.time() - epoch_start_time
        print_at_master(
            "\nFinished Epoch, Training Rate: {:.1f} [img/sec]".format(len(train_loader) *
                                                                       args.batch_size / epoch_time * max(num_distrib(),
                                                                                                          1)))

                                                                                                          

        # validation epoch
        validate(args, history, val_loader, model)

    return history

def validate(args, history, val_loader, model):
    loss_fn = CrossEntropyLS(args.label_smooth)
    print_at_master("starting validation")
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    val_loss_meter = AverageMeter()  # Add an AverageMeter to track validation loss
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):

            # mixed precision
            with autocast():
                logits = model(input).float()

            # Calculate loss and record it with the AverageMeter
            loss = loss_fn(logits, target)
            val_loss_meter.update(loss.item(), input.size(0))

            # measure accuracy and record accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            if num_distrib() > 1:
                acc1 = reduce_tensor(acc1, num_distrib())
                acc5 = reduce_tensor(acc5, num_distrib())
                torch.cuda.synchronize()
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

    history['val_acc1'].append(top1.avg)
    history['val_acc5'].append(top5.avg)

    # Save the validation loss for the current epoch
    history['val_loss'].append(val_loss_meter.avg)

    print_at_master("Validation results:")
    print_at_master('Acc_Top1 [%] {:.2f},  Acc_Top5 [%] {:.2f} '.format(top1.avg, top5.avg))
    model.train()


def plot_history(history):
    epochs = range(1, len(history['train_acc1']) + 1)

    plt.figure(figsize=(12, 6))

    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['val_loss'], label='Training Loss')
    plt.plot(epochs, history['train_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot Top-1 accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_acc1'], label='Training Accuracy')
    plt.plot(epochs, history['val_acc1'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Top-1 Accuracy')
    plt.legend()

    # Plot Top-5 accuracy
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['train_acc5'], label='Training Accuracy')
    plt.plot(epochs, history['val_acc5'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Top-5 Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    
    
if __name__ == '__main__':
    main()
