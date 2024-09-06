import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched

from llama import LLaMA_adapter

def train_one_epoch(model: LLaMA_adapter,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    # model.module.set_default_trainability()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter
    segment_size = args.segment_size

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    # for data_iter_step, (examples, labels, example_mask, imgs) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    for data_iter_step, (examples, labels, prompt_mask) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        # segment
        examples_segs = torch.tensor_split(examples, list(range(segment_size, examples.shape[1], segment_size)), dim=1)
        labels_segs = torch.tensor_split(labels, list(range(segment_size, labels.shape[1], segment_size)), dim=1)
        memory, norm_term = {'long':{}, 'query':{}}, {}
        start_pos = 0
        c_loss_value = 0
        for i in range(len(examples_segs)):
            with torch.cuda.amp.autocast():
                c_loss, memory, norm_term = model(examples_segs[i], labels_segs[i], start_pos, prompt_mask, memory, norm_term)
            start_pos += examples_segs[i].shape[1]

            c_loss_value_seg = c_loss.item()
            if not math.isfinite(c_loss_value_seg):
                print("Loss is {}, stopping training".format(c_loss_value_seg))
                sys.exit(1)
            # c_loss /= (accum_iter * len(examples_segs))
            c_loss /= accum_iter 
            loss_scaler(c_loss, optimizer, parameters=model.parameters(),
                        update_grad=((data_iter_step + 1) % accum_iter == 0) and (i == len(examples_segs)-1)) # TODO: check 
            c_loss_value += c_loss_value_seg
        
        c_loss_value /= len(examples_segs)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(closs=c_loss_value)
        # metric_logger.update(mloss=m_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        c_loss_value_reduce = misc.all_reduce_mean(c_loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('c_train_loss', c_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
