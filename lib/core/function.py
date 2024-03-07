import time, os, torch
from core.evaluate import accuracy
from utils.utils import save_batch_heatmaps

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict, device, logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (img, coc_label) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        img = img.to(device)
        # compute output
        outputs_coc = model(img)

        coc_label = coc_label.to(device)
        
        loss = criterion(outputs_coc, coc_label)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), img.size(0))

        avg_acc = accuracy(outputs_coc.detach().cpu().numpy(),
                                          coc_label.detach().cpu().numpy())
        acc.update(avg_acc)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=img.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, 
                      acc=acc
                      )
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}_{}.jpg'.format(os.path.join(output_dir, 'train'), epoch, i)
            save_batch_heatmaps(img, outputs_coc, prefix)
def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, device, logger, writer_dict=None,epoch=0):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (img, coc_label) in enumerate(val_loader):
            # compute output
            img = img.to(device)
            coc_label = coc_label.to(device)

            outputs_coc = model(img)
        
            loss = criterion(outputs_coc, coc_label)
            num_images = img.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            avg_acc = accuracy(outputs_coc.detach().cpu().numpy(),
                                            coc_label.detach().cpu().numpy())
            acc.update(avg_acc)
            batch_time.update(time.time() - end)
            end = time.time()
            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                        'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                            i, len(val_loader), batch_time=batch_time,
                            loss=losses
                            , acc=acc)
                prefix = '{}_{}_{}.jpg'.format(os.path.join(output_dir, 'test'), epoch, i)
                save_batch_heatmaps(img, outputs_coc, prefix)

                logger.info(msg)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )

            writer_dict['valid_global_steps'] = global_steps + 1

    return losses.avg, acc.avg

class TVLoss(torch.nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]



