import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch.utils.data import DataLoader
from net.HRMamba import HRMamba as VSSM
import argparse
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lrs
from data import get_training_set, get_eval_set
from utils.my_utils import *
import random
import torch.nn.functional as F
from pytorch_msssim import ssim
from datetime import datetime
import logging
from tqdm import tqdm
import numpy as np
from thop import profile

try:
    from net.losses import CharbonnierLoss
except ImportError:
    CharbonnierLoss = torch.nn.L1Loss

parser = argparse.ArgumentParser(description='PyTorch HRMamba training script')
parser.add_argument('--batchSize', type=int, default=1)
parser.add_argument('--nEpochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=4)
parser.add_argument('--decay', type=int, default=200)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--data_train', default='./datasets/LSUI_UIEB_train/input')
parser.add_argument('--label_train', default='./datasets/LSUI_UIEB_train/GT')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--data_test', default='./datasets/testDataset/Test_L400/input')
parser.add_argument('--label_test', default='./datasets/testDataset/Test_L400/gt')
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--save_folder', default='./weights/LSUI_UIEB/hrmamba/')
parser.add_argument('--pretrained', default='')
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--eval_start_epoch', type=int, default=1)
parser.add_argument('--checkpoint_start_epoch', type=int, default=1)
parser.add_argument('--checkpoint_interval', type=int, default=1)
parser.add_argument('--save_mid_model', type=bool, default=False)
parser.add_argument('--mid_model_interval', type=int, default=10)
opt = parser.parse_args()

if not os.path.exists(opt.save_folder):
    os.makedirs(opt.save_folder)

log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

def setup_logger():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(log_dir, f'trainlog_{timestamp}_uieb_r90_hrmamba.txt')
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

logger = setup_logger()

def seed_torch(seed=opt.seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_torch()
cudnn.benchmark = True

model = VSSM().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-4, betas=(0.9, 0.999))
milestones = [i for i in range(1, opt.nEpochs+1) if i % opt.decay == 0]
scheduler = lrs.MultiStepLR(optimizer, milestones, opt.gamma)
criterion = CharbonnierLoss().cuda()

def eval():
    model.eval()
    test_set = get_eval_set(opt.data_test, opt.label_test)
    test_loader = DataLoader(test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    PSNR, SSIM = [], []
    with torch.no_grad():
        for batch in test_loader:
            input, label, _ = batch
            input = input.cuda()
            label = label.cuda()
            out = model(input)
            psnr_val = 10 * torch.log10(1 / F.mse_loss(out, label)).item()
            ssim_val = ssim(out, label, data_range=1, win_size=7, win_sigma=1.5, size_average=False).item()
            PSNR.append(psnr_val)
            SSIM.append(ssim_val)
    torch.cuda.empty_cache()
    return np.mean(PSNR), np.mean(SSIM)

def save_best_model(psnr, epoch):
    path = os.path.join(opt.save_folder, 'best_model.pth')
    torch.save(model.state_dict(), path)
    logger.info(f"New best model saved (PSNR={psnr:.4f}, epoch={epoch}) -> {path}")

def save_mid_model(epoch):
    path = os.path.join(opt.save_folder, f'model_epoch{epoch}.pth')
    torch.save(model.state_dict(), path)
    logger.debug(f"Intermediate model saved: {path}")

def save_full_checkpoint(epoch, optimizer, scheduler, best_psnr, filename='checkpoint_last.pth.tar'):
    path = os.path.join(opt.save_folder, filename)
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_psnr': best_psnr,
    }, path)
    logger.info("Full checkpoint saved")

def load_checkpoint_if_exists():
    path = os.path.join(opt.save_folder, 'checkpoint_last.pth.tar')
    if opt.resume and os.path.isfile(path):
        logger.info(f"Loading checkpoint '{path}'")
        chk = torch.load(path)
        start_epoch = chk['epoch'] + 1
        model.load_state_dict(chk['state_dict'])
        optimizer.load_state_dict(chk['optimizer'])
        scheduler.load_state_dict(chk['scheduler'])
        best_psnr = chk.get('best_psnr', -1.0)
        logger.info(f"Resuming from epoch {start_epoch}, best_psnr = {best_psnr:.4f}")
        return start_epoch, best_psnr
    else:
        logger.info("No checkpoint found, starting from scratch.")
        return 1, -1.0

def train():
    logger.info(f"Model save directory: {opt.save_folder}")
    start_epoch, best_psnr = load_checkpoint_if_exists()
    if opt.pretrained and not os.path.exists(os.path.join(opt.save_folder, 'checkpoint_last.pth.tar')):
        if os.path.isfile(opt.pretrained):
            logger.info(f"Loading pretrained weights from {opt.pretrained}")
            model.load_state_dict(torch.load(opt.pretrained), strict=False)

    for epoch in range(start_epoch, opt.nEpochs + 1):
        model.train()
        epoch_loss = 0
        pbar = tqdm(total=len(training_data_loader), desc=f"Epoch {epoch}/{opt.nEpochs}", ncols=100, leave=False)
        for batch in training_data_loader:
            input, label = batch[0].cuda(), batch[1].cuda()
            optimizer.zero_grad()
            out = model(input)
            loss = criterion(label, out)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({'Loss': loss.item()})
            pbar.update(1)
        pbar.close()
        scheduler.step()
        avg_loss = epoch_loss / len(training_data_loader)

        if epoch >= opt.checkpoint_start_epoch and (epoch - opt.checkpoint_start_epoch) % opt.checkpoint_interval == 0:
            save_full_checkpoint(epoch, optimizer, scheduler, best_psnr)

        if opt.save_mid_model and epoch % opt.mid_model_interval == 0:
            save_mid_model(epoch)

        if epoch >= opt.eval_start_epoch:
            psnr_avg, ssim_avg = eval()
            log_line = f'Epoch {epoch}: PSNR={psnr_avg:.4f}, SSIM={ssim_avg:.4f}, loss={avg_loss:.6f}'
            if psnr_avg > best_psnr:
                best_psnr = psnr_avg
                save_best_model(psnr_avg, epoch)
                log_line += ' [Best]'
            logger.info(log_line)
        else:
            logger.info(f'Epoch {epoch}: loss={avg_loss:.6f}')

        if epoch == opt.nEpochs:
            final_path = os.path.join(opt.save_folder, 'final_model.pth')
            torch.save(model.state_dict(), final_path)
            logger.info(f"Final model saved to {final_path}")

if __name__ == '__main__':
    if opt.gpu_mode and not torch.cuda.is_available():
        raise Exception("No GPU found")
    print('===> Model save directory:', opt.save_folder)
    print('===> Log directory:', os.path.abspath(log_dir))
    print('===> Loading datasets')
    train_set = get_training_set(opt.data_train, opt.label_train, opt.patch_size, opt.data_augmentation)
    training_data_loader = DataLoader(train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    print('===> Building model')
    tmp_model = VSSM().cuda()
    dummy = torch.randn(1, 3, 256, 256).cuda()
    flops, params = profile(tmp_model, inputs=(dummy,), verbose=False)
    print(f'Model FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M')
    del tmp_model, dummy
    torch.cuda.empty_cache()
    train()