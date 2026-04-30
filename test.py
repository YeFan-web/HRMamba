from __future__ import print_function
from torch.utils.data import DataLoader

from net.HRMamba import HRMamba as VSSM
from pytorch_msssim import ssim
import numpy
import torch
import time
import os
import argparse
import torch.nn.functional as F
from PIL import Image

# 导入原有的数据集函数（需要有 get_eval_set 和新的 get_infer_set）
from data import get_eval_set, get_infer_set
from utils.my_utils import torch_to_np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def generate_result_paths():
    """生成输出文件夹路径"""
    variable_parts = ["LSUI", "EUVP", "R90x256"]
    constant_part = "_hrmamba"
    return [f"./results/{var}{constant_part}" for var in variable_parts]

parser = argparse.ArgumentParser(description='PyTorch HRMamba Inference & Evaluation')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader')
parser.add_argument('--rgb_range', type=int, default=1)
parser.add_argument('--data_test', type=str, nargs='+', default=[
    './datasets/testDataset/Test_L400/input',
    './datasets/testDataset/Test_E515/Inp',
    './datasets/testDataset/Test_R90/input',
])
parser.add_argument('--label_test', type=str, nargs='+', #default=None,
                    default=[
                        './datasets/testDataset/Test_L400/gt',
                        './datasets/testDataset/Test_E515/GTr',
                        './datasets/testDataset/Test_R90/gt',
                      ],
                    help='Ground truth directories (if omitted, inference only)')
parser.add_argument('--model', default='./pretrained_models/HRMamba.pth',
                    help='Pretrained model path')
parser.add_argument('--output_folder', type=str, nargs='+', default=generate_result_paths())
parser.add_argument('--compute_metrics', default=True,
                    help='Compute PSNR/SSIM (requires --label_test)')

opt = parser.parse_args()

# 保证输出文件夹数量与数据测试集数量匹配
if len(opt.output_folder) != len(opt.data_test):
    if len(opt.output_folder) == 1:
        opt.output_folder = opt.output_folder * len(opt.data_test)
    else:
        raise ValueError("Number of output folders must equal number of test datasets or be 1.")

print('===> Building model')
model = VSSM().cuda()
model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
model.eval()
print('Pre-trained model is loaded.')

def my_save_image(name, img_array, output_folder):
    """保存单张图像，img_array 形状 (C,H,W), 范围 0-1"""
    os.makedirs(output_folder, exist_ok=True)
    if img_array.shape[0] == 3:
        img_array = numpy.transpose(img_array, (1, 2, 0))
    if len(img_array.shape) == 3 and img_array.shape[0] == 1 and img_array.shape[1] == 1:
        img_array = img_array.reshape(256, 256, 3)
    img = Image.fromarray((img_array * 255).astype(numpy.uint8))
    base = os.path.splitext(name)[0]  # 去掉扩展名，例如 "test_p295_"
    png_name = base + '.png'
    img.save(os.path.join(output_folder, png_name), format='PNG')

def run():
    torch.set_grad_enabled(False)
    data_range = 1
    win_size = 7
    win_sigma = 1.5
    K = (0.01, 0.03)

    for k in range(len(opt.data_test)):
        has_gt = (opt.label_test is not None and k < len(opt.label_test) and opt.label_test[k] is not None)
        compute = opt.compute_metrics and has_gt

        # ---- 提取数据集短名称（新添加的代码）----
        path = opt.data_test[k]
        ds_name = os.path.basename(os.path.dirname(path))   # 例如 'LSUI_test400'
        # -------------------------------------

        if compute:
            print(f'Evaluating with GT: {opt.data_test[k]}')
            dataset = get_eval_set(opt.data_test[k], opt.label_test[k])
        else:
            print(f'Inference only: {opt.data_test[k]}')
            dataset = get_infer_set(opt.data_test[k])

        loader = DataLoader(dataset, num_workers=opt.threads, batch_size=1, shuffle=False)

        PSNR_list = []
        SSIM_list = []
        total_time = 0

        for batch in loader:
            if compute:
                input, label, name = batch
                name = name[0]          # 解包元组
                input = input.cuda()
                label = label.cuda()
            else:
                input, name = batch
                name = name[0]
                input = input.cuda()

            start = time.time()
            out = model(input)
            total_time += time.time() - start

            # 保存图像
            out_np = numpy.clip(torch_to_np(out), 0, 1)
            output_dir = opt.output_folder[k]
            my_save_image(name, out_np, output_dir)

            if compute:
                psnr_val = 10 * torch.log10(1 / F.mse_loss(out, label)).item()
                ssim_val = ssim(out, label, data_range=data_range, win_size=win_size,
                                win_sigma=win_sigma, K=K, size_average=False).item()
                PSNR_list.append(psnr_val)
                SSIM_list.append(ssim_val)

        avg_time = total_time / len(dataset) * 1000  # 毫秒
        if compute:
            avg_psnr = numpy.mean(PSNR_list)
            avg_ssim = numpy.mean(SSIM_list)
            # 修改输出：显示数据集短名称而非序号
            print(f'{ds_name}: PSNR={avg_psnr:.4f} dB, SSIM={avg_ssim:.4f}, Time={avg_time:.2f}ms/img')
        else:
            print(f'Inference finished for {len(dataset)} images, saved to {opt.output_folder[k]}, Time={avg_time:.2f}ms/img')

        del input, out
        torch.cuda.empty_cache()

if __name__ == '__main__':
    run()