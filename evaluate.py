import os
import cv2
import numpy as np
import scipy.special
import scipy.io
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from utils.uqim_utils import getUIQM
import argparse

def uciqe(loc):
    img_bgr = cv2.imread(loc)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    coe_metric = [0.4680, 0.2745, 0.2576]
    img_lum = img_lab[..., 0] / 255
    img_a = img_lab[..., 1] / 255
    img_b = img_lab[..., 2] / 255
    img_chr = np.sqrt(np.square(img_a) + np.square(img_b))
    img_sat = img_chr / np.sqrt(np.square(img_chr) + np.square(img_lum))
    aver_sat = np.mean(img_sat)
    aver_chr = np.mean(img_chr)
    var_chr = np.sqrt(np.mean(abs(1 - np.square(aver_chr / (img_chr + 1e-8)))))
    dtype = img_lum.dtype
    nbins = 256 if dtype == 'uint8' else 65536
    hist, _ = np.histogram(img_lum, nbins)
    cdf = np.cumsum(hist) / np.sum(hist)
    ilow = np.where(cdf > 0.01)[0]
    ihigh = np.where(cdf >= 0.99)[0]
    if len(ilow) == 0 or len(ihigh) == 0:
        con_lum = 0.5
    else:
        tol = [ilow[0] / (nbins - 1), ihigh[0] / (nbins - 1)]
        con_lum = tol[1] - tol[0]
    quality_val = coe_metric[0] * var_chr + coe_metric[1] * con_lum + coe_metric[2] * aver_sat
    return quality_val

gamma_range = np.arange(0.2, 10, 0.001)
a = scipy.special.gamma(2.0 / gamma_range) ** 2
b = scipy.special.gamma(1.0 / gamma_range)
c = scipy.special.gamma(3.0 / gamma_range)
prec_gammas = a / (b * c)

def aggd_features(imdata):
    imdata = imdata.flatten()
    imdata2 = imdata * imdata
    left = imdata2[imdata < 0]
    right = imdata2[imdata >= 0]
    left_mean = np.sqrt(np.mean(left)) if len(left) > 0 else 0
    right_mean = np.sqrt(np.mean(right)) if len(right) > 0 else 0
    if right_mean == 0:
        gamma_hat = np.inf
    else:
        gamma_hat = left_mean / right_mean
    r_hat = (np.mean(np.abs(imdata)) ** 2) / (np.mean(imdata2) + 1e-8)
    rhat_norm = r_hat * (((gamma_hat ** 3 + 1) * (gamma_hat + 1)) / ((gamma_hat ** 2 + 1) ** 2))
    pos = np.argmin((prec_gammas - rhat_norm) ** 2)
    alpha = gamma_range[pos]
    gam1 = scipy.special.gamma(1.0 / alpha)
    gam2 = scipy.special.gamma(2.0 / alpha)
    gam3 = scipy.special.gamma(3.0 / alpha)
    aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
    bl = aggdratio * left_mean
    br = aggdratio * right_mean
    N = (br - bl) * (gam2 / gam1)
    return alpha, N, bl, br, left_mean, right_mean

def paired_product(new_im):
    h = np.roll(new_im, 1, axis=1)
    v = np.roll(new_im, 1, axis=0)
    d1 = np.roll(np.roll(new_im, 1, axis=0), 1, axis=1)
    d2 = np.roll(np.roll(new_im, 1, axis=0), -1, axis=1)
    return h * new_im, v * new_im, d1 * new_im, d2 * new_im

def gen_gauss_window(lw, sigma):
    sd = sigma ** 2
    lw = int(lw)
    weights = np.zeros(2 * lw + 1)
    weights[lw] = 1.0
    for i in range(1, lw + 1):
        tmp = np.exp(-0.5 * (i ** 2) / sd)
        weights[lw + i] = tmp
        weights[lw - i] = tmp
    return weights / weights.sum()

def compute_image_mscn_transform(image, C=1, avg_window=None):
    if avg_window is None:
        avg_window = gen_gauss_window(3, 7.0 / 6.0)
    mu = scipy.ndimage.correlate1d(image, avg_window, mode='constant')
    mu = scipy.ndimage.correlate1d(mu, avg_window, mode='constant')
    mu_sq = mu ** 2
    var = scipy.ndimage.correlate1d(image ** 2, avg_window, mode='constant')
    var = scipy.ndimage.correlate1d(var, avg_window, mode='constant')
    var = np.sqrt(np.abs(var - mu_sq))
    return (image - mu) / (var + C), var, mu

def extract_subband_feats(mscncoefs):
    feat = []
    alpha_m, N, bl, br, _, _ = aggd_features(mscncoefs)
    feat.append(alpha_m)
    feat.append((bl + br) / 2.0)
    pps = paired_product(mscncoefs)
    for pp in pps:
        alpha, N, bl, br, _, _ = aggd_features(pp)
        feat.append(alpha)
        feat.append(N)
        feat.append(bl)
        feat.append(br)
    return np.array(feat)

def extract_patches(img, patch_size):
    h, w = img.shape
    patches = []
    for i in range(0, h - patch_size + 1, patch_size):
        for j in range(0, w - patch_size + 1, patch_size):
            patches.append(img[i:i+patch_size, j:j+patch_size])
    feats = [extract_subband_feats(p) for p in patches]
    return np.array(feats)

def niqe_score(img, model_params):
    patch_size = 96
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.shape[0] < patch_size or img.shape[1] < patch_size:
        raise ValueError(f"Image too small for NIQE, need >= {patch_size}x{patch_size}")
    h, w = img.shape
    img = img[:h - (h % patch_size), :w - (w % patch_size)]
    img2 = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    mscn1, _, _ = compute_image_mscn_transform(img.astype(np.float32))
    mscn2, _, _ = compute_image_mscn_transform(img2.astype(np.float32))
    feats1 = extract_patches(mscn1, patch_size)
    feats2 = extract_patches(mscn2, patch_size // 2)
    feats = np.hstack((feats1, feats2))
    sample_mu = np.mean(feats, axis=0)
    sample_cov = np.cov(feats.T)
    pop_mu = model_params['mu']
    pop_cov = model_params['cov']
    X = sample_mu - pop_mu
    covmat = (pop_cov + sample_cov) / 2.0
    pinv = scipy.linalg.pinv(covmat)
    return np.sqrt(np.dot(np.dot(X, pinv), X))

def rmetrics(img1, img2):
    ssim = compare_ssim(img1, img2, channel_axis=2, data_range=255)
    psnr = compare_psnr(img1, img2, data_range=255)
    return psnr, ssim

def get_enhanced_image_path(result_path, gt_filename, method_name):
    corname, corexten = os.path.splitext(gt_filename)
    if method_name == "_hrmamba":
        fname = corname + '.png'
    else:
        fname = gt_filename
    return os.path.join(result_path, fname)

def process_method(input_root, gt_root, method_name, niqe_model, auto_gt=True):
    result_path = input_root + method_name
    if not os.path.isdir(result_path):
        print(f"Result directory {result_path} not found, skipping.")
        return

    has_gt = False
    gt_files = []
    if auto_gt and gt_root and os.path.isdir(gt_root):
        gt_files = [f for f in os.listdir(gt_root) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))]
        if gt_files:
            has_gt = True
            print(f"GT found for {result_path}, will compute PSNR/SSIM.")
        else:
            print("GT directory empty, running in no-reference mode.")
    else:
        print("No GT provided or auto_gt disabled, running in no-reference mode.")

    per_img_txt = os.path.join(result_path, f"{os.path.basename(input_root)}{method_name}_metrics.txt")
    summary_dir = os.path.dirname(input_root)
    summary_txt = os.path.join(summary_dir, f"{os.path.basename(input_root)}_metrics_all.txt")
    os.makedirs(summary_dir, exist_ok=True)
    open(per_img_txt, 'w').close()

    sum_psnr, sum_ssim, cnt_psnr = 0, 0, 0
    sum_uiqm, sum_uicqe, sum_niqe, cnt_noref = 0, 0, 0, 0

    if has_gt:
        for gt_file in gt_files:
            enhanced_path = get_enhanced_image_path(result_path, gt_file, method_name)
            if not os.path.exists(enhanced_path):
                print(f"Warning: Enhanced image not found: {enhanced_path}, skipping {gt_file}")
                continue
            img = cv2.imread(enhanced_path)
            if img is None:
                print(f"Warning: Cannot read {enhanced_path}, skipping")
                continue
            gt_img = cv2.imread(os.path.join(gt_root, gt_file))
            if gt_img is None:
                print(f"Warning: Cannot read GT {gt_file}, skipping")
                continue
            if gt_img.shape != img.shape:
                print(f"Shape mismatch: {gt_file} GT {gt_img.shape} vs enhanced {img.shape}, skipping")
                continue

            try:
                uiqm, _, _, _ = getUIQM(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            except Exception as e:
                print(f"UIQM error for {gt_file}: {e}")
                uiqm = -1
            try:
                uicqe_val = uciqe(enhanced_path)
            except Exception as e:
                print(f"UICQE error for {gt_file}: {e}")
                uicqe_val = -1
            niqe_val = -1
            if niqe_model:
                try:
                    niqe_val = niqe_score(img, niqe_model)
                except Exception as e:
                    print(f"NIQE error for {gt_file}: {e}")

            try:
                psnr_val, ssim_val = rmetrics(gt_img, img)
            except Exception as e:
                print(f"PSNR/SSIM error for {gt_file}: {e}")
                psnr_val = ssim_val = -1

            with open(per_img_txt, 'a') as f:
                f.write(f'{gt_file}: psnr={psnr_val:.4f} ssim={ssim_val:.4f} uiqm={uiqm:.4f} uicqe={uicqe_val:.4f} niqe={niqe_val:.4f}\n')

            if psnr_val >= 0:
                sum_psnr += psnr_val
                sum_ssim += ssim_val
                cnt_psnr += 1
            if uiqm >= 0 and uicqe_val >= 0 and niqe_val >= 0:
                sum_uiqm += uiqm
                sum_uicqe += uicqe_val
                sum_niqe += niqe_val
                cnt_noref += 1
    else:
        img_files = [f for f in os.listdir(result_path) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))]
        for img_file in img_files:
            img_path = os.path.join(result_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            try:
                uiqm, _, _, _ = getUIQM(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            except:
                uiqm = -1
            try:
                uicqe_val = uciqe(img_path)
            except:
                uicqe_val = -1
            niqe_val = -1
            if niqe_model:
                try:
                    niqe_val = niqe_score(img, niqe_model)
                except:
                    pass
            with open(per_img_txt, 'a') as f:
                f.write(f'{img_file}: uiqm={uiqm:.4f} uicqe={uicqe_val:.4f} niqe={niqe_val:.4f}\n')
            if uiqm >= 0 and uicqe_val >= 0 and niqe_val >= 0:
                sum_uiqm += uiqm
                sum_uicqe += uicqe_val
                sum_niqe += niqe_val
                cnt_noref += 1

    # 计算平均值
    avg_psnr = avg_ssim = avg_uiqm = avg_uicqe = avg_niqe = None
    if cnt_psnr > 0:
        avg_psnr = sum_psnr / cnt_psnr
        avg_ssim = sum_ssim / cnt_psnr
    if cnt_noref > 0:
        avg_uiqm = sum_uiqm / cnt_noref
        avg_uicqe = sum_uicqe / cnt_noref
        avg_niqe = sum_niqe / cnt_noref

    with open(summary_txt, 'a') as f:
        prefix = f"{os.path.basename(input_root)}{method_name}_Average:"
        line = prefix
        if avg_psnr is not None:
            line += f" psnr={avg_psnr:.4f} ssim={avg_ssim:.4f}"
        if avg_uiqm is not None:
            line += f" uiqm={avg_uiqm:.4f} uicqe={avg_uicqe:.4f} niqe={avg_niqe:.4f}"
        f.write(line + "\n")

    # 打印到控制台
    print(f"\n========== {os.path.basename(input_root)}{method_name} Average Results ==========")
    line_parts = []
    if avg_psnr is not None:
        line_parts.append(f"PSNR={avg_psnr:.4f} dB SSIM={avg_ssim:.4f}")
    if avg_uiqm is not None:
        line_parts.append(f"UIQM={avg_uiqm:.4f} UICQE={avg_uicqe:.4f} NIQE={avg_niqe:.4f}")
    if line_parts:
        print(" ".join(line_parts))
    print(f"Detailed per-image metrics saved to: {per_img_txt}")
    print(f"Summary appended to: {summary_txt}\n")

def main():
    parser = argparse.ArgumentParser(description="Evaluate underwater image enhancement metrics.")
    parser.add_argument('-i', '--input', action='append', default=[],
                        help="Input root directory (can be specified multiple times). Defaults to built-in list.")
    parser.add_argument('-g', '--gt', action='append', default=[],
                        help="Corresponding ground truth directory (same order as --input).")
    parser.add_argument('-m', '--method', action='append', default=[],
                        help="Method name suffix (e.g., _hrmamba). Can be multiple.")
    parser.add_argument('--model-params', default="utils/modelparameters.mat",
                        help="Path to NIQE model parameters .mat file.")
    parser.add_argument('--no-auto-gt', action='store_false', dest='auto_gt',
                        help="Disable automatic use of ground truth (no-reference only).")
    args = parser.parse_args()

    # 默认值（与原硬编码一致）
    default_inputs = [
        "./results/EUVP",
        "./results/LSUI",
        "./results/R90x256",
    ]
    default_gts = [
        './datasets/testDataset/Test_E515/GTr',
        './datasets/testDataset/Test_L400/gt',
        './datasets/testDataset/Test_R90/gt',
    ]
    default_methods = ["_hrmamba"]

    input_roots = args.input if args.input else default_inputs
    gt_roots = args.gt if args.gt else default_gts
    methods = args.method if args.method else default_methods

    if len(gt_roots) == 0:
        gt_roots = [None] * len(input_roots)
    elif len(gt_roots) != len(input_roots):
        print("Error: Number of --gt must match number of --input (or be empty).")
        sys.exit(1)

    # 加载 NIQE 模型
    if not os.path.exists(args.model_params):
        print(f"Warning: NIQE model file {args.model_params} not found. NIQE will be skipped.")
        niqe_model = None
    else:
        mat = scipy.io.loadmat(args.model_params)
        niqe_model = {'mu': np.ravel(mat['mu_prisparam']), 'cov': mat['cov_prisparam']}

    for idx, in_root in enumerate(input_roots):
        gt_root = gt_roots[idx] if idx < len(gt_roots) else None
        print(f"\nProcessing input root: {in_root}")
        for method in methods:
            process_method(in_root, gt_root, method, niqe_model, auto_gt=args.auto_gt)

if __name__ == '__main__':
    import sys
    main()