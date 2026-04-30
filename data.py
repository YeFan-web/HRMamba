from torchvision.transforms import Compose, ToTensor
from dataset import DatasetFromFolderEval, DatasetFromFolder, DatasetFromFolderInfer


def transform():
    return Compose([
        ToTensor(),
    ])

from torchvision import transforms




def get_training_set(data_dir, label_dir, patch_size, data_augmentation):
    return DatasetFromFolder(data_dir, label_dir, patch_size, data_augmentation, transform=transform())


def get_eval_set(data_dir, label_dir):
    return DatasetFromFolderEval(data_dir, label_dir, transform=transform())

def get_infer_set(data_dir):
    """返回只包含输入图像的数据集，用于推理（无GT）"""
    return DatasetFromFolderInfer(data_dir, transform=transform())