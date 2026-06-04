"""
dataset.py — 统一数据集
支持四种增强模式（对应实验三）：
  augment='none'       : 不做任何增强
  augment='flip'       : 上下翻转 + 水平翻转 + 90/180/270旋转
  augment='gamma_only' : 仅gamma随机校正（不做翻转）
  augment='gamma'      : flip增强 + gamma随机校正（组合增强）
"""
import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SIRRDataset(Dataset):
    IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

    def __init__(
        self,
        root_dir,
        input_label='haze',
        target_label='gt',
        patch_size=64,
        augment='none',        # 'none' | 'flip' | 'gamma_only' | 'gamma'
        gamma_range=(0.6, 1.4),  # gamma增强的范围
        transform=None,
    ):
        self.root_dir = root_dir
        self.input_dir = os.path.join(root_dir, input_label)
        self.target_dir = os.path.join(root_dir, target_label)
        self.patch_size = int(patch_size) if patch_size is not None else None
        self.augment_mode = augment
        self.gamma_range = gamma_range
        self.transform = transform

        if not os.path.isdir(self.input_dir):
            raise FileNotFoundError(f"Input folder not found: {self.input_dir}")
        if not os.path.isdir(self.target_dir):
            raise FileNotFoundError(f"Target folder not found: {self.target_dir}")

        # 收集配对文件
        input_files = sorted([
            f for f in os.listdir(self.input_dir)
            if os.path.isfile(os.path.join(self.input_dir, f))
            and f.lower().endswith(self.IMG_EXTS)
        ])

        paired = []
        missing = 0
        for f in input_files:
            gt_name = self._find_gt_name(f)
            if gt_name and os.path.exists(os.path.join(self.target_dir, gt_name)):
                paired.append((f, gt_name))
            else:
                missing += 1

        self.files = paired
        print(f"[Dataset] {len(self.files)} pairs in {root_dir} "
              f"(missing: {missing}, augment: {augment})")

    def _find_gt_name(self, input_name):
        """
        Haze4K: 769_0.72_1.83.png -> 769.png (ID在第一个下划线前)
        """
        # 先尝试同名
        if os.path.exists(os.path.join(self.target_dir, input_name)):
            return input_name
        # 提取ID
        base = os.path.splitext(input_name)[0]
        img_id = base.split('_')[0]
        for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            candidate = img_id + ext
            if os.path.exists(os.path.join(self.target_dir, candidate)):
                return candidate
        return None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        input_name, gt_name = self.files[idx]
        input_path = os.path.join(self.input_dir, input_name)
        target_path = os.path.join(self.target_dir, gt_name)

        input_img = Image.open(input_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')

        # 确保尺寸一致
        if input_img.size != target_img.size:
            target_img = target_img.resize(input_img.size, Image.BICUBIC)

        # 数据增强
        if self.augment_mode in ('flip', 'gamma'):
            input_img, target_img = self._augment_flip(input_img, target_img)

        if self.augment_mode in ('gamma', 'gamma_only'):
            input_img, target_img = self._augment_gamma(input_img, target_img)

        # 转 numpy -> tensor
        input_np = np.array(input_img).transpose(2, 0, 1).astype(np.float32) / 255.0
        target_np = np.array(target_img).transpose(2, 0, 1).astype(np.float32) / 255.0

        # Crop
        if self.patch_size is not None:
            input_np, target_np = self._crop_pair(input_np, target_np)

        input_tensor = torch.from_numpy(input_np)
        target_tensor = torch.from_numpy(target_np)

        if self.transform:
            input_tensor = self.transform(input_tensor)
            target_tensor = self.transform(target_tensor)

        return input_tensor, target_tensor

    def _crop_pair(self, img, gt):
        _, h, w = img.shape
        if h < self.patch_size or w < self.patch_size:
            # 图太小则resize
            from PIL import Image as PILImage
            img_pil = PILImage.fromarray((img.transpose(1, 2, 0) * 255).astype(np.uint8))
            gt_pil = PILImage.fromarray((gt.transpose(1, 2, 0) * 255).astype(np.uint8))
            img_pil = img_pil.resize((self.patch_size, self.patch_size), PILImage.BICUBIC)
            gt_pil = gt_pil.resize((self.patch_size, self.patch_size), PILImage.BICUBIC)
            img = np.array(img_pil).transpose(2, 0, 1).astype(np.float32) / 255.0
            gt = np.array(gt_pil).transpose(2, 0, 1).astype(np.float32) / 255.0
            return img, gt
        ix = random.randint(0, h - self.patch_size)
        iy = random.randint(0, w - self.patch_size)
        return img[:, ix:ix+self.patch_size, iy:iy+self.patch_size], \
               gt[:, ix:ix+self.patch_size, iy:iy+self.patch_size]

    def _augment_flip(self, img, gt):
        """上下翻转 + 水平翻转 + 随机旋转"""
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            gt = gt.transpose(Image.FLIP_TOP_BOTTOM)
        angle = random.choice([0, 90, 180, 270])
        if angle:
            img = img.rotate(angle, expand=False)
            gt = gt.rotate(angle, expand=False)
        return img, gt

    def _augment_gamma(self, img, gt):
        """
        Gamma校正增强：对输入有雾图和GT同时做相同的gamma变换
        gamma < 1 -> 提亮, gamma > 1 -> 压暗
        只改变亮度分布，不改变雾的相对关系
        """
        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        img_np = np.array(img).astype(np.float32) / 255.0
        gt_np = np.array(gt).astype(np.float32) / 255.0
        img_np = np.clip(np.power(img_np, gamma), 0, 1)
        gt_np = np.clip(np.power(gt_np, gamma), 0, 1)
        img = Image.fromarray((img_np * 255).astype(np.uint8))
        gt = Image.fromarray((gt_np * 255).astype(np.uint8))
        return img, gt
