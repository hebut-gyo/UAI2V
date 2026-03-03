import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import numpy as np
from typing import Dict, List, Optional
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ============================================================
# 损失函数（内置，无需额外文件）
# ============================================================

class WarpFlowLoss(nn.Module):
    """
    WarpResidualFlowModulator 的训练损失。

    总损失 = w1 * L_object + w2 * L_background + w3 * L_direction

    - L_object:     物体区域的 Smooth L1 损失（对幅度差异鲁棒）
    - L_background: 背景区域的 L1 损失（惩罚伪影扩散）
    - L_direction:  物体区域的方向一致性损失（1 - cosine similarity）
    """

    def __init__(
        self,
        w_object=1.0,
        w_background=0.1,
        w_direction=0.5,
        object_loss_type='smooth_l1',
        mask_threshold=1e-4,
    ):
        super().__init__()
        self.w_object = w_object
        self.w_background = w_background
        self.w_direction = w_direction
        self.object_loss_type = object_loss_type
        self.mask_threshold = mask_threshold

    def forward(self, pred, target):
        """
        pred:   [B, 2, T, H, W]
        target: [B, 2, T, H, W]

        返回: loss_dict，包含 'loss' 和各分量
        """
        # mask
        target_mask = (target.abs().sum(dim=1, keepdim=True) > self.mask_threshold).float()
        bg_mask = 1.0 - target_mask

        # 1. 物体区域损失
        if self.object_loss_type == 'smooth_l1':
            pixel_loss = F.smooth_l1_loss(pred, target, reduction='none', beta=1.0)
        elif self.object_loss_type == 'l1':
            pixel_loss = F.l1_loss(pred, target, reduction='none')
        else:
            pixel_loss = F.mse_loss(pred, target, reduction='none')

        n_obj = target_mask.sum().clamp(min=1.0)
        l_object = (pixel_loss * target_mask).sum() / n_obj

        # 2. 背景区域损失
        n_bg = bg_mask.sum().clamp(min=1.0)
        l_background = (pred.abs() * bg_mask).sum() / n_bg

        # 3. 方向一致性损失
        l_direction = self._direction_loss(pred, target, target_mask)

        total = (self.w_object * l_object
                 + self.w_background * l_background
                 + self.w_direction * l_direction)

        return {
            'loss': total,
            'l_object': l_object.detach(),
            'l_background': l_background.detach(),
            'l_direction': l_direction.detach(),
            'n_obj_pixels': n_obj.detach(),
        }

    def _direction_loss(self, pred, target, mask):
        pred_mag = pred.norm(dim=1, keepdim=True).clamp(min=1e-6)
        target_mag = target.norm(dim=1, keepdim=True).clamp(min=1e-6)

        motion_mask = (target_mag > 0.1).float() * mask
        n_motion = motion_mask.sum().clamp(min=1.0)
        if n_motion < 1.5:
            return torch.tensor(0.0, device=pred.device)

        cos_sim = (pred * target).sum(dim=1, keepdim=True) / (pred_mag * target_mag)
        dir_loss = (1.0 - cos_sim) * motion_mask
        return dir_loss.sum() / n_motion


# ============================================================
# 数据集
# ============================================================

class FlowModulatorDataset(Dataset):
    """
    光流调制器训练数据集

    数据结构：
    data_root/
    ├── camera_flows/          # 纯相机运动光流 (homography-based)
    │   └── camera_flow_XXXXXX.npz   # [13, 2, 60, 90]
    ├── object_flows/          # 静态视角物体光流
    │   └── obj_flow_XXXXXX.npz      # [13, 2, 60, 90]
    └── object_flows_gt/       # 动态视角物体光流 (GT)
        └── obj_flow_gt_XXXXXX.npz   # [13, 2, 60, 90]
    """

    def __init__(self, data_root: str, max_samples: Optional[int] = None):
        self.data_root = Path(data_root)

        self.camera_flows_dir = self.data_root / 'camera_flows'
        self.static_obj_dir = self.data_root / 'object_flows'
        self.gt_obj_dir = self.data_root / 'object_flows_gt'

        self._check_directories()
        self.samples = self._load_samples(max_samples)
        logger.info(f"Loaded {len(self.samples)} samples from {data_root}")

    def _check_directories(self):
        for dir_path, name in [
            (self.camera_flows_dir, 'camera_flows'),
            (self.static_obj_dir, 'object_flows'),
            (self.gt_obj_dir, 'object_flows_gt'),
        ]:
            if not dir_path.exists():
                raise FileNotFoundError(
                    f"Directory not found: {dir_path}\n"
                    f"Please ensure '{name}' exists in {self.data_root}\n"
                    f"Run preprocess_fixed.py to generate camera_flows/"
                )

    def _load_samples(self, max_samples: Optional[int] = None) -> List[Dict]:
        samples = []
        static_files = sorted(self.static_obj_dir.glob('obj_flow_*.npz'))
        if max_samples:
            static_files = static_files[:max_samples]

        for sf in static_files:
            idx = sf.stem.split('_')[-1]  # e.g. '000000'
            camera_f = self.camera_flows_dir / f'camera_flow_{idx}.npz'
            gt_f = self.gt_obj_dir / f'obj_flow_gt_{idx}.npz'

            if camera_f.exists() and gt_f.exists():
                samples.append({
                    'idx': idx,
                    'camera_flow': camera_f,
                    'static_obj': sf,
                    'gt_obj': gt_f,
                })
            else:
                missing = []
                if not camera_f.exists(): missing.append(str(camera_f))
                if not gt_f.exists(): missing.append(str(gt_f))
                logger.warning(f"Skipping sample {idx}: missing {missing}")

        if not samples:
            raise ValueError(f"No valid samples found in {self.data_root}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            # 所有 npz 的 'flows' 键形状为 [13, 2, 60, 90]
            # 转为 [2, 13, 60, 90] 即 [C, T, H, W]
            camera_flow = self._load_npz(sample['camera_flow'])
            static_obj = self._load_npz(sample['static_obj'])
            gt_obj = self._load_npz(sample['gt_obj'])

            return {
                'idx': sample['idx'],
                'camera_flow': camera_flow,   # [2, T, H, W]
                'static_obj': static_obj,     # [2, T, H, W]
                'gt_obj': gt_obj,             # [2, T, H, W]
            }
        except Exception as e:
            logger.error(f"Error loading sample {sample['idx']}: {e}")
            raise

    def _load_npz(self, path: Path) -> torch.Tensor:
        data = np.load(path)
        arr = data['flows']  # [13, 2, 60, 90]
        t = torch.from_numpy(arr).float()
        # [T, 2, H, W] → [2, T, H, W]
        if t.ndim == 4 and t.shape[1] == 2:
            t = t.permute(1, 0, 2, 3)
        return t


# ============================================================
# 训练器
# ============================================================

class FlowModulatorTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.global_step = 0

        # tensorboard
        self.writer = None
        if not args.test_only:
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = Path(args.output_dir) / 'logs'
                log_dir.mkdir(parents=True, exist_ok=True)
                self.writer = SummaryWriter(str(log_dir))
                logger.info(f"TensorBoard logging to {log_dir}")
            except ImportError:
                logger.warning("tensorboard not installed, skipping logging")

        # 数据集：训练集和验证集在不同目录
        self.train_dataset = FlowModulatorDataset(args.data_root, args.max_samples)
        val_root = Path(args.val_root) if args.val_root else Path(args.data_root).parent / 'test'
        if val_root.exists():
            self.val_dataset = FlowModulatorDataset(str(val_root), args.max_samples)
            logger.info(f"Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)} (from {val_root})")
        else:
            self.val_dataset = None
            logger.warning(f"Val directory not found: {val_root}, skipping validation")

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = None
        if self.val_dataset is not None:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )

        # 模型
        from modules.traj_moudule import WarpResidualFlowModulator
        self.model = WarpResidualFlowModulator(
            hidden=32,
            num_res_blocks=2,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {n_params:,}")

        # 损失函数
        self.criterion = WarpFlowLoss(
            w_object=args.w_object,
            w_background=args.w_background,
            w_direction=args.w_direction,
        )

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        # 学习率调度：OneCycleLR（快速上升到峰值，再平缓下降）
        steps_per_epoch = len(self.train_loader)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=args.learning_rate,
            epochs=args.num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.05,       # 前5%步数做warmup（很短）
            anneal_strategy='cos',
            div_factor=5,         # 初始lr = max_lr/5 = 1e-3
            final_div_factor=50,  # 最终lr = max_lr/250 = 2e-5
        )

    def train(self):
        best_val_loss = float('inf')

        for epoch in range(self.args.num_epochs):
            # ── 训练 ──
            train_metrics = self._train_epoch(epoch)

            # ── 验证 ──
            val_metrics = None
            if self.val_loader is not None:
                val_metrics = self._validate(epoch)

            # ── 日志 ──
            lr = self.optimizer.param_groups[0]['lr']
            log_msg = (
                f"Epoch {epoch:03d} | "
                f"Train loss={train_metrics['loss']:.6f} "
                f"(obj={train_metrics['l_object']:.6f} "
                f"bg={train_metrics['l_background']:.6f} "
                f"dir={train_metrics['l_direction']:.6f})"
            )
            if val_metrics:
                log_msg += f" | Val loss={val_metrics['loss']:.6f}"
            log_msg += f" | LR={lr:.2e}"
            logger.info(log_msg)

            if self.writer:
                self.writer.add_scalar('epoch/train_loss', train_metrics['loss'], epoch)
                self.writer.add_scalar('epoch/lr', lr, epoch)
                for k in ['l_object', 'l_background', 'l_direction']:
                    self.writer.add_scalar(f'epoch/train_{k}', train_metrics[k], epoch)
                if val_metrics:
                    self.writer.add_scalar('epoch/val_loss', val_metrics['loss'], epoch)
                    for k in ['l_object', 'l_background', 'l_direction']:
                        self.writer.add_scalar(f'epoch/val_{k}', val_metrics[k], epoch)

            # ── 保存 ──
            if (epoch + 1) % self.args.save_interval == 0:
                self._save_checkpoint(epoch, train_metrics['loss'], tag='latest')

            if val_metrics and val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self._save_checkpoint(epoch, val_metrics['loss'], tag='best')
                logger.info(f"  ↑ New best val loss: {best_val_loss:.6f}")

        if self.writer:
            self.writer.close()

    def _train_epoch(self, epoch):
        self.model.train()
        accum = {'loss': 0, 'l_object': 0, 'l_background': 0, 'l_direction': 0}
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Train {epoch:03d}", ncols=120)
        for batch in pbar:
            static_obj = batch['static_obj'].to(self.device)     # [B,2,T,H,W]
            camera_flow = batch['camera_flow'].to(self.device)   # [B,2,T,H,W]
            gt_obj = batch['gt_obj'].to(self.device)             # [B,2,T,H,W]

            # 前向
            pred = self.model(static_obj, camera_flow)

            # 损失
            loss_dict = self.criterion(pred, gt_obj)
            loss = loss_dict['loss']

            # 跳过无物体的 batch
            if loss_dict['n_obj_pixels'] < 10:
                continue

            # 反向
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )
            self.optimizer.step()
            self.scheduler.step()

            # 累计
            for k in accum:
                accum[k] += loss_dict[k].item() if k != 'loss' else loss.item()
            n_batches += 1
            self.global_step += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'obj': f"{loss_dict['l_object'].item():.4f}",
                'dir': f"{loss_dict['l_direction'].item():.4f}",
                'grad': f"{grad_norm:.2f}",
            })

            # step 级别 tensorboard
            if self.writer and self.global_step % self.args.log_interval == 0:
                self.writer.add_scalar('step/loss', loss.item(), self.global_step)
                self.writer.add_scalar('step/grad_norm', grad_norm, self.global_step)
                self.writer.add_scalar('step/lr',
                                       self.optimizer.param_groups[0]['lr'],
                                       self.global_step)

        n_batches = max(n_batches, 1)
        return {k: v / n_batches for k, v in accum.items()}

    @torch.no_grad()
    def _validate(self, epoch):
        self.model.eval()
        accum = {'loss': 0, 'l_object': 0, 'l_background': 0, 'l_direction': 0}
        n_batches = 0

        for batch in self.val_loader:
            static_obj = batch['static_obj'].to(self.device)
            camera_flow = batch['camera_flow'].to(self.device)
            gt_obj = batch['gt_obj'].to(self.device)

            pred = self.model(static_obj, camera_flow)
            loss_dict = self.criterion(pred, gt_obj)

            if loss_dict['n_obj_pixels'] < 10:
                continue

            for k in accum:
                accum[k] += loss_dict[k].item() if k != 'loss' else loss_dict['loss'].item()
            n_batches += 1

        n_batches = max(n_batches, 1)
        return {k: v / n_batches for k, v in accum.items()}

    def _save_checkpoint(self, epoch, loss, tag='latest'):
        out_dir = Path(self.args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        path = out_dir / f"flow_modulator_{tag}.pth"
        torch.save({
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'args': vars(self.args),
        }, path)
        logger.info(f"Checkpoint saved: {path}")


# ============================================================
# 主函数
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train WarpResidualFlowModulator')

    # 数据
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--val_root', type=str, default=None,
                        help='Validation data root. Default: {data_root}/../test')
    parser.add_argument('--output_dir', type=str,
                        default='/data/gy/CogVideo/finetune/outputs/flow_modulator')
    parser.add_argument('--max_samples', type=int, default=None)

    # 训练
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=5e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--warmup_steps', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')

    # 损失权重
    parser.add_argument('--w_object', type=float, default=1.0)
    parser.add_argument('--w_background', type=float, default=0.01)
    parser.add_argument('--w_direction', type=float, default=0.1)

    # 日志 & 保存
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=5)

    # 模式
    parser.add_argument('--test_only', action='store_true')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    if args.test_only:
        # 测试数据加载
        dataset = FlowModulatorDataset(args.data_root, args.max_samples)
        print(f"\nDataset size: {len(dataset)}")
        print(f"Testing data loading...\n")

        for i in tqdm(range(min(len(dataset), 10)), desc="Loading samples"):
            sample = dataset[i]
            print(f"  [{i}] camera_flow={sample['camera_flow'].shape} "
                  f"static_obj={sample['static_obj'].shape} "
                  f"gt_obj={sample['gt_obj'].shape}")

            # 检查数值合理性
            gt = sample['gt_obj']
            static = sample['static_obj']
            cam = sample['camera_flow']
            gt_mask = (gt.abs().sum(dim=0, keepdim=True) > 1e-4).float()
            obj_ratio = gt_mask.mean().item()
            print(f"       obj_ratio={obj_ratio:.4f} "
                  f"static_mag={static.norm(dim=0).mean():.4f} "
                  f"camera_mag={cam.norm(dim=0).mean():.4f} "
                  f"gt_mag={gt.norm(dim=0).mean():.4f}")

        print("\nData loading test passed!")
    else:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        trainer = FlowModulatorTrainer(args)
        trainer.train()


if __name__ == '__main__':
    main()