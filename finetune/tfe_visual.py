"""
WarpResidualFlowModulator v2 可视化工具

Baseline 改为 Simple Add: static + camera * mask
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
plt.rcParams['font.family'] = 'DejaVu Sans'


class FlowModulatorDataset(Dataset):
    def __init__(self, data_root: str, max_samples: int = None):
        self.data_root = Path(data_root)
        self.camera_dir = self.data_root / 'camera_flows'
        self.static_dir = self.data_root / 'object_flows'
        self.gt_dir = self.data_root / 'object_flows_gt'
        for d, name in [(self.camera_dir, 'camera_flows'),
                        (self.static_dir, 'object_flows'),
                        (self.gt_dir, 'object_flows_gt')]:
            if not d.exists():
                raise FileNotFoundError(f"{name}/ not found in {self.data_root}")
        self.samples = self._load_samples(max_samples)
        logger.info(f"Loaded {len(self.samples)} samples from {data_root}")

    def _load_samples(self, max_samples):
        samples = []
        static_files = sorted(self.static_dir.glob('obj_flow_*.npz'))
        if max_samples:
            static_files = static_files[:max_samples]
        for sf in static_files:
            idx = sf.stem.split('_')[-1]
            cam_f = self.camera_dir / f'camera_flow_{idx}.npz'
            gt_f = self.gt_dir / f'obj_flow_gt_{idx}.npz'
            if cam_f.exists() and gt_f.exists():
                samples.append({'idx': idx, 'camera': cam_f, 'static': sf, 'gt': gt_f})
        if not samples:
            raise ValueError(f"No valid samples in {self.data_root}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'idx': s['idx'],
            'camera_flow': self._load(s['camera']),
            'static_obj': self._load(s['static']),
            'gt_obj': self._load(s['gt']),
        }

    @staticmethod
    def _load(path):
        arr = np.load(str(path))['flows']
        t = torch.from_numpy(arr).float()
        if t.ndim == 4 and t.shape[1] == 2:
            t = t.permute(1, 0, 2, 3)
        return t


class FlowModulatorVisualizer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        from modules.traj_moudule import WarpResidualFlowModulator
        self.model = WarpResidualFlowModulator(
            hidden=32, num_res_blocks=2,
        ).to(self.device)

        ckpt = torch.load(args.checkpoint_path, map_location='cpu')
        self.model.load_state_dict(ckpt.get('model_state_dict', ckpt))
        self.model.eval()
        logger.info(f"Loaded checkpoint: epoch={ckpt.get('epoch','?')}, loss={ckpt.get('loss','?')}")

        self.dataset = FlowModulatorDataset(args.data_root, args.max_samples)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=0)

    @torch.no_grad()
    def visualize_all(self):
        all_epe_pred, all_epe_static, all_epe_add = [], [], []

        for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Visualizing")):
            if batch_idx >= self.args.num_samples:
                break

            static_obj = batch['static_obj'].to(self.device)
            camera_flow = batch['camera_flow'].to(self.device)
            gt_obj = batch['gt_obj'].to(self.device)

            pred = self.model(static_obj, camera_flow)

            # Simple Add baseline
            mask = (static_obj.abs().sum(dim=1, keepdim=True) > 1e-4).float()
            simple_add = static_obj + camera_flow * mask

            static_np = static_obj[0].cpu().numpy()
            camera_np = camera_flow[0].cpu().numpy()
            pred_np = pred[0].cpu().numpy()
            gt_np = gt_obj[0].cpu().numpy()
            add_np = simple_add[0].cpu().numpy()

            metrics = self._visualize_sample(
                batch_idx, static_np, camera_np, pred_np, gt_np, add_np
            )
            all_epe_pred.append(metrics['epe_pred'])
            all_epe_static.append(metrics['epe_static'])
            all_epe_add.append(metrics['epe_add'])

        if all_epe_pred:
            self._summary_report(all_epe_pred, all_epe_static, all_epe_add)

    def _visualize_sample(self, idx, static_np, camera_np, pred_np, gt_np, add_np):
        _, T, H, W = static_np.shape
        key_frames = [0, T // 2, T - 1]
        frame_labels = ['First', 'Middle', 'Last']

        fig = plt.figure(figsize=(22, 28))
        gs = GridSpec(6, 3, figure=fig, hspace=0.35, wspace=0.25)

        rows = [
            ('Static Object Flow', static_np, '#2196F3'),
            ('Camera Flow', camera_np, '#FF9800'),
            ('Simple Add (static+camera)', add_np, '#9C27B0'),
            ('Predicted (model)', pred_np, '#F44336'),
            ('Ground Truth', gt_np, '#4CAF50'),
        ]

        for row_i, (label, flow_arr, color) in enumerate(rows):
            for col_i, (fi, fname) in enumerate(zip(key_frames, frame_labels)):
                ax = fig.add_subplot(gs[row_i, col_i])
                rgb = self._flow_to_color(flow_arr[:, fi])
                ax.imshow(rgb)
                if col_i == 0:
                    ax.set_ylabel(label, fontsize=10, fontweight='bold', color=color)
                if row_i == 0:
                    ax.set_title(fname, fontsize=11, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])

        epe_pred = self._epe_per_frame(pred_np, gt_np)
        epe_static = self._epe_per_frame(static_np, gt_np)
        epe_add = self._epe_per_frame(add_np, gt_np)

        # 5a: EPE over time
        ax1 = fig.add_subplot(gs[5, 0])
        frames = range(T)
        ax1.plot(frames, epe_pred, 'r-', lw=2, label=f'Pred ({epe_pred.mean():.3f})')
        ax1.plot(frames, epe_add, 'm--', lw=1.5, label=f'Add ({epe_add.mean():.3f})')
        ax1.plot(frames, epe_static, 'b:', lw=1.5, label=f'Static ({epe_static.mean():.3f})')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('EPE (px)')
        ax1.set_title('End-Point Error', fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 5b: Magnitude over time
        ax2 = fig.add_subplot(gs[5, 1])
        mag_pred = self._mag_per_frame(pred_np)
        mag_gt = self._mag_per_frame(gt_np)
        mag_static = self._mag_per_frame(static_np)
        mag_cam = self._mag_per_frame(camera_np)
        ax2.plot(frames, mag_gt, 'g-', lw=2.5, label='GT')
        ax2.plot(frames, mag_pred, 'r--', lw=2, label='Pred')
        ax2.plot(frames, mag_static, 'b:', lw=1.5, label='Static')
        ax2.plot(frames, mag_cam, 'm-.', lw=1, label='Camera', alpha=0.6)
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Mean Magnitude (px)')
        ax2.set_title('Flow Magnitude', fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 5c: Model vs Simple Add improvement
        ax3 = fig.add_subplot(gs[5, 2])
        improvement = epe_add - epe_pred
        ax3.bar(frames, improvement, color=['green' if v > 0 else 'red' for v in improvement],
                alpha=0.7, width=0.8)
        ax3.axhline(0, color='black', lw=0.5)
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('EPE Reduction (px)')
        ax3.set_title('Model vs Simple Add (green=model better)', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        imp_add = (epe_add.mean() - epe_pred.mean()) / (epe_add.mean() + 1e-8) * 100
        imp_static = (epe_static.mean() - epe_pred.mean()) / (epe_static.mean() + 1e-8) * 100
        fig.suptitle(
            f'Sample {idx}  |  EPE: {epe_pred.mean():.4f}px  |  '
            f'vs Add: {imp_add:+.1f}%  |  vs Static: {imp_static:+.1f}%',
            fontsize=14, fontweight='bold', y=0.995
        )

        plt.savefig(self.output_dir / f'sample_{idx:04d}.png', dpi=120, bbox_inches='tight')
        plt.close()

        return {
            'epe_pred': epe_pred.mean(),
            'epe_static': epe_static.mean(),
            'epe_add': epe_add.mean(),
        }

    def _summary_report(self, epe_pred, epe_static, epe_add):
        epe_pred = np.array(epe_pred)
        epe_static = np.array(epe_static)
        epe_add = np.array(epe_add)
        n = len(epe_pred)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        ax = axes[0]
        ax.hist(epe_pred, bins=20, alpha=0.7, color='red', label='Pred')
        ax.hist(epe_add, bins=20, alpha=0.5, color='purple', label='Simple Add')
        ax.hist(epe_static, bins=20, alpha=0.3, color='blue', label='Static')
        ax.set_xlabel('Mean EPE (px)')
        ax.set_ylabel('Count')
        ax.set_title('EPE Distribution', fontweight='bold')
        ax.legend()

        ax = axes[1]
        x = range(n)
        ax.plot(x, epe_static, 'b:', lw=1, label='Static', alpha=0.6)
        ax.plot(x, epe_add, 'm--', lw=1.5, label='Simple Add')
        ax.plot(x, epe_pred, 'r-', lw=2, label='Pred')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Mean EPE (px)')
        ax.set_title('Per-Sample EPE', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.axis('off')
        imp_vs_add = (epe_add.mean() - epe_pred.mean()) / (epe_add.mean() + 1e-8) * 100
        imp_vs_static = (epe_static.mean() - epe_pred.mean()) / (epe_static.mean() + 1e-8) * 100
        table_data = [
            ['Metric', 'Static', 'Simple Add', 'Pred (ours)'],
            ['Mean EPE', f'{epe_static.mean():.4f}', f'{epe_add.mean():.4f}', f'{epe_pred.mean():.4f}'],
            ['Std EPE', f'{epe_static.std():.4f}', f'{epe_add.std():.4f}', f'{epe_pred.std():.4f}'],
            ['Min EPE', f'{epe_static.min():.4f}', f'{epe_add.min():.4f}', f'{epe_pred.min():.4f}'],
            ['Max EPE', f'{epe_static.max():.4f}', f'{epe_add.max():.4f}', f'{epe_pred.max():.4f}'],
            ['', '', '', ''],
            ['vs Add', '', '', f'{imp_vs_add:+.1f}%'],
            ['vs Static', '', '', f'{imp_vs_static:+.1f}%'],
        ]
        table = ax.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.5)
        for j in range(4):
            table[0, j].set_text_props(fontweight='bold')
        ax.set_title('Summary', fontweight='bold', pad=20)

        fig.suptitle(
            f'Overall Summary ({n} samples)  |  '
            f'Pred EPE: {epe_pred.mean():.4f}px  |  '
            f'vs Add: {imp_vs_add:+.1f}%  |  vs Static: {imp_vs_static:+.1f}%',
            fontsize=13, fontweight='bold'
        )

        plt.savefig(self.output_dir / 'summary.png', dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(
            f"\n{'='*60}\n"
            f"SUMMARY ({n} samples)\n"
            f"  Static EPE:     {epe_static.mean():.4f} +/- {epe_static.std():.4f}\n"
            f"  Simple Add EPE: {epe_add.mean():.4f} +/- {epe_add.std():.4f}\n"
            f"  Pred EPE:       {epe_pred.mean():.4f} +/- {epe_pred.std():.4f}\n"
            f"  vs Add:    {imp_vs_add:+.1f}%\n"
            f"  vs Static: {imp_vs_static:+.1f}%\n"
            f"{'='*60}"
        )

    @staticmethod
    def _flow_to_color(flow_frame):
        dx, dy = flow_frame[0].astype(np.float32), flow_frame[1].astype(np.float32)
        mag, ang = cv2.cartToPolar(dx, dy)
        H, W = dx.shape
        hsv = np.zeros((H, W, 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    @staticmethod
    def _epe_per_frame(flow_a, flow_b):
        diff = flow_a - flow_b
        ep = np.sqrt(diff[0] ** 2 + diff[1] ** 2)
        mask = np.sqrt(flow_b[0] ** 2 + flow_b[1] ** 2) > 1e-4
        result = np.zeros(ep.shape[0])
        for t in range(ep.shape[0]):
            if mask[t].any():
                result[t] = ep[t][mask[t]].mean()
        return result

    @staticmethod
    def _mag_per_frame(flow):
        mag = np.sqrt(flow[0] ** 2 + flow[1] ** 2)
        mask = mag > 1e-4
        result = np.zeros(mag.shape[0])
        for t in range(mag.shape[0]):
            if mask[t].any():
                result[t] = mag[t][mask[t]].mean()
        return result


def main():
    parser = argparse.ArgumentParser(description='WarpResidualFlowModulator Visualization')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/flow_modulator_vis')
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    vis = FlowModulatorVisualizer(args)
    vis.visualize_all()
    logger.info("Done!")


if __name__ == '__main__':
    main()