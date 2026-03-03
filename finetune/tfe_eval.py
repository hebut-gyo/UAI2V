"""
WarpResidualFlowModulator v2 评估器

Baseline 改为 Simple Add: static + camera * mask
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from typing import Dict, List
import logging
from tqdm import tqdm
import json
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


class FlowModulatorEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        from modules.traj_moudule import WarpResidualFlowModulator
        self.model = WarpResidualFlowModulator(
            hidden=32, num_res_blocks=2,
        ).to(self.device)

        ckpt = torch.load(args.checkpoint_path, map_location='cpu')
        self.model.load_state_dict(ckpt.get('model_state_dict', ckpt))
        self.model.eval()
        logger.info(f"Loaded checkpoint: epoch={ckpt.get('epoch','?')}, loss={ckpt.get('loss','?')}")

        self.dataset = FlowModulatorDataset(args.data_root, args.max_samples)
        self.dataloader = DataLoader(
            self.dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers, pin_memory=True,
        )

        if args.visualize:
            self.vis_dir = Path(args.output_dir) / 'visualizations'
            self.vis_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def evaluate(self):
        logger.info(f"Evaluating on {len(self.dataset)} samples...")
        all_metrics: List[Dict] = []

        for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Evaluating")):
            static_obj = batch['static_obj'].to(self.device)
            camera_flow = batch['camera_flow'].to(self.device)
            gt_obj = batch['gt_obj'].to(self.device)

            # 模型推理
            pred = self.model(static_obj, camera_flow)

            # Simple Add baseline: static + camera (物体区域)
            mask = (static_obj.abs().sum(dim=1, keepdim=True) > 1e-4).float()
            simple_add = static_obj + camera_flow * mask

            for i in range(static_obj.shape[0]):
                m = self._compute_sample_metrics(
                    pred[i], gt_obj[i], static_obj[i], simple_add[i], camera_flow[i]
                )
                m['idx'] = batch['idx'][i]
                all_metrics.append(m)

            if self.args.visualize and batch_idx < self.args.num_visualize:
                for i in range(min(static_obj.shape[0], self.args.num_visualize - batch_idx * self.args.batch_size)):
                    global_i = batch_idx * self.args.batch_size + i
                    self._visualize_sample(
                        global_i,
                        static_obj[i], camera_flow[i],
                        simple_add[i], pred[i], gt_obj[i]
                    )

        summary = self._summarize(all_metrics)
        self._save_results(summary, all_metrics)
        self._print_report(summary)
        return summary

    def _compute_sample_metrics(self, pred, gt, static, simple_add, camera) -> Dict:
        mask = (gt.abs().sum(dim=0, keepdim=True) > 1e-4).float()
        n_obj = mask.sum().clamp(min=1.0)

        if n_obj < 10:
            return {k: 0.0 for k in [
                'epe_pred', 'epe_simple_add', 'epe_static',
                'angle_error', 'relative_error',
                'magnitude_error', 'temporal_smoothness',
                'obj_pixel_ratio', 'camera_magnitude',
            ]}

        metrics = {}
        metrics['epe_pred'] = self._masked_epe(pred, gt, mask, n_obj)
        metrics['epe_simple_add'] = self._masked_epe(simple_add, gt, mask, n_obj)
        metrics['epe_static'] = self._masked_epe(static, gt, mask, n_obj)

        gt_mag = gt.norm(dim=0, keepdim=True)
        motion_mask = (gt_mag > 0.1).float() * mask
        n_motion = motion_mask.sum().clamp(min=1.0)

        if n_motion > 1:
            cos_sim = (pred * gt).sum(dim=0, keepdim=True) / (
                pred.norm(dim=0, keepdim=True).clamp(min=1e-6) *
                gt.norm(dim=0, keepdim=True).clamp(min=1e-6)
            )
            angle = torch.acos(cos_sim.clamp(-1, 1)) * 180 / np.pi
            metrics['angle_error'] = (angle * motion_mask).sum().item() / n_motion.item()
        else:
            metrics['angle_error'] = 0.0

        rel_err = self._epe_map(pred, gt) / gt_mag.clamp(min=1e-4)
        metrics['relative_error'] = (rel_err * motion_mask).sum().item() / n_motion.item()

        pred_mag = pred.norm(dim=0, keepdim=True)
        mag_err = (pred_mag - gt_mag).abs()
        metrics['magnitude_error'] = (mag_err * mask).sum().item() / n_obj.item()

        if pred.shape[1] > 1:
            pred_diff = (pred[:, 1:] - pred[:, :-1]).norm(dim=0, keepdim=True)
            gt_diff = (gt[:, 1:] - gt[:, :-1]).norm(dim=0, keepdim=True)
            temp_err = (pred_diff - gt_diff).abs()
            mask_t = mask[:, 1:]
            n_t = mask_t.sum().clamp(min=1.0)
            metrics['temporal_smoothness'] = (temp_err * mask_t).sum().item() / n_t.item()
        else:
            metrics['temporal_smoothness'] = 0.0

        total_pixels = gt.shape[1] * gt.shape[2] * gt.shape[3]
        metrics['obj_pixel_ratio'] = n_obj.item() / total_pixels
        metrics['camera_magnitude'] = camera.norm(dim=0).mean().item()

        return metrics

    @staticmethod
    def _epe_map(a, b):
        return (a - b).norm(dim=0, keepdim=True)

    @staticmethod
    def _masked_epe(a, b, mask, n_obj):
        epe = (a - b).norm(dim=0, keepdim=True)
        return (epe * mask).sum().item() / n_obj.item()

    def _summarize(self, all_metrics: List[Dict]) -> Dict:
        valid = [m for m in all_metrics if m.get('epe_pred', 0) > 0 or m.get('epe_static', 0) > 0]
        if not valid:
            return {}

        keys = ['epe_pred', 'epe_simple_add', 'epe_static',
                'angle_error', 'relative_error',
                'magnitude_error', 'temporal_smoothness']

        summary = {}
        for k in keys:
            vals = [m[k] for m in valid if k in m]
            if vals:
                summary[k] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'median': float(np.median(vals)),
                    'min': float(np.min(vals)),
                    'max': float(np.max(vals)),
                }

        epe_pred = np.array([m['epe_pred'] for m in valid])
        epe_add = np.array([m['epe_simple_add'] for m in valid])
        epe_static = np.array([m['epe_static'] for m in valid])

        summary['improvement_vs_simple_add'] = float((epe_add.mean() - epe_pred.mean()) / (epe_add.mean() + 1e-8))
        summary['improvement_vs_static'] = float((epe_static.mean() - epe_pred.mean()) / (epe_static.mean() + 1e-8))
        summary['n_samples'] = len(valid)

        cam_mags = np.array([m['camera_magnitude'] for m in valid])
        thresholds = [('low_camera', cam_mags < np.percentile(cam_mags, 33)),
                      ('mid_camera', (cam_mags >= np.percentile(cam_mags, 33)) & (cam_mags < np.percentile(cam_mags, 67))),
                      ('high_camera', cam_mags >= np.percentile(cam_mags, 67))]

        for group_name, group_mask in thresholds:
            if group_mask.sum() > 0:
                summary[f'{group_name}_epe_pred'] = float(epe_pred[group_mask].mean())
                summary[f'{group_name}_epe_add'] = float(epe_add[group_mask].mean())
                summary[f'{group_name}_epe_static'] = float(epe_static[group_mask].mean())
                summary[f'{group_name}_n'] = int(group_mask.sum())

        return summary

    def _visualize_sample(self, idx, static, camera, simple_add, pred, gt):
        flows = {
            'static': static, 'camera': camera,
            'simple_add': simple_add, 'pred': pred, 'gt': gt,
        }
        for name, flow in flows.items():
            video = self._flow_to_video(flow.cpu().numpy())
            path = self.vis_dir / f'sample_{idx:04d}_{name}.mp4'
            self._save_video(video, path)

    @staticmethod
    def _flow_to_video(flow: np.ndarray) -> np.ndarray:
        T = flow.shape[1]
        frames = []
        for t in range(T):
            dx, dy = flow[0, t].astype(np.float32), flow[1, t].astype(np.float32)
            mag, ang = cv2.cartToPolar(dx, dy)
            H, W = dx.shape
            hsv = np.zeros((H, W, 3), dtype=np.uint8)
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 1] = 255
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            frames.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
        return np.stack(frames)

    @staticmethod
    def _save_video(video: np.ndarray, path: Path):
        T, H, W, _ = video.shape
        out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 8.0, (W, H))
        for frame in video:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()

    def _save_results(self, summary, all_metrics):
        out_dir = Path(self.args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / 'evaluation_results.json', 'w') as f:
            json.dump(summary, f, indent=2)

        csv_path = out_dir / 'per_sample_metrics.csv'
        keys = ['idx', 'epe_pred', 'epe_simple_add', 'epe_static',
                'angle_error', 'magnitude_error', 'camera_magnitude', 'obj_pixel_ratio']
        with open(csv_path, 'w') as f:
            f.write(','.join(keys) + '\n')
            for m in all_metrics:
                vals = [str(m.get(k, '')) for k in keys]
                f.write(','.join(vals) + '\n')

        logger.info(f"Results saved to {out_dir}")

    def _print_report(self, summary):
        if not summary:
            print("No valid samples to evaluate!")
            return

        n = summary.get('n_samples', 0)
        print(f"\n{'='*70}")
        print(f"EVALUATION REPORT  ({n} samples)")
        print(f"{'='*70}")

        print(f"\n--- EPE (End-Point Error, lower is better) ---")
        for method in ['epe_static', 'epe_simple_add', 'epe_pred']:
            if method in summary:
                s = summary[method]
                label = method.replace('epe_', '').replace('_', ' ').capitalize()
                print(f"  {label:12s}: {s['mean']:.4f} +/- {s['std']:.4f}  "
                      f"(med={s['median']:.4f}, range=[{s['min']:.4f}, {s['max']:.4f}])")

        print(f"\n--- Improvement ---")
        print(f"  vs Simple Add: {summary.get('improvement_vs_simple_add', 0)*100:+.1f}%")
        print(f"  vs Static:     {summary.get('improvement_vs_static', 0)*100:+.1f}%")

        print(f"\n--- Other Metrics ---")
        for k in ['angle_error', 'magnitude_error', 'temporal_smoothness']:
            if k in summary:
                s = summary[k]
                unit = 'deg' if 'angle' in k else 'px'
                print(f"  {k:25s}: {s['mean']:.4f} +/- {s['std']:.4f} {unit}")

        print(f"\n--- By Camera Motion ---")
        for group in ['low_camera', 'mid_camera', 'high_camera']:
            n_g = summary.get(f'{group}_n', 0)
            if n_g > 0:
                ep = summary.get(f'{group}_epe_pred', 0)
                ea = summary.get(f'{group}_epe_add', 0)
                es = summary.get(f'{group}_epe_static', 0)
                imp = (ea - ep) / (ea + 1e-8) * 100
                print(f"  {group:12s} (n={n_g:3d}): "
                      f"pred={ep:.4f}  add={ea:.4f}  static={es:.4f}  "
                      f"imp_vs_add={imp:+.1f}%")

        print(f"{'='*70}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate WarpResidualFlowModulator')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/flow_modulator_eval')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--num_visualize', type=int, default=10)
    args = parser.parse_args()

    evaluator = FlowModulatorEvaluator(args)
    evaluator.evaluate()


if __name__ == '__main__':
    main()