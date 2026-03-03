import os
import re
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
import tqdm
import torch
from einops import rearrange
from torch.nn.functional import interpolate
# ============================================================
# ====================== 全局参数区 ===========================
# ============================================================

VIDEOS_ROOT = r"/data/gy/UAVDT/官方/test"          # 50 个视频文件夹
ANNOTATION_ROOT = r"/data/gy/UAVDT/官方/test_anno"     # 50 个标注 txt
OUTPUT_ROOT = r"/data/gy/UAVDT/CogVideo-Track-TinyFlow/test"                      # 最终输出目录

CLIP_LEN = 49
STRIDE = 2
START_STEP = CLIP_LEN * STRIDE   # 不重叠：98

FPS = 8

GRID_STEP = 16
RANSAC_THRESH = 3.0
MASK_DILATE = 8

IMG_EXTS = {".jpg", ".png", ".jpeg"}

# 是否对补偿后的 bbox 做裁剪到图像范围内（建议 True）
CLIP_COMP_BBOX_TO_IMAGE = False
from raft.core.raft import RAFT
from raft.core.utils import flow_viz
from raft.core.utils.utils import InputPadder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

import argparse

args = argparse.Namespace(
    small=False,          # 若想用 RAFT-Small 改成 True
    mixed_precision=False,
    alternate_corr=False,
    dropout=0.
)
RAPT_MODEL  = "raft-things.pth"   # 下载地址见下方
# 1. 加载模型（离线）
raft_model = RAFT(args)
raft_model.load_state_dict(
    {k.replace("module.", ""): v for k, v in torch.load(RAPT_MODEL, map_location="cpu").items()}
    )
raft_model = raft_model.to(DEVICE).eval()
# ============================================================
# ====================== 工具函数 =============================
# ============================================================

def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def list_frames(folder):
    files = []
    for f in os.listdir(folder):
        if os.path.splitext(f)[1].lower() in IMG_EXTS:
            files.append(os.path.join(folder, f))
    return sorted(files, key=lambda x: natural_key(os.path.basename(x)))

def affine2x3_to_3x3(A):
    M = np.eye(3, dtype=np.float64)
    M[:2] = A.astype(np.float64)
    return M

def safe_inv(M):
    try:
        return np.linalg.inv(M)
    except:
        return np.eye(3, dtype=np.float64)

def clip_bbox_to_image(l, t, w, h, W, H):
    # 将 bbox 裁剪到 [0,W-1]x[0,H-1]，并保证 w/h 非负
    l2 = max(0.0, min(float(l), W - 1.0))
    t2 = max(0.0, min(float(t), H - 1.0))
    r2 = max(0.0, min(float(l + w), W - 1.0))
    b2 = max(0.0, min(float(t + h), H - 1.0))
    w2 = max(0.0, r2 - l2)
    h2 = max(0.0, b2 - t2)
    return l2, t2, w2, h2

def find_prompt_text(video_dir: str) -> str:
    """
    UAVDT 每个视频文件夹里常有一个文本描述 txt（不一定叫 prompt.txt）。
    这里做一个稳健策略：
    - 优先找 prompt.txt
    - 否则找该目录下“第一个 .txt 文件”（按自然排序），读整文件并 strip
    - 找不到返回空串
    """
    p1 = os.path.join(video_dir, "prompt.txt")
    if os.path.exists(p1):
        try:
            with open(p1, "r", encoding="utf-8") as f:
                return f.read().strip()
        except:
            pass

    txts = [f for f in os.listdir(video_dir) if f.lower().endswith(".txt")]
    txts.sort(key=natural_key)
    for fn in txts:
        p = os.path.join(video_dir, fn)
        try:
            with open(p, "r", encoding="utf-8") as f:
                return f.read().strip()
        except:
            continue
    return ""


# ============================================================
# ====================== 标注结构 =============================
# ============================================================

@dataclass
class AnnRow:
    frame: int
    tid: int
    l: float
    t: float
    w: float
    h: float
    score: float
    in_view: int
    occ: int

def read_annotations(path) -> Dict[int, List[AnnRow]]:
    data: Dict[int, List[AnnRow]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            if len(p) < 9:
                continue
            r = AnnRow(
                frame=int(p[0]),
                tid=int(p[1]),
                l=float(p[2]),
                t=float(p[3]),
                w=float(p[4]),
                h=float(p[5]),
                score=float(p[6]),
                in_view=int(float(p[7])),
                occ=int(float(p[8])),
            )
            data.setdefault(r.frame, []).append(r)

    return data

def write_ann(path, rows: List[AnnRow]):
    # 排序：方便你检查每帧多个框是否都在
    rows = sorted(rows, key=lambda x: (x.frame, x.tid))
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(
                f"{r.frame},{r.tid},{r.l:.2f},{r.t:.2f},{r.w:.2f},{r.h:.2f},"
                f"{r.score},{r.in_view},{r.occ}\n"
            )


# ============================================================
# ====================== 光流 & 相机运动 ======================
# ============================================================

def compute_flow(img0_gray, img1_gray):
    # 转 RGB 三通道
    img1 = cv2.cvtColor(img0_gray, cv2.COLOR_GRAY2RGB)
    img2 = cv2.cvtColor(img1_gray, cv2.COLOR_GRAY2RGB)
    img1 = torch.from_numpy(img1).permute(2,0,1).float()[None].to(DEVICE)
    img2 = torch.from_numpy(img2).permute(2,0,1).float()[None].to(DEVICE)

    padder = InputPadder(img1.shape)
    img1, img2 = padder.pad(img1, img2)

    with torch.no_grad():
        flow_low, flow_up = raft_model(img1, img2, iters=20, test_mode=True)
    flow = flow_up[0].permute(1,2,0).cpu().numpy()  # (H,W,2)
    # 去除padding，确保光流尺寸与原始图像一致
    flow = padder.unpad(flow)
    return flow
    # dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    # # dis.setUseSpatialPropagation(True)
    # dis.setGradientDescentIterations(128)   # 默认 12
    # dis.setFinestScale(0)                   # 使用原图分辨率
    # dis.setPatchStride(1) 
    # return dis.calc(img0_gray, img1_gray, None).astype(np.float32)


def build_mask(H, W, anns0: Optional[List[AnnRow]], anns1: Optional[List[AnnRow]]):
    """
    255 表示需要排除（前景目标区域）
    """
    mask = np.zeros((H, W), np.uint8)
    for anns in [anns0, anns1]:
        if anns is None:
            continue
        for a in anns:
            x1 = int(max(0, min(W - 1, a.l)))
            y1 = int(max(0, min(H - 1, a.t)))
            x2 = int(max(0, min(W - 1, a.l + a.w)))
            y2 = int(max(0, min(H - 1, a.t + a.h)))
            if x2 > x1 and y2 > y1:
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    if MASK_DILATE > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MASK_DILATE * 2 + 1, MASK_DILATE * 2 + 1))
        mask = cv2.dilate(mask, k, iterations=1)
    return mask

def estimate_homography(flow, mask):
    """
    用 flow 的真实尺寸决定采样网格，避免任何 H/W 不一致。
    使用透视变换（homography）替代仿射变换，以处理镜头旋转带来的透视效果。
    """
    Hf, Wf = flow.shape[:2]

    if mask is not None and mask.shape[:2] != (Hf, Wf):
        mask = cv2.resize(mask, (Wf, Hf), interpolation=cv2.INTER_NEAREST)

    ys, xs = np.mgrid[GRID_STEP // 2:Hf:GRID_STEP, GRID_STEP // 2:Wf:GRID_STEP]
    pts = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32)

    # 双保险 clip
    pts[:, 0] = np.clip(pts[:, 0], 0, Wf - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, Hf - 1)

    if mask is not None:
        keep = mask[pts[:, 1].astype(np.int32), pts[:, 0].astype(np.int32)] == 0
        pts = pts[keep]

    if pts.shape[0] < 4:
        return np.eye(3, dtype=np.float64)

    dst = pts + flow[pts[:, 1].astype(np.int32), pts[:, 0].astype(np.int32)]

    # 使用 findHomography 替代 estimateAffinePartial2D
    H, _ = cv2.findHomography(
        pts, dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESH,
        maxIters=2000,
        confidence=0.99
    )
    if H is None:
        return np.eye(3, dtype=np.float64)

    return H


# ============================================================
# ====================== bbox 还原 ============================
# ============================================================

def warp_bbox(bbox, invM):
    l, t, w, h = bbox
    pts = np.array([
        [l,     t,     1.0],
        [l + w, t,     1.0],
        [l,     t + h, 1.0],
        [l + w, t + h, 1.0]
    ], dtype=np.float64).T  # 3x4

    pts2 = invM @ pts
    pts2 = pts2[:2] / np.maximum(pts2[2:], 1e-9)
    x1, y1 = pts2.min(1)
    x2, y2 = pts2.max(1)
    return float(x1), float(y1), float(x2 - x1), float(y2 - y1)


# ============================================================
# ====================== 视频保存 =============================
# ============================================================

def write_mp4(frames, out, size_wh):
    """
    Resize frames to target size before writing
    """
    W, H = size_wh
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in frames:
        img = cv2.imread(f)
        if img is None:
            continue
        # Resize to target size (480x720)
        img = cv2.resize(img, (W, H))
        vw.write(img)
    vw.release()


# ============================================================
# ====================== 主流程 ===============================
# ============================================================

def main():
    ensure_dir(OUTPUT_ROOT)
    for d in ["videos", "images", "flows", "trajectories", "trajectories_with_image", "trajectories_gt", "trajectory_videos", "trajectory_rgb", "trajectory_arrays", "gt_trajectory_videos", "object_flows", "object_flows_gt", "object_flows_videos", "object_flows_gt_videos"]:
        ensure_dir(os.path.join(OUTPUT_ROOT, d))

    f_v = open(os.path.join(OUTPUT_ROOT, "videos.txt"), "w", encoding="utf-8")
    f_i = open(os.path.join(OUTPUT_ROOT, "images.txt"), "w", encoding="utf-8")
    f_p = open(os.path.join(OUTPUT_ROOT, "prompts.txt"), "w", encoding="utf-8")
    f_f = open(os.path.join(OUTPUT_ROOT, "flows.txt"), "w", encoding="utf-8")
    f_t = open(os.path.join(OUTPUT_ROOT, "trajectories.txt"), "w", encoding="utf-8")
    f_gt = open(os.path.join(OUTPUT_ROOT, "trajectories_gt.txt"), "w", encoding="utf-8")

    sample_id = 0

    video_dirs = sorted([d for d in os.listdir(VIDEOS_ROOT) if os.path.isdir(os.path.join(VIDEOS_ROOT, d))],
                        key=natural_key)
    ann_files = sorted([f for f in os.listdir(ANNOTATION_ROOT) if f.lower().endswith(".txt")],
                       key=natural_key)
    # 全局统计变量
    global_stats = {
        'static_flow_mags': [],
        'camera_flow_mags': [],
        'gt_flow_mags': [],
        'camera_motion_mags': []
    }
    for vdir, annf in tqdm.tqdm(list(zip(video_dirs, ann_files)),
                                desc="Processing videos",
                                total=min(len(video_dirs), len(ann_files)),
                                ncols=100):
        vpath = os.path.join(VIDEOS_ROOT, vdir)
        frames = list_frames(vpath)
        if len(frames) == 0:
            continue

        ann = read_annotations(os.path.join(ANNOTATION_ROOT, annf))

        # prompt：尽量从视频目录下的 txt 找
        prompt = find_prompt_text(vpath)


        max_start = len(frames) - STRIDE * (CLIP_LEN - 1)
        if max_start <= 0:
            continue

        for s in range(0, max_start, START_STEP):
            idxs = [s + STRIDE * i for i in range(CLIP_LEN)]
            if idxs[-1] >= len(frames):
                break

            clip_frames = [frames[i] for i in idxs]
            frame_ids = [i + 1 for i in idxs]  # 标注是 1-based frame_index

            # 以固定尺寸480*720为 clip 坐标系
            img0 = cv2.imread(clip_frames[0])
            if img0 is None:
                continue
            # 获取原始分辨率
            orig_H, orig_W = img0.shape[:2]
            # 计算缩放比例
            scale_x = 720 / orig_W
            scale_y = 480 / orig_H
            # Resize to fixed size (480x720)
            img0 = cv2.resize(img0, (720, 480))
            Hc, Wc = 480, 720

            # 累计仿射：cum[t] = M_{0->t}
            cum = [np.eye(3, dtype=np.float64)]
            gray_prev = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

            flows = []
            for i in range(CLIP_LEN - 1):
                img = cv2.imread(clip_frames[i + 1])
                # Resize to fixed size (480x720)
                img = cv2.resize(img, (720, 480))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                flow = compute_flow(gray_prev, gray)
                if flow.shape[:2] != (Hc, Wc):
                    scale_y_flow = Hc / flow.shape[0]
                    scale_x_flow = Wc / flow.shape[1]
                    flow = cv2.resize(flow, (Wc, Hc), interpolation=cv2.INTER_LINEAR)
                    flow[:, :, 0] *= scale_x_flow
                    flow[:, :, 1] *= scale_y_flow
                flows.append(flow)
                
                # 为当前帧和下一帧创建缩放后的标注
                def scale_anns(anns):
                    if anns is None:
                        return None
                    scaled_anns = []
                    for a in anns:
                        scaled_a = AnnRow(
                            frame=a.frame,
                            tid=a.tid,
                            l=a.l * scale_x,
                            t=a.t * scale_y,
                            w=a.w * scale_x,
                            h=a.h * scale_y,
                            score=a.score,
                            in_view=a.in_view,
                            occ=a.occ
                        )
                        scaled_anns.append(scaled_a)
                    return scaled_anns
                
                mask = build_mask(
                    Hc, Wc,
                    scale_anns(ann.get(frame_ids[i])),
                    scale_anns(ann.get(frame_ids[i + 1]))
                )

                M = estimate_homography(flow, mask)
                cum.append(M @ cum[-1])
                gray_prev = gray

            cum = np.stack(cum, axis=0)  # (49,3,3)
            flows = np.array(flows)   # (48,H,W,2)
            # === 关键：每帧所有 bbox 都做补偿 ===
            comp: List[AnnRow] = []
            for t, fid in enumerate(frame_ids):
                invM = safe_inv(cum[t])
                rows = ann.get(fid, [])
                for r in rows:
                    # 先将标注坐标缩放到480*720分辨率
                    scaled_l = r.l * scale_x
                    scaled_t = r.t * scale_y
                    scaled_w = r.w * scale_x
                    scaled_h = r.h * scale_y
                    
                    # 首帧(t=0)直接使用原始坐标，不进行变换，确保与gt一致
                    if t == 0:
                        l, t2, w, h = scaled_l, scaled_t, scaled_w, scaled_h
                    else:
                        # 然后进行轨迹补偿
                        l, t2, w, h = warp_bbox((scaled_l, scaled_t, scaled_w, scaled_h), invM)

                    if CLIP_COMP_BBOX_TO_IMAGE:
                        l, t2, w, h = clip_bbox_to_image(l, t2, w, h, Wc, Hc)

                    comp.append(AnnRow(
                        frame=t + 1,  # clip 内 1..49
                        tid=r.tid,
                        l=l, t=t2, w=w, h=h,
                        score=r.score, in_view=r.in_view, occ=r.occ
                    ))

            sid = f"{sample_id:06d}"

            # 保存文件
            video_rel = f"videos/clip_{sid}.mp4"
            image_rel = f"images/img_{sid}.jpg"
            flow_rel = f"flows/flow_{sid}.npz"
            traj_rel = f"trajectories/ann_{sid}.txt"

            write_mp4(clip_frames, os.path.join(OUTPUT_ROOT, video_rel), size_wh=(Wc, Hc))
            cv2.imwrite(os.path.join(OUTPUT_ROOT, image_rel), img0)
            
            # 可视化轨迹：在首帧图像上绘制所有目标的轨迹
            # 1. 按目标ID分组轨迹点
            trajectories = {}
            for r in comp:
                if r.tid not in trajectories:
                    trajectories[r.tid] = []
                # 计算中心点坐标
                center_x = r.l + r.w / 2
                center_y = r.t + r.h / 2
                trajectories[r.tid].append((center_x, center_y))
            
            # 2. 读取首帧图像并绘制轨迹
            img = cv2.imread(os.path.join(OUTPUT_ROOT, image_rel))
            
            # 3. 为每个目标绘制轨迹
            for tid, points in trajectories.items():
                if len(points) < 2:
                    continue  # 至少需要两个点才能绘制轨迹
                # 绘制轨迹线
                for i in range(len(points) - 1):
                    cv2.line(img, tuple(map(int, points[i])), tuple(map(int, points[i+1])), (0, 0, 255), 2)
                # 绘制轨迹点
                # for point in points:
                #     cv2.circle(img, tuple(map(int, point)), 3, (0, 0, 255), -1)
                # 在轨迹起点标注目标ID
                cv2.putText(img, f"ID: {tid}", tuple(map(int, points[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # 4. 保存带有轨迹的图像到trajectories_with_image文件夹
            trajectory_img_rel = f"trajectories_with_image/img_{sid}_trajectory.jpg"
            trajectory_img_path = os.path.join(OUTPUT_ROOT, trajectory_img_rel)
            cv2.imwrite(trajectory_img_path, img)
            
            # 5. 生成不加首帧图的轨迹图，只包含轨迹线和点
            # 创建空白图像，大小与首帧相同
            blank_img = np.zeros_like(img)
            # 为每个目标绘制轨迹
            for tid, points in trajectories.items():
                if len(points) < 2:
                    continue  # 至少需要两个点才能绘制轨迹
                # 绘制轨迹线
                for i in range(len(points) - 1):
                    cv2.line(blank_img, tuple(map(int, points[i])), tuple(map(int, points[i+1])), (0, 0, 255), 2)        
            # 6. 保存不加首帧图的轨迹图到trajectories文件夹
            trajectory_only_rel = f"trajectories/traj_{sid}.jpg"
            trajectory_only_path = os.path.join(OUTPUT_ROOT, trajectory_only_rel)
            cv2.imwrite(trajectory_only_path, blank_img)
            
            # 7. 绘制真实轨迹（原始未补偿的轨迹）
            # 按目标ID分组原始轨迹点
            gt_trajectories = {}
            for i, fid in enumerate(frame_ids):
                for r in ann.get(fid, []):
                    if r.tid not in gt_trajectories:
                        gt_trajectories[r.tid] = []
                    # 先将标注坐标缩放到480*720分辨率
                    scaled_l = r.l * scale_x
                    scaled_t = r.t * scale_y
                    scaled_w = r.w * scale_x
                    scaled_h = r.h * scale_y
                    # 计算缩放后的中心点坐标
                    center_x = scaled_l + scaled_w / 2
                    center_y = scaled_t + scaled_h / 2
                    gt_trajectories[r.tid].append((center_x, center_y))
            
            # 创建空白图像用于绘制真实轨迹
            gt_blank_img = np.zeros_like(img)
            # 为每个目标绘制真实轨迹
            for tid, points in gt_trajectories.items():
                if len(points) < 2:
                    continue  # 至少需要两个点才能绘制轨迹
                # 绘制轨迹线（使用蓝色区分）
                for i in range(len(points) - 1):
                    cv2.line(gt_blank_img, tuple(map(int, points[i])), tuple(map(int, points[i+1])), (0, 0, 255), 2)   
            # 8. 保存真实轨迹图到trajectories_gt文件夹
            gt_trajectory_rel = f"trajectories_gt/traj_gt_{sid}.jpg"
            gt_trajectory_path = os.path.join(OUTPUT_ROOT, gt_trajectory_rel)
            cv2.imwrite(gt_trajectory_path, gt_blank_img)
            
            flows = torch.from_numpy(flows).float()
            # T H W 2->T 2 H W
            flows = flows.permute(0, 3, 1, 2).contiguous()
            flows = rearrange(flows,"(t k) c h w -> t k c h w",k = 4)
            flows_lat = flows.mean(dim=1)
            zero = torch.zeros_like(flows_lat[:1])
            flows_lat = torch.cat([zero, flows_lat], dim=0)

            flows_lat_small = interpolate(
                flows_lat, 
                size=(Hc//8, Wc//8),  # (60, 90)
                mode='bilinear', 
                align_corners=False
            )  # [13, 2, 60, 90]
            flow_scale_x = (Wc // 8) / Wc   # = 1/8
            flow_scale_y = (Hc // 8) / Hc   # = 1/8
            
            np.savez_compressed(
            os.path.join(OUTPUT_ROOT, flow_rel),
            cum=cum,
            flows=flows_lat_small,
            H=np.int32(Hc // 8),
            W=np.int32(Wc // 8),
            )
            #flows_compressed = np.array([cv2.resize(flow, (Wc//8, Hc//8)) for flow in flows])
            #np.savez_compressed(os.path.join(OUTPUT_ROOT, flow_rel), cum=cum, flows=flows_compressed, H=np.int32(Hc//8), W=np.int32(Wc//8))
            write_ann(os.path.join(OUTPUT_ROOT, traj_rel), comp)
            
            # 保存补偿后的轨迹视频
            trajectory_video_rel = f"trajectory_videos/traj_video_{sid}.mp4"
            trajectory_video_path = os.path.join(OUTPUT_ROOT, trajectory_video_rel)

            # 新增: 保存原始热力图数组
            trajectory_array_rel = f"trajectory_arrays/traj_{sid}.npz"
            trajectory_array_path = os.path.join(OUTPUT_ROOT, trajectory_array_rel)

            vw = cv2.VideoWriter(trajectory_video_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (Wc, Hc))
            heatmap_frames = []  # 收集所有帧

            for i, frame_path in enumerate(clip_frames):
                heatmap = np.zeros((Hc, Wc), dtype=np.float32)
                
                for tid, points in trajectories.items():
                    if len(points) > i:
                        x, y = map(int, points[i])
                        sigma = 15
                        y_grid, x_grid = np.ogrid[0:Hc, 0:Wc]
                        gaussian = np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2))
                        heatmap += gaussian
                
                # 归一化
                if heatmap.max() > 0:
                    heatmap_normalized = (heatmap / heatmap.max() * 255).astype(np.uint8)
                else:
                    heatmap_normalized = np.zeros((Hc, Wc), dtype=np.uint8)
                
                # 转为 RGB 三通道(用于训练)
                heatmap_rgb = np.stack([heatmap_normalized] * 3, axis=-1)  # [H, W, 3]
                heatmap_frames.append(heatmap_rgb)
                
                # 可视化用的彩色版本
                heatmap_color = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
                vw.write(heatmap_color)

            vw.release()

            # 保存为 numpy 数组(训练时加载这个)
            heatmap_video = np.stack(heatmap_frames, axis=0)  # [T, H, W, 3]
            np.savez_compressed(trajectory_array_path, trajectory=heatmap_video)

            # 更新索引文件
            f_traj_array = open(os.path.join(OUTPUT_ROOT, "trajectory_arrays.txt"), "a")
            f_traj_array.write(trajectory_array_rel + "\n")
            f_traj_array.close()
            
            # 保存真实视角轨迹视频（热力图）
            gt_trajectory_video_rel = f"gt_trajectory_videos/traj_gt_video_{sid}.mp4"
            gt_trajectory_video_path = os.path.join(OUTPUT_ROOT, gt_trajectory_video_rel)
            
            # 创建视频写入器
            gt_vw = cv2.VideoWriter(gt_trajectory_video_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (Wc, Hc))
            
            # 为每一帧绘制真实视角轨迹热力图
            for i, frame_path in enumerate(clip_frames):
                # 创建热力图
                gt_frame_heatmap = np.zeros((Hc, Wc), dtype=np.float32)
                
                # 为每个物体位置添加高斯分布
                for tid, points in gt_trajectories.items():
                    if len(points) > i:
                        current_point = points[i]
                        x, y = map(int, current_point)
                        
                        # 添加高斯分布
                        sigma = 15  # 高斯分布的标准差
                        y_grid, x_grid = np.ogrid[0:Hc, 0:Wc]
                        gaussian = np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2))
                        gt_frame_heatmap += gaussian
                
                # 归一化热力图到0-255
                if gt_frame_heatmap.max() > 0:
                    gt_frame_heatmap_normalized = (gt_frame_heatmap / gt_frame_heatmap.max() * 255).astype(np.uint8)
                else:
                    gt_frame_heatmap_normalized = np.zeros((Hc, Wc), dtype=np.uint8)
                
                # 将热力图转换为彩色图像（蓝色到红色）
                gt_frame_heatmap_color = cv2.applyColorMap(gt_frame_heatmap_normalized, cv2.COLORMAP_JET)
                
                # 在第一帧标注目标ID
                if i == 0:
                    for tid, points in gt_trajectories.items():
                        if len(points) > i:
                            current_point = points[i]
                            x, y = map(int, current_point)
                            cv2.putText(gt_frame_heatmap_color, f"ID: {tid}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                                        
                gt_vw.write(gt_frame_heatmap_color)
            
            gt_vw.release()
            
            # 提取并保存物体运动光流
            # 1. 首帧视角下的物体运动光流（补偿后的）
            object_flows_dict = {}
            # 2. 正常视角下的物体运动光流（原始的）
            object_flows_gt_dict = {}
            
            # ── 第一步：收集所有 tid（必须最先执行）
            all_tids = set()
            for fid in frame_ids:
                for r in ann.get(fid, []):
                    all_tids.add(r.tid)
            # ── 第二步：构建 comp_by_tid（None 填充，帧索引对齐）
            comp_by_tid = {}
            for tid in all_tids:
                comp_by_tid[tid] = [None] * len(frame_ids)  # 长度=49，None填充

            for r in comp:
                frame_idx = r.frame - 1   # r.frame 是 clip内 1-based，转为 0-based
                if r.tid in comp_by_tid:
                    comp_by_tid[r.tid][frame_idx] = r
            
            # 按目标ID分组原始边界框
            gt_by_tid = {}
            for tid in all_tids:
                gt_by_tid[tid] = [None] * len(frame_ids)
            
            for i, fid in enumerate(frame_ids):
                for r in ann.get(fid, []):
                    # 缩放到480*720分辨率
                    scaled_l = r.l * scale_x
                    scaled_t = r.t * scale_y
                    scaled_w = r.w * scale_x
                    scaled_h = r.h * scale_y
                    gt_by_tid[r.tid][i] = (scaled_l, scaled_t, scaled_w, scaled_h)
            
            # 提取物体光流
            # 计算缩放因子，将480×720缩放到60×90
            scale_x_small = (Wc // 8) / Wc  # 90/720 = 1/8
            scale_y_small = (Hc // 8) / Hc  # 60/480 = 1/8
            
            for tid in comp_by_tid:
                if tid not in gt_by_tid:
                    continue
                # ← 在这里初始化（外层循环，每个 tid 只执行一次）
                object_flows_dict[tid] = [None] * (len(frame_ids) - 1)     # 长度48
                object_flows_gt_dict[tid] = [None] * (len(frame_ids) - 1)  # 长度48

                comp_bboxes = comp_by_tid[tid]
                gt_bboxes = gt_by_tid[tid]
                
                if len(comp_bboxes) != len(gt_bboxes):
                    continue
                
                # # 首帧视角下的物体光流
                # object_flows = []
                # # 正常视角下的物体光流
                # object_flows_gt = []

                # 计算光流差异（物体运动 - 全局运动）
                for i in range(len(comp_bboxes) - 1):
                    # 同时检查 comp 和 gt 两侧
                    if comp_bboxes[i] is None or comp_bboxes[i + 1] is None:
                        continue
                    if gt_bboxes[i] is None or gt_bboxes[i + 1] is None:
                        continue
                    
                    # ── 静态视角：直接用补偿后 bbox 中心坐标差分 ────────
                    # comp_bboxes 已经通过 warp_bbox 消除了相机运动，
                    # 所以相邻帧的坐标差就是纯物体运动。
                    cb_curr = comp_bboxes[i]
                    cb_next = comp_bboxes[i + 1]

                    # ── 动态视角：GT 解包（需要在 debug print 之前）────────
                    gl_curr, gt_curr, gw_curr, gh_curr = gt_bboxes[i]
                    gl_next, gt_next, gw_next, gh_next = gt_bboxes[i + 1]

                    gx_curr = gl_curr + gw_curr / 2
                    gy_curr = gt_curr + gh_curr / 2
                    gx_next = gl_next + gw_next / 2
                    gy_next = gt_next + gh_next / 2

                    if i == 0 and tid == list(gt_by_tid.keys())[0]:
                        print(f"\n=== 第一帧 ===")
                        print(f"comp_bbox: l={cb_curr.l:.2f}, t={cb_curr.t:.2f}")
                        print(f"gt_bbox:   l={gl_curr:.2f}, t={gt_curr:.2f}")
                        print(f"差异: Δl={abs(cb_curr.l - gl_curr):.2f}, Δt={abs(cb_curr.t - gt_curr):.2f}")
                    
                    dx_static = (cb_next.l + cb_next.w / 2) - (cb_curr.l + cb_curr.w / 2)
                    dy_static = (cb_next.t + cb_next.h / 2) - (cb_curr.t + cb_curr.h / 2)
                    
                    # 缩放到 60×90 分辨率
                    static_flow = np.array([dx_static / 8.0,
                        dy_static / 8.0], dtype=np.float32)
                    object_flows_dict[tid][i] = static_flow

                    # ── 动态视角：GT 就是 UAVDT 原始标注差分（动态视角下表观位移）
                    dx_gt = gx_next - gx_curr
                    dy_gt = gy_next - gy_curr
                    
                    # 缩放到 60×90 分辨率
                    gt_flow = np.array([dx_gt / 8.0, dy_gt / 8.0], dtype=np.float32)
                    object_flows_gt_dict[tid][i] = gt_flow
            
            Hs, Ws = Hc // 8, Wc // 8  # 60, 90

            obj_flow_dense    = np.zeros((len(frame_ids) - 1, 2, Hs, Ws), dtype=np.float32)
            obj_flow_gt_dense = np.zeros((len(frame_ids) - 1, 2, Hs, Ws), dtype=np.float32)

            for tid in comp_by_tid:
                if tid not in gt_by_tid:
                    continue
                comp_bboxes = comp_by_tid[tid]
                gt_bboxes   = gt_by_tid[tid]

                for i in range(len(frame_ids) - 1):
                    # ── 静态视角 ──
                    if (object_flows_dict.get(tid) is not None
                            and object_flows_dict[tid][i] is not None
                            and comp_bboxes[i] is not None):
                        dx, dy = object_flows_dict[tid][i]          # 60×90 分辨率
                        cb = comp_bboxes[i]
                        # 将 bbox 坐标映射到 60×90
                        x1 = int(np.clip(cb.l * flow_scale_x, 0, Ws - 1))
                        y1 = int(np.clip(cb.t * flow_scale_y, 0, Hs - 1))
                        x2 = int(np.clip((cb.l + cb.w) * flow_scale_x, 0, Ws - 1)) + 1
                        y2 = int(np.clip((cb.t + cb.h) * flow_scale_y, 0, Hs - 1)) + 1
                        if x2 > x1 and y2 > y1:
                            obj_flow_dense[i, 0, y1:y2, x1:x2] = dx
                            obj_flow_dense[i, 1, y1:y2, x1:x2] = dy

                    # ── 动态视角 ──
                    if (object_flows_gt_dict.get(tid) is not None
                            and object_flows_gt_dict[tid][i] is not None
                            and gt_bboxes[i] is not None):
                        dx, dy = object_flows_gt_dict[tid][i]
                        gl, gt2, gw, gh = gt_bboxes[i]
                        x1 = int(np.clip(gl * flow_scale_x, 0, Ws - 1))
                        y1 = int(np.clip(gt2 * flow_scale_y, 0, Hs - 1))
                        x2 = int(np.clip((gl + gw) * flow_scale_x, 0, Ws - 1)) + 1
                        y2 = int(np.clip((gt2 + gh) * flow_scale_y, 0, Hs - 1)) + 1
                        if x2 > x1 and y2 > y1:
                            obj_flow_gt_dense[i, 0, y1:y2, x1:x2] = dx
                            obj_flow_gt_dense[i, 1, y1:y2, x1:x2] = dy

            # ── 分组（同全局光流：48 → 12组，每组4帧取均值）+ 首帧zero ──────
            obj_flow_tensor    = torch.from_numpy(obj_flow_dense)    # (48, 2, 60, 90)
            obj_flow_gt_tensor = torch.from_numpy(obj_flow_gt_dense) # (48, 2, 60, 90)

            obj_flow_lat    = rearrange(obj_flow_tensor,    "(t k) c h w -> t k c h w", k=4).mean(dim=1)  # (12, 2, 60, 90)
            obj_flow_gt_lat = rearrange(obj_flow_gt_tensor, "(t k) c h w -> t k c h w", k=4).mean(dim=1)

            zero = torch.zeros_like(obj_flow_lat[:1])
            obj_flow_lat    = torch.cat([zero, obj_flow_lat],    dim=0)  # (13, 2, 60, 90)
            obj_flow_gt_lat = torch.cat([zero, obj_flow_gt_lat], dim=0)  # (13, 2, 60, 90)

            # ── 保存 ────────────────────────────────────────────────────────
            object_flows_rel    = f"object_flows/obj_flow_{sid}.npz"
            object_flows_gt_rel = f"object_flows_gt/obj_flow_gt_{sid}.npz"

            np.savez_compressed(
                os.path.join(OUTPUT_ROOT, object_flows_rel),
                flows=obj_flow_lat.numpy(),    # (13, 2, 60, 90)
                H=np.int32(Hs),
                W=np.int32(Ws),
            )
            np.savez_compressed(
                os.path.join(OUTPUT_ROOT, object_flows_gt_rel),
                flows=obj_flow_gt_lat.numpy(), # (13, 2, 60, 90)
                H=np.int32(Hs),
                W=np.int32(Ws),
            )
            # 计算当前样本的统计数据
            camera_flow_mag = np.sqrt(flows_lat_small[:, 0]**2 + flows_lat_small[:, 1]**2).mean().item()
            global_stats['camera_flow_mags'].append(camera_flow_mag)
            
            # ✅ 只统计第一帧（i=0）的光流，与DEBUG输出保持一致
            # 稠密光流: (48, 2, 60, 90)，取第一帧: (2, 60, 90)
            static_frame0 = obj_flow_dense[0]  # (2, 60, 90)
            gt_frame0 = obj_flow_gt_dense[0]   # (2, 60, 90)
            
            static_mag_frame0 = np.sqrt(static_frame0[0]**2 + static_frame0[1]**2)  # (60, 90)
            gt_mag_frame0 = np.sqrt(gt_frame0[0]**2 + gt_frame0[1]**2)              # (60, 90)
            
            # 只统计非零区域（物体所在位置）
            static_mask = static_mag_frame0 > 1e-6
            gt_mask = gt_mag_frame0 > 1e-6
            
            static_mag = static_mag_frame0[static_mask].mean() if static_mask.any() else 0.0
            gt_mag = gt_mag_frame0[gt_mask].mean() if gt_mask.any() else 0.0
            
            # 实时打印当前样本的统计信息
            print(f"[样本 {sid}] 静态物体运动: {static_mag:.3f} px | 相机运动: {camera_flow_mag:.3f} px | 真实物体运动: {gt_mag:.3f} px")
            print(f"  (第一帧非零像素: static={static_mask.sum()}, gt={gt_mask.sum()}, 总像素={static_mag_frame0.size})")
            
            # 保存全局相机运动光流（60×90分辨率）
            global_flow_rel = f"global_flows/global_flow_{sid}.npz"
            global_flow_path = os.path.join(OUTPUT_ROOT, global_flow_rel)
            os.makedirs(os.path.dirname(global_flow_path), exist_ok=True)
            np.savez_compressed(
                global_flow_path,
                flows=flows_lat_small,
                H=np.int32(Hc // 8),
                W=np.int32(Wc // 8),
            )
            
            # 保存物体运动光流可视化视频
            object_flows_video_rel      = f"object_flows_videos/obj_flow_{sid}.mp4"
            object_flows_gt_video_rel   = f"object_flows_gt_videos/obj_flow_gt_{sid}.mp4"
            object_flows_video_path     = os.path.join(OUTPUT_ROOT, object_flows_video_rel)
            object_flows_gt_video_path  = os.path.join(OUTPUT_ROOT, object_flows_gt_video_rel)

            # 在 480×720 上叠加可视化
            vw_obj    = cv2.VideoWriter(object_flows_video_path,    cv2.VideoWriter_fourcc(*"mp4v"), FPS, (Wc, Hc))
            vw_obj_gt = cv2.VideoWriter(object_flows_gt_video_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (Wc, Hc))

            COLOR_TABLE = [
                (0, 255, 0), (255, 128, 0), (0, 128, 255),
                (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 255, 0),
            ]
            tid_list  = sorted(comp_by_tid.keys())
            tid_color = {tid: COLOR_TABLE[k % len(COLOR_TABLE)] for k, tid in enumerate(tid_list)}

            first_frame = cv2.imread(clip_frames[0])
            first_frame = cv2.resize(first_frame, (Wc, Hc))
            
            # 将保存的稠密光流转换回numpy用于可视化验证
            obj_flow_vis = obj_flow_dense.copy()  # (48, 2, 60, 90)
            obj_flow_gt_vis = obj_flow_gt_dense.copy()

            for i, frame_path in enumerate(clip_frames):
                # ── 静态视角：首帧作为固定背景 ───────────────────────────
                frame_comp   = first_frame.copy()
                overlay_comp = frame_comp.copy()

                # ── 动态视角：当前帧作为背景 ─────────────────────────────
                frame_gt   = cv2.imread(frame_path)
                frame_gt   = cv2.resize(frame_gt, (Wc, Hc))
                overlay_gt = frame_gt.copy()

                for tid in comp_by_tid:
                    if tid not in gt_by_tid:
                        continue
                    comp_bboxes = comp_by_tid[tid]
                    gt_bboxes   = gt_by_tid[tid]
                    color = tid_color.get(tid, (0, 255, 0))

                    # ── 静态视角：补偿后bbox + 静态纯物体运动箭头 ────────
                    if comp_bboxes[i] is not None:
                        cb = comp_bboxes[i]
                        cx = int(cb.l + cb.w / 2)
                        cy = int(cb.t + cb.h / 2)
                        cv2.rectangle(overlay_comp,
                                    (int(cb.l), int(cb.t)),
                                    (int(cb.l + cb.w), int(cb.t + cb.h)),
                                    color, 1)
                        cv2.circle(overlay_comp, (cx, cy), 3, color, -1)
                        # 静态纯物体运动箭头（直接从保存的稠密光流读取，确保一致性）
                        if i < len(comp_bboxes) - 1 and comp_bboxes[i + 1] is not None:
                            # 从稠密光流中读取当前bbox中心位置的光流值
                            cx_small = int(cx * flow_scale_x)
                            cy_small = int(cy * flow_scale_y)
                            cx_small = np.clip(cx_small, 0, Ws - 1)
                            cy_small = np.clip(cy_small, 0, Hs - 1)
                            dx = obj_flow_vis[i, 0, cy_small, cx_small]  # 60×90分辨率
                            dy = obj_flow_vis[i, 1, cy_small, cx_small]
                            ex = int(cx + dx * 8.0 * 3)  # 还原到480×720再*3放大
                            ey = int(cy + dy * 8.0 * 3)
                            ex = max(0, min(ex, Wc - 1))
                            ey = max(0, min(ey, Hc - 1))
                            cv2.arrowedLine(overlay_comp, (cx, cy), (ex, ey), color, 2, tipLength=0.3)
                        cv2.putText(overlay_comp, f"ID:{tid}",
                                    (int(cb.l), int(cb.t) - 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                    # ── 动态视角：GT bbox + 动态纯物体运动箭头 ───────────
                    if gt_bboxes[i] is not None:
                        gl, gt2, gw, gh = gt_bboxes[i]
                        gx = int(gl + gw / 2)
                        gy = int(gt2 + gh / 2)
                        cv2.rectangle(overlay_gt,
                                    (int(gl), int(gt2)),
                                    (int(gl + gw), int(gt2 + gh)),
                                    color, 2)
                        cv2.circle(overlay_gt, (gx, gy), 4, color, -1)
                        # 动态纯物体运动箭头（直接从保存的稠密光流读取，确保一致性）
                        if i < len(gt_bboxes) - 1 and gt_bboxes[i + 1] is not None:
                            # 从稠密光流中读取当前bbox中心位置的光流值
                            gx_small = int(gx * flow_scale_x)
                            gy_small = int(gy * flow_scale_y)
                            gx_small = np.clip(gx_small, 0, Ws - 1)
                            gy_small = np.clip(gy_small, 0, Hs - 1)
                            dx = obj_flow_gt_vis[i, 0, gy_small, gx_small]  # 60×90分辨率
                            dy = obj_flow_gt_vis[i, 1, gy_small, gx_small]
                            ex = int(gx + dx * 8.0 * 3)  # 还原到480×720再*3放大
                            ey = int(gy + dy * 8.0 * 3)
                            ex = max(0, min(ex, Wc - 1))
                            ey = max(0, min(ey, Hc - 1))
                            cv2.arrowedLine(overlay_gt, (gx, gy), (ex, ey), color, 2, tipLength=0.3)
                        cv2.putText(overlay_gt, f"ID:{tid}",
                                    (int(gl), int(gt2) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # ── 半透明融合并写入 ──────────────────────────────────────
                vis_comp = cv2.addWeighted(overlay_comp, 0.7, frame_comp, 0.3, 0)
                vis_gt   = cv2.addWeighted(overlay_gt,   0.7, frame_gt,   0.3, 0)

                cv2.putText(vis_comp, f"Static View (Pure Object Motion) - Frame {i+1}", (5, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(vis_gt,   f"Dynamic View (GT Total - Camera) - Frame {i+1}", (5, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                vw_obj.write(vis_comp)
                vw_obj_gt.write(vis_gt)

            vw_obj.release()
            vw_obj_gt.release()

            
            # 生成RGB时间编码的轨迹图
            trajectory_rgb_rel = f"trajectory_rgb/traj_rgb_{sid}.jpg"
            trajectory_rgb_path = os.path.join(OUTPUT_ROOT, trajectory_rgb_rel)
            
            # 创建空白RGB图像
            rgb_img = np.zeros((Hc, Wc, 3), dtype=np.uint8)
            
            # 为每个目标生成时间编码的轨迹
            for tid, points in trajectories.items():
                if len(points) < 2:
                    continue
                
                # 为轨迹点生成时间编码颜色（蓝色到红色）
                for i, point in enumerate(points):
                    # 计算时间比例
                    t = i / (len(points) - 1)
                    # 蓝色到红色的渐变
                    blue = int(255 * (1 - t))
                    red = int(255 * t)
                    color = (blue, 0, red)  # BGR格式
                    
                    # 绘制轨迹点
                    x, y = map(int, point)
                    if 0 <= x < Wc and 0 <= y < Hc:
                        cv2.circle(rgb_img, (x, y), 3, color, -1)
                
                # 绘制轨迹线
                for i in range(len(points) - 1):
                    t1 = i / (len(points) - 1)
                    t2 = (i + 1) / (len(points) - 1)
                    color1 = (int(255 * (1 - t1)), 0, int(255 * t1))
                    color2 = (int(255 * (1 - t2)), 0, int(255 * t2))
                    
                    # 绘制渐变线条
                    x1, y1 = map(int, points[i])
                    x2, y2 = map(int, points[i+1])
                    if 0 <= x1 < Wc and 0 <= y1 < Hc and 0 <= x2 < Wc and 0 <= y2 < Hc:
                        cv2.line(rgb_img, (x1, y1), (x2, y2), color1, 2)
            
            cv2.imwrite(trajectory_rgb_path, rgb_img)

            # === 关键：索引文件严格一一对应（写相对路径更稳）===
            f_v.write(video_rel + "\n")
            f_i.write(image_rel + "\n")
            f_p.write((prompt or "") + "\n")
            f_f.write(flow_rel + "\n")
            f_t.write(trajectory_only_rel + "\n")
            f_gt.write(gt_trajectory_rel + "\n")

            sample_id += 1

    f_v.close()
    f_i.close()
    f_p.close()
    f_f.close()
    f_t.close()
    f_gt.close()
    print(f"Finished. Total samples: {sample_id}")
    print("\n" + "="*60)
    print("GLOBAL STATISTICS (480×720 resolution)")
    print("="*60)

    if len(global_stats['static_flow_mags']) > 0:
        print(f"Static Object Flow (pure object motion):")
        print(f"  Mean: {np.mean(global_stats['static_flow_mags']):.2f} px")
        print(f"  Std:  {np.std(global_stats['static_flow_mags']):.2f} px")
        print(f"  Min:  {np.min(global_stats['static_flow_mags']):.2f} px")
        print(f"  Max:  {np.max(global_stats['static_flow_mags']):.2f} px")

    if len(global_stats['camera_motion_mags']) > 0:
        print(f"\nCamera Motion (from homography):")
        print(f"  Mean: {np.mean(global_stats['camera_motion_mags']):.2f} px")
        print(f"  Std:  {np.std(global_stats['camera_motion_mags']):.2f} px")
        print(f"  Min:  {np.min(global_stats['camera_motion_mags']):.2f} px")
        print(f"  Max:  {np.max(global_stats['camera_motion_mags']):.2f} px")

    if len(global_stats['camera_flow_mags']) > 0:
        print(f"\nCamera Flow (global, 60×90 resolution):")
        print(f"  Mean: {np.mean(global_stats['camera_flow_mags']):.3f} px")
        print(f"  Std:  {np.std(global_stats['camera_flow_mags']):.3f} px")
        print(f"  Min:  {np.min(global_stats['camera_flow_mags']):.3f} px")
        print(f"  Max:  {np.max(global_stats['camera_flow_mags']):.3f} px")

    if len(global_stats['gt_flow_mags']) > 0:
        print(f"\nGT Object Flow (object + camera):")
        print(f"  Mean: {np.mean(global_stats['gt_flow_mags']):.2f} px")
        print(f"  Std:  {np.std(global_stats['gt_flow_mags']):.2f} px")
        print(f"  Min:  {np.min(global_stats['gt_flow_mags']):.2f} px")
        print(f"  Max:  {np.max(global_stats['gt_flow_mags']):.2f} px")

    print("\n" + "="*60)
    print(f"Total samples: {sample_id}")
    print("="*60)

if __name__ == "__main__":
    main()
