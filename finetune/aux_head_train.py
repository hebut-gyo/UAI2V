import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from torch.optim import AdamW
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finetune.modules.aux_head import AuxHead
from safetensors.torch import load_file
from tqdm import tqdm
class AuxHeadDataset(Dataset):
    def __init__(self, npz_root, latent_root):
        """
        npz_root: 你保存 flow 的 npz 文件目录
        latent_root: 你用 VAE encoder 预先保存的 latent [13,16,60,90]
        """
        self.npz_files = sorted(glob.glob(os.path.join(npz_root, "*.npz")))
        self.latent_root = latent_root
        self.latent_files = sorted([
            os.path.join(latent_root, f)
            for f in os.listdir(latent_root)
            if f.endswith('.safetensors')  # 或你的文件后缀
        ])
    def __len__(self):
        return len(self.npz_files)
    def __getitem__(self, idx):
        npz = np.load(self.npz_files[idx])
        flow_gt = npz["flows"].astype(np.float32)

        latent_path = self.latent_files[idx]
        latent_dict = load_file(latent_path)
        latent = latent_dict["encoded_video"]             # [13,16,60,90]
        return latent, torch.from_numpy(flow_gt)
@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_l1 = 0
    count = 0
    for latent, flow_gt in loader:
        latent = latent.to(device)
        flow_gt = flow_gt.to(device)
        flow_pred = model(latent.float())
        # 忽略 t=0
        flow_pred = flow_pred[:, 1:]
        flow_gt = flow_gt[:, 1:]
        l1 = (flow_pred - flow_gt).abs().mean()
        total_l1 += l1.item()
        count += 1
    return total_l1 / count

def train():
    device = "cuda:0"
    epochs = 200
    batch_size = 8
    lr = 1e-3
    save_path = "aux_head_best.pth"
    train_flow_folder = r"G:\UAVDT\CogVideo-Track-Tiny\train"
    train_latent_folder = r"G:\UAVDT\CogVideo-Track-Tiny\train"
    val_flow_folder = r"G:\UAVDT\CogVideo-Track-Tiny\test"
    val_latent_folder = r"G:\UAVDT\CogVideo-Track-Tiny\test"
    model = AuxHead().to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    train_set = AuxHeadDataset(train_flow_folder, train_latent_folder)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_set = AuxHeadDataset(val_flow_folder, val_latent_folder,)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    best_val = float("inf")

    for epoch in tqdm(range(epochs)):
        model.train()
        total_train_loss = 0
        for latent, flow_gt in train_loader:
            latent = latent.to(device)  # [B,13,16,60,90]
            flow_gt = flow_gt.to(device)  # [B,13, 2,60,90]
            flow_pred = model(latent.float())  # forward
            # L1 光流监督
            loss = (flow_pred[:, 1:] - flow_gt[:, 1:]).abs().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        train_loss = total_train_loss / len(train_loader)
        val_loss = validate(model, val_loader, device)
        print(
            f"[Epoch {epoch:02d}] "
            f"Train L1: {train_loss:.6f} | "
            f"Val L1: {val_loss:.6f}"
        )
        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model (val={best_val:.6f})")

    print("Training finished.")
    print(f"Best validation loss: {best_val:.6f}")
def eval():
    device = "cuda"
    model = AuxHead().to(device)
    model.load_state_dict(torch.load("latent_flow_net.pth"))
    model.eval()
    # ---------------------------
    # 加载测试集
    # ---------------------------
    test_flow_folder = r"G:\UAVDT\CogVideo-Track-Tiny\test"
    test_latent_folder = r"G:\UAVDT\CogVideo-Track-Tiny\test"
    test_set = AuxHeadDataset(test_flow_folder, test_latent_folder)
    test_loader = DataLoader(test_set, batch_size=4, shuffle=False)
    # ---------------------------
    # 评估指标
    # ---------------------------
    def compute_epe(flow_pred, flow_gt):
        # flow: [B,T,2,H,W]
        return torch.norm(flow_pred - flow_gt, dim=2).mean()
    # ---------------------------
    # 验证循环
    # ---------------------------
    total_l1 = 0
    total_epe = 0
    count = 0
    with torch.no_grad():
        for latent, flow_gt in test_loader:
            latent = latent.to(device)
            flow_gt = flow_gt.to(device)
            flow_pred = model(latent.float())
            # 忽略 t=0 (第一帧为0，无监督意义)
            flow_pred = flow_pred[:, 1:]
            flow_gt = flow_gt[:, 1:]
            l1 = (flow_pred - flow_gt).abs().mean()
            epe = compute_epe(flow_pred, flow_gt)
            total_l1 += l1.item()
            total_epe += epe.item()
            count += 1

    print("====================================")
    print(f"Test L1 Loss : {total_l1 / count:.6f}")
    print(f"Test EPE     : {total_epe / count:.6f}")
    print("====================================")
if __name__ == '__main__':
    train()