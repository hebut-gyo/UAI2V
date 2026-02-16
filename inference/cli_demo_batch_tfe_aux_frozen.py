import argparse
import logging
import os
from typing import Optional
from safetensors.torch import load_file
import torch
from diffusers import (
    CogVideoXDPMScheduler,
    #CogVideoXImageToVideoPipeline,
    CogVideoXPipeline,
    CogVideoXVideoToVideoPipeline,
)
import sys
sys.path.insert(0, '/data/gy/CogVideo')
from finetune.models.pipeline_cogvideox_image2video import CogVideoXImageToVideoPipeline
from finetune.models.cogvideox_transformer_3d import CogVideoXTransformer3DModel
from diffusers.utils import export_to_video, load_image, load_video
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
logging.basicConfig(level=logging.INFO)
from torchvision import transforms

RESOLUTION_MAP = {
    "cogvideox1.5-5b-i2v": (768, 1360),
    "cogvideox1.5-5b": (768, 1360),
    "cogvideox-5b-i2v": (480, 720),
    "cogvideox-5b": (480, 720),
    "cogvideox-2b": (480, 720),
}
from safetensors.torch import load_file, save_file

def clear_vae_cache(vae):
    if hasattr(vae, "decoder") and hasattr(vae.decoder, "conv_cache"):
        vae.decoder.conv_cache = {}
    if hasattr(vae, "conv_cache"):
        vae.conv_cache = {}
def get_resolution(model_name: str, width: Optional[int], height: Optional[int], generate_type: str):
    """统一处理不同模型的分辨率规则"""
    default_h, default_w = RESOLUTION_MAP.get(model_name, (480, 720))

    if width is None or height is None:
        logging.info(f"Using default resolution ({default_h}, {default_w}) for {model_name}")
        return default_h, default_w

    # 只有 I2V 支持自定义分辨率
    if generate_type != "i2v" and (height, width) != (default_h, default_w):
        logging.warning(
            f"{model_name} does not support custom resolution for {generate_type}. "
            f"Falling back to default ({default_h}, {default_w})."
        )
        return default_h, default_w

    return height, width


def load_pipeline(model_path: str, generate_type: str, dtype: torch.dtype):
    """统一创建 pipeline"""
    if generate_type == "i2v":
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            model_path,
            transformer=CogVideoXTransformer3DModel.from_pretrained(
                model_path,
                subfolder="transformer",
                torch_dtype=dtype,
            ),
            torch_dtype=dtype
        )
        # pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
    elif generate_type == "t2v":
        pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    else:
        pipe = CogVideoXVideoToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)

    # 统一调度器
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    return pipe


def generate_single(
    pipe,
    image_path: str,
    prompt: str,
    flow: Optional[torch.FloatTensor],
    traj: Optional[torch.Tensor],
    output_path: str,
    height: int,
    width: int,
    generate_type: str,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    num_videos_per_prompt: int,
    seed: int,
    fps: int
):
    """真正执行生成的部分"""
    generator = torch.Generator().manual_seed(seed)

    if generate_type == "i2v":
        image = load_image(image_path)
        video = None
        result = pipe(
            height=height,
            width=width,
            prompt=prompt,
            image=image,
            # TODO 传入轨迹和光流
            traj_static=traj,
            video_flow=flow,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            use_dynamic_cfg=False,
            generator=generator
        )
    elif generate_type == "v2v":
        image = None
        video = load_video(image_path)
        result = pipe(
            height=height,
            width=width,
            prompt=prompt,
            video=video,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            use_dynamic_cfg=True,
            generator=generator,
        )
    else:
        image = None
        result = pipe(
            height=height,
            width=width,
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            use_dynamic_cfg=True,
            generator=generator,
        )

    frames = result.frames[0]
    export_to_video(frames, output_path, fps=fps)
    logging.info(f"Saved: {output_path}")
    clear_vae_cache(pipe.vae)
    torch.cuda.empty_cache()


def read_list(path: str):
    """更加 robust 的读取"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def load_safetensors_to_device(safetensor_path, device, dtype):
    """
    兼容低版本safetensors的权重加载方法
    :param safetensor_path: 权重文件路径
    :param device: 目标设备（cuda/cpu）
    :param dtype: 目标精度
    :return: 转移到指定设备+精度的state_dict
    """
    assert os.path.exists(safetensor_path), f"权重文件不存在: {safetensor_path}"
    # 第一步：低版本仅支持CPU加载
    state_dict = load_file(safetensor_path, device="cpu")
    # 第二步：手动将每个权重转移到目标设备+精度
    for k in state_dict:
        state_dict[k] = state_dict[k].to(device=device, dtype=dtype, non_blocking=True)
    return state_dict
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch video generation")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--generate_type", type=str, default="i2v")
    parser.add_argument("--lora_path", type=str, default=None)

    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)

    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--num_videos_per_prompt", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}, dtype: {dtype}")

    images = read_list(os.path.join(args.data_path, "images.txt"))
    prompts = read_list(os.path.join(args.data_path, "prompts.txt"))
    # TODO 轨迹图和光流
    flows = read_list(os.path.join(args.data_path, "flows.txt"))
    trajs = read_list(os.path.join(args.data_path, "trajectory_videos.txt"))

    assert len(images) == len(prompts), "images.txt 和 prompts.txt 行数不一致！"

    model_name = args.model_path.split("/")[-1].lower()

    # Pipeline 在外部加载一次即可
    pipe = load_pipeline(args.model_path, args.generate_type, dtype)

    if hasattr(pipe.transformer, "traj_extractor"):
        logging.info("Initializing traj_extractor (Meta Tensor) to CUDA...")
        pipe.transformer.traj_extractor.to_empty(device=device)
    if hasattr(pipe.transformer, "fuser"):
        logging.info("Initializing fuser (Meta Tensor) to CUDA...")
        pipe.transformer.fuser.to_empty(device=device)
    if hasattr(pipe.transformer, "aux_head"):
        logging.info("Initializing aux_head (Meta Tensor) to CUDA...")
        pipe.transformer.aux_head.to_empty(device=device)
        # --------------------------
        # 步骤2：将整个模型转CUDA+指定精度（此时Meta Tensor层已初始化，无报错）
        # --------------------------
    logging.info("Moving entire pipeline to CUDA...")
    pipe.to(device=device, dtype=dtype)

    # --------------------------
    # 步骤3：加载自定义层（traj_extractor/fuser）权重并直接放到CUDA
    # --------------------------
    if args.lora_path and device.type == "cuda":
        logging.info("[Step 5] Loading custom layers (traj_extractor/fuser) weights...")
        # 加载traj_extractor权重（低版本兼容方案）
        te_path = os.path.join(args.lora_path, "traj_extractor.safetensors")
        te_state_dict = load_safetensors_to_device(te_path, device, dtype)
        missing, unexpected = pipe.transformer.traj_extractor.load_state_dict(te_state_dict, strict=False)
        logging.info(f"  traj_extractor - missing: {len(missing)}, unexpected: {len(unexpected)}")

        # 加载fuser权重（低版本兼容方案）
        fuser_path = os.path.join(args.lora_path, "fuser.safetensors")
        fuser_state_dict = load_safetensors_to_device(fuser_path, device, dtype)
        missing, unexpected = pipe.transformer.fuser.load_state_dict(fuser_state_dict, strict=False)
        logging.info(f"  fuser - missing: {len(missing)}, unexpected: {len(unexpected)}")

        aux_path = r"/data/gy/CogVideo/finetune/aux_head_best.pth"
        sd = torch.load(str(aux_path), map_location="cpu")
        missing, unexpected = pipe.transformer.aux_head.load_state_dict(sd, strict=True)
        logging.info(f"  aux_head - missing: {len(missing)}, unexpected: {len(unexpected)}")

    # pipe.enable_model_cpu_offload()
    # pipe.enable_sequential_cpu_offload()

    logging.info(f"Total tasks: {len(images)}")

    for img_name, prompt, traj_name,flow_name in zip(images, prompts, trajs,flows):
        clean_name = os.path.basename(img_name)
        image_path = os.path.join(args.data_path, "images", clean_name)
        clean_name = os.path.basename(traj_name)
        traj_path = os.path.join(args.data_path, "trajectory_videos", clean_name)
        clean_name = os.path.basename(flow_name)
        flow_path = os.path.join(args.data_path, "flows", clean_name)

        flow = np.load(flow_path)["flows"]
        flow = torch.from_numpy(flow)
        flow = flow.permute(1, 0, 2, 3).contiguous()  # T 2 H W -> 2 T H W
        flow.unsqueeze_(0)
        flow = flow.to(device=device,dtype=pipe.transformer.dtype)

        traj = load_video(traj_path)
        transform = transforms.ToTensor()  # 将 PIL Image (H,W,C) -> tensor (C,H,W)，并归一化到 [0,1]
        # 转换所有帧并堆叠
        traj = torch.stack([transform(img) for img in traj])  # (T, C, H, W) (49, 3, 480, 720)

        output_path = os.path.join(
            args.output_dir,
            os.path.splitext(clean_name)[0] + ".mp4",
        )

        height, width = get_resolution(model_name, args.width, args.height, args.generate_type)

        print(type(pipe.transformer))
        print(pipe.transformer.__class__.__module__)
        generate_single(
            pipe=pipe,
            image_path=image_path,
            prompt=prompt,
            # TODO 光流和轨迹
            flow=flow,
            traj=traj,
            output_path=output_path,
            height=height,
            width=width,
            generate_type=args.generate_type,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            num_videos_per_prompt=args.num_videos_per_prompt,
            seed=args.seed,
            fps=args.fps
        )

    logging.info("Batch generation completed.")
