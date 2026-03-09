"""
CogVideoX I2V 批量推理脚本
支持多样本 batch 并行推理，显著减少总推理时间
"""

import argparse
import logging
import os
from typing import Optional, List
from safetensors.torch import load_file
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import CogVideoXDPMScheduler
from diffusers.utils import export_to_video, load_image, load_video
import math
import sys

sys.path.insert(0, '/data/gy/CogVideo')
from finetune.models.pipeline_cogvideox_image2video import CogVideoXImageToVideoPipeline
from finetune.models.cogvideox_transformer_3d import CogVideoXTransformer3DModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def read_list(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_safetensors_to_device(safetensor_path, device, dtype):
    assert os.path.exists(safetensor_path), f"Not found: {safetensor_path}"
    state_dict = load_file(safetensor_path, device="cpu")
    for k in state_dict:
        state_dict[k] = state_dict[k].to(device=device, dtype=dtype, non_blocking=True)
    return state_dict


def clear_vae_cache(vae):
    if hasattr(vae, "decoder") and hasattr(vae.decoder, "conv_cache"):
        vae.decoder.conv_cache = {}
    if hasattr(vae, "conv_cache"):
        vae.conv_cache = {}


def load_pipeline(model_path, dtype):
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        model_path,
        transformer=CogVideoXTransformer3DModel.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=dtype,
        ),
        torch_dtype=dtype,
    )
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    return pipe


def prepare_batch_data(
    data_path, images, prompts, static_flows, camera_flows,
    batch_indices, device, dtype, height, width
):
    """准备一个 batch 的数据"""
    batch_images = []
    batch_prompts = []
    batch_camera_flows = []
    batch_static_flows = []
    batch_names = []

    transform = transforms.ToTensor()

    for idx in batch_indices:
        # Image
        img_name = os.path.basename(images[idx])
        image_path = os.path.join(data_path, "images", img_name)
        batch_images.append(image_path)
        batch_prompts.append(prompts[idx])

        # Flow: [2, T, H, W]
        camera_flow_name = os.path.basename(camera_flows[idx])
        camera_flow_path = os.path.join(data_path, "camera_flows", camera_flow_name)
        camera_flow = np.load(camera_flow_path)["flows"]
        camera_flow = torch.from_numpy(camera_flow).permute(1, 0, 2, 3).contiguous()  # T,2,H,W -> 2,T,H,W
        batch_camera_flows.append(camera_flow)

        # Traj: [T, C, H, W] -> will be handled by pipeline
        static_flow_name = os.path.basename(static_flows[idx])
        static_flow_path = os.path.join(data_path, "object_flows", static_flow_name)
        static_flow = np.load(static_flow_path)["flows"]
        static_flow = torch.from_numpy(static_flow).permute(1, 0, 2, 3).contiguous()  # T,2,H,W -> 2,T,H,W
        batch_static_flows.append(static_flow)

        batch_names.append(os.path.splitext(camera_flow_name)[0])

    # Stack flows: [B, 2, T, H, W]
    batch_camera_flows = torch.stack(batch_camera_flows).to(device=device, dtype=dtype)

    # Stack trajs: [B, T, C, H, W] -> pipeline will handle permute
    batch_static_flows = torch.stack(batch_static_flows).to(device=device, dtype=dtype)

    return batch_images, batch_prompts, batch_camera_flows, batch_static_flows, batch_names


def generate_batch(
    pipe, batch_images, batch_prompts, b_static_flows, b_camera_flows,
    height, width, num_frames, num_inference_steps, guidance_scale, seed, device
):
    """
    批量推理：一次去噪循环生成多个视频
    
    关键修改：将 pipeline.__call__ 的逻辑拆出来手动控制 batch
    """
    # generator = torch.Generator(device=device).manual_seed(seed)
    batch_size = len(batch_images)
    generators = [torch.Generator(device=device).manual_seed(seed) for _ in range(batch_size)]

    with torch.no_grad():
        # 1. Encode prompts (batch)
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=batch_prompts,
            negative_prompt=[""] * batch_size,
            do_classifier_free_guidance=guidance_scale > 1.0,
            num_videos_per_prompt=1,
            device=device,
        )
        do_cfg = guidance_scale > 1.0
        if do_cfg:
            prompt_embeds_combined = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        else:
            prompt_embeds_combined = prompt_embeds

        # 2. Prepare images (batch encode)
        processed_images = []
        for img_path in batch_images:
            img = load_image(img_path)
            img_tensor = pipe.video_processor.preprocess(img, height=height, width=width)
            processed_images.append(img_tensor)
        images_batch = torch.cat(processed_images, dim=0).to(device, dtype=prompt_embeds.dtype)

        # 3. Prepare latents (逐个 encode 再 stack，因为 VAE 编码是轻量的)
        latent_channels = pipe.transformer.config.in_channels // 2
        num_frames_latent = (num_frames - 1) // pipe.vae_scale_factor_temporal + 1

        patch_size_t = pipe.transformer.config.patch_size_t
        additional_frames = 0
        if patch_size_t is not None and num_frames_latent % patch_size_t != 0:
            additional_frames = patch_size_t - num_frames_latent % patch_size_t
            num_frames += additional_frames * pipe.vae_scale_factor_temporal
            num_frames_latent += additional_frames

        h_latent = height // pipe.vae_scale_factor_spatial
        w_latent = width // pipe.vae_scale_factor_spatial

        shape = (batch_size, num_frames_latent, latent_channels, h_latent, w_latent)
        if patch_size_t is not None:
            shape = (batch_size, num_frames_latent + num_frames_latent % patch_size_t, latent_channels, h_latent, w_latent)

        # Encode images to latents
        all_image_latents = []
        for i in range(batch_size):
            img = images_batch[i:i+1].unsqueeze(2)  # [1, C, 1, H, W]
            img_latent = pipe.vae.encode(img).latent_dist.sample(generators[i])
            if not pipe.vae.config.invert_scale_latents:
                img_latent = pipe.vae_scaling_factor_image * img_latent
            else:
                img_latent = 1 / pipe.vae_scaling_factor_image * img_latent
            img_latent = img_latent.permute(0, 2, 1, 3, 4)  # [1, 1, C, H, W]

            # Pad
            padding = torch.zeros(1, num_frames_latent - 1, latent_channels, h_latent, w_latent,
                                device=device, dtype=prompt_embeds.dtype)
            img_latent = torch.cat([img_latent, padding], dim=1)

            if patch_size_t is not None:
                first_frame = img_latent[:, :img_latent.size(1) % patch_size_t, ...]
                img_latent = torch.cat([first_frame, img_latent], dim=1)

            all_image_latents.append(img_latent)

        image_latents = torch.cat(all_image_latents, dim=0)  # [B, F, C, H, W]

        # Random noise
        # latents = torch.randn(shape, generator=generator, device=device, dtype=prompt_embeds.dtype)
        latent_list = []
        single_shape = (1,) + shape[1:]  # [1, F, C, H, W]
        for i in range(batch_size):
            lat = torch.randn(single_shape, generator=generators[i], device=device, dtype=prompt_embeds.dtype)
            latent_list.append(lat)
        latents = torch.cat(latent_list, dim=0)
        latents = latents * pipe.scheduler.init_noise_sigma

        # 4. Prepare rotary embeddings
        image_rotary_emb = (
            pipe._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if pipe.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 5. OFS embedding
        ofs_emb = None if pipe.transformer.config.ofs_embed_dim is None else latents.new_full((1,), fill_value=2.0)

        # 6. Prepare for CFG
        if do_cfg:
            camera_flow_input = torch.cat([b_camera_flows, b_camera_flows], dim=0)
            static_flow_input = torch.cat([b_static_flows, b_static_flows], dim=0)
        else:
            camera_flow_input = b_camera_flows
            static_flow_input = b_static_flows


        # 7. Denoising loop
        # timesteps, num_inference_steps = pipe.scheduler.timesteps, num_inference_steps
        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = pipe.scheduler.timesteps

        old_pred_original_sample = None
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

            latent_image_input = torch.cat([image_latents] * 2) if do_cfg else image_latents
            latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)

            timestep = t.expand(latent_model_input.shape[0])

            with torch.cuda.amp.autocast():
                noise_pred = pipe.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds_combined,
                    static_flow=static_flow_input,
                    camera_flow=camera_flow_input,
                    timestep=timestep,
                    ofs=ofs_emb,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )[0]
            noise_pred = noise_pred.float()

            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if not isinstance(pipe.scheduler, CogVideoXDPMScheduler):
                latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            else:
                latents, old_pred_original_sample = pipe.scheduler.step(
                    noise_pred, old_pred_original_sample, t,
                    timesteps[i - 1] if i > 0 else None,
                    latents, return_dict=False,
                )
            latents = latents.to(prompt_embeds.dtype)

            if (i + 1) % 10 == 0:
                logging.info(f"  Step {i+1}/{len(timesteps)}")

        # 8. Decode (逐个解码以节省显存)
        latents = latents[:, additional_frames:]
        all_videos = []
        for i in range(batch_size):
            single_latent = latents[i:i+1]
            video = pipe.decode_latents(single_latent)
            video = pipe.video_processor.postprocess_video(video=video, output_type="pil")
            all_videos.append(video[0])
            clear_vae_cache(pipe.vae)

    return all_videos


def main():
    parser = argparse.ArgumentParser(description="Batch video generation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Number of videos per batch. 2-4 for 80GB GPU, 1-2 for 24GB")
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data lists
    images = read_list(os.path.join(args.data_path, "images.txt"))
    prompts = read_list(os.path.join(args.data_path, "prompts.txt"))
    camera_flows = read_list(os.path.join(args.data_path, "camera_flows.txt"))
    static_flows = read_list(os.path.join(args.data_path, "static_flows.txt"))
    assert len(images) == len(prompts) == len(camera_flows) == len(static_flows)

    # Load pipeline
    pipe = load_pipeline(args.model_path, dtype)

    # Initialize meta tensors
    for attr in ["traj_extractor", "fuser"]:
        if hasattr(pipe.transformer, attr):
            logging.info(f"Initializing {attr} meta tensors...")
            getattr(pipe.transformer, attr).to_empty(device=device)

    pipe.to(device=device, dtype=dtype)

    # Load custom weights
    if args.lora_path:
        te_path = os.path.join(args.lora_path, "traj_extractor.safetensors")
        if os.path.exists(te_path):
            sd = load_safetensors_to_device(te_path, device, dtype)
            pipe.transformer.traj_extractor.load_state_dict(sd, strict=False)
            logging.info("Loaded traj_extractor weights")

        fuser_path = os.path.join(args.lora_path, "fuser.safetensors")
        if os.path.exists(fuser_path):
            sd = load_safetensors_to_device(fuser_path, device, dtype)
            pipe.transformer.fuser.load_state_dict(sd, strict=False)
            logging.info("Loaded fuser weights")

    # Batch inference
    total = len(images)
    bs = args.batch_size
    num_batches = math.ceil(total / bs)
    logging.info(f"Total: {total} samples, batch_size={bs}, {num_batches} batches")

    for batch_idx in range(num_batches):
        start = batch_idx * bs
        end = min(start + bs, total)
        batch_indices = list(range(start, end))
        current_bs = len(batch_indices)

        logging.info(f"\n{'='*50}")
        logging.info(f"Batch {batch_idx+1}/{num_batches} (samples {start}-{end-1})")

        # Prepare batch data
        batch_images, batch_prompts, batch_camera_flows, batch_static_flows, batch_names = \
            prepare_batch_data(
                args.data_path, images, prompts, static_flows, camera_flows,
                batch_indices, device, dtype, args.height, args.width
            )

        # Generate
        try:
            all_videos = generate_batch(
                pipe, batch_images, batch_prompts, batch_static_flows, batch_camera_flows,
                args.height, args.width, args.num_frames,
                args.num_inference_steps, args.guidance_scale, args.seed, device
            )

            # Save
            for vid_idx, (video, name) in enumerate(zip(all_videos, batch_names)):
                output_path = os.path.join(args.output_dir, f"{name}.mp4")
                export_to_video(video, output_path, fps=args.fps)
                logging.info(f"  Saved: {output_path}")

        except torch.cuda.OutOfMemoryError:
            logging.warning(f"OOM at batch_size={current_bs}! Falling back to single inference...")
            torch.cuda.empty_cache()
            # 回退到逐个推理
            for idx in batch_indices:
                single_indices = [idx]
                b_imgs, b_prompts, b_static_flows, b_camera_flows, b_names = \
                    prepare_batch_data(
                        args.data_path, images, prompts, static_flows, camera_flows,
                        single_indices, device, dtype, args.height, args.width
                    )
                try:
                    vids = generate_batch(
                        pipe, b_imgs, b_prompts, b_static_flows, b_camera_flows,
                        args.height, args.width, args.num_frames,
                        args.num_inference_steps, args.guidance_scale, args.seed, device
                    )
                    output_path = os.path.join(args.output_dir, f"{b_names[0]}.mp4")
                    export_to_video(vids[0], output_path, fps=args.fps)
                    logging.info(f"  Saved (single): {output_path}")
                except Exception as e:
                    logging.error(f"  Failed sample {idx}: {e}")
                torch.cuda.empty_cache()

        torch.cuda.empty_cache()

    logging.info("\nBatch generation completed.")


if __name__ == "__main__":
    main()