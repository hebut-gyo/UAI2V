import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import torch
from accelerate.logging import get_logger
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from typing_extensions import override
import numpy as np
from finetune.constants import LOG_LEVEL, LOG_NAME

from .utils import (
    load_images,
    load_images_from_videos,
    load_prompts,
    load_videos,
    preprocess_image_with_resize,
    preprocess_video_with_buckets,
    preprocess_video_with_resize,
)


if TYPE_CHECKING:
    from finetune.trainer import Trainer

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

logger = get_logger(LOG_NAME, LOG_LEVEL)

def _extract_frame_index(p: Path) -> int:
    """
    e.g. Frame_0.png / Frame_0012.jpg / 000123.png
    take the last integer in stem as index
    """
    nums = re.findall(r"\d+", p.stem)
    return int(nums[-1]) if nums else -1

class Uavdt(Dataset):
    """
    UAVDT-M/
      M0001/
        Frame_0.png
        Frame_1.png
        ...
      M0002/
        ...
    Output aligns with WebVid:
      data = {
        'video': Tensor [C,T,H,W] in [-1,1],
        'caption': str,
        'path': str,          # folder path
        'fps': int,
        'frame_stride': int
      }
    """
    def __init__(
        self,
        data_root: str,
        flow_root: str,
        frame_stride: int,
        train_resolution: str,
        valid_ids: List[str],
        device: torch.device,
        trainer: "Trainer" = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        data_root = Path(data_root)
        self.valid_ids = valid_ids
        self.train_resolution = train_resolution
        self.frame_stride = frame_stride
        # self.prompts = load_prompts(data_root / caption_column)
        # self.videos = load_videos(data_root / video_column)
        # self.images = load_images_from_videos(self.videos)
        self.trainer = trainer

        self.device = device
        self.encode_video = trainer.encode_video
        self.encode_text = trainer.encode_text
        # 1) scan folders and frames
        self.videos: List[Dict[str, Any]] = []  # each: {"vid": "M0001", "dir": Path, "frames": List[Path]}
        self.flows: List[Dict[str, Any]] = []
        video_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
        if self.valid_ids is not None:
            video_dirs = [p for p in video_dirs if p.name in self.valid_ids]
        for vd in video_dirs:
            frames = [p for p in vd.iterdir() if p.is_file() ]
            if not frames:
                continue
            vd.sort(key=_extract_frame_index)
            self.videos.append({"vid": vd.name, "dir": vd, "frames": frames})

        # 2) build index: either per-video or per-clip
        self.index = []  # list of (video_idx, start_idx)
        for vi, item in enumerate(self.videos):
            n = len(item["frames"])
            if n < self.train_resolution[0]:
                continue
            for s in range(0, n - self.train_resolution[0] + 1, self.train_resolution[0]):
                self.index.append((vi, s))
        print(f">>> UAVDT Dataset, total clips: {len(self.index)}")

    def __len__(self) -> int:
        return len(self.index)
    def _read_frames(self, frame_paths: List[Path]) -> torch.Tensor:
        """
        Read selected frames -> Tensor [C,T,H,W], float32
        """
        # torchvision.io.read_image returns uint8 [C,H,W]
        imgs = []
        for p in frame_paths:
            img = read_image(str(p))  # uint8 [C,H,W], RGB if png/jpg (may be BGR for some? normally RGB)
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            elif img.shape[0] > 3:
                img = img[:3]  # drop alpha if exists
            imgs.append(img)

        frames = torch.stack(imgs, dim=1).float()  # [C,T,H,W], still 0..255
        return frames
    def _load_prompt_txt(self, video_dir: Path) -> str:
        prompt_path = video_dir / "prompt.txt"
        if prompt_path.exists():
            try:
                with open(prompt_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                if len(text) > 0:
                    return text
            except Exception as e:
                print(f"[WARN] Failed to read {prompt_path}: {e}")
        return ""
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if isinstance(idx, list):
            # Here, index is actually a list of data objects that we need to return.
            # The BucketSampler should ideally return indices. But, in the sampler, we'd like
            # to have information about num_frames, height and width. Since this is not stored
            # as metadata, we need to read the video to get this information. You could read this
            # information without loading the full video in memory, but we do it anyway. In order
            # to not load the video twice (once to get the metadata, and once to return the loaded video
            # based on sampled indices), we cache it in the BucketSampler. When the sampler is
            # to yield, we yield the cache data instead of indices. So, this special check ensures
            # that data is not loaded a second time. PRs are welcome for improvements.
            return idx
        while True:
            idx = idx % len(self.index)
            video_i, start_idx = self.index[idx]
            item = self.videos[video_i]
            vdir = item["dir"]
            frames_all: List[Path] = item["frames"]
            frame_num = len(frames_all)
            required_frame_num = frame_stride * (self.train_resolution[0] - 1) + 1
            if frame_num < self.train_resolution[0]:
                # too short
                if self.split_to_clips:
                    idx += 1
                    continue
            # if not enough frames for stride, adapt stride (same spirit as WebVid)
            if frame_num < required_frame_num:
                if frame_num < required_frame_num * 0.5:
                    idx += 1
                    continue
                else:
                    frame_stride = max(frame_num // self.train_resolution[0], 1)
                    required_frame_num = frame_stride * (self.train_resolution[0] - 1) + 1
                    if frame_num < required_frame_num:
                        idx += 1
                        continue
            # choose start index
            if start_idx is None:
                random_range = frame_num - required_frame_num
                start_idx = random.randint(0, random_range) if random_range > 0 else 0

            frame_indices = [start_idx + frame_stride * i for i in range(self.train_resolution[0])]
            if max(frame_indices) >= frame_num:
                # invalid
                idx += 1
                continue
            frame_paths = [frames_all[i] for i in frame_indices]
            try:
                frames = self._read_frames(frame_paths)  # [C,T,H,W], 0..255
                break
            except Exception as e:
                print(f"Read frames failed! dir={vdir}, start={start_idx}, stride={frame_stride}, err={e}")
                idx += 1
                continue

        prompt = self.prompts[index]
        video = self.videos[index]
        image = self.images[index]
        train_resolution_str = "x".join(str(x) for x in self.trainer.args.train_resolution)

        cache_dir = self.trainer.args.data_root / "cache"
        camera_dir = self.trainer.args.data_root / "camera_npz"
        video_latent_dir = (
            cache_dir / "video_latent" / self.trainer.args.model_name / train_resolution_str
        )
        prompt_embeddings_dir = cache_dir / "prompt_embeddings"
        video_latent_dir.mkdir(parents=True, exist_ok=True)
        prompt_embeddings_dir.mkdir(parents=True, exist_ok=True)

        prompt_hash = str(hashlib.sha256(prompt.encode()).hexdigest())
        prompt_embedding_path = prompt_embeddings_dir / (prompt_hash + ".safetensors")
        encoded_video_path = video_latent_dir / (video.stem + ".safetensors")
        camera_npz_path = camera_dir / (video.stem + ".npz")

        if prompt_embedding_path.exists():
            prompt_embedding = load_file(prompt_embedding_path)["prompt_embedding"]
            logger.debug(
                f"process {self.trainer.accelerator.process_index}: Loaded prompt embedding from {prompt_embedding_path}",
                main_process_only=False,
            )
        else:
            prompt_embedding = self.encode_text(prompt)
            prompt_embedding = prompt_embedding.to("cpu")
            # [1, seq_len, hidden_size] -> [seq_len, hidden_size]
            prompt_embedding = prompt_embedding[0]
            save_file({"prompt_embedding": prompt_embedding}, prompt_embedding_path)
            logger.info(
                f"Saved prompt embedding to {prompt_embedding_path}", main_process_only=False
            )

        if encoded_video_path.exists():
            encoded_video = load_file(encoded_video_path)["encoded_video"]
            logger.debug(f"Loaded encoded video from {encoded_video_path}", main_process_only=False)
            # shape of image: [C, H, W]
            _, image = self.preprocess(None, self.images[index])
            image = self.image_transform(image)
        else:
            frames, image = self.preprocess(video, image)
            frames = frames.to(self.device)
            image = image.to(self.device)
            image = self.image_transform(image)
            # Current shape of frames: [F, C, H, W]
            frames = self.video_transform(frames)

            # Convert to [B, C, F, H, W]
            frames = frames.unsqueeze(0)
            frames = frames.permute(0, 2, 1, 3, 4).contiguous()
            encoded_video = self.encode_video(frames)

            # [1, C, F, H, W] -> [C, F, H, W]
            encoded_video = encoded_video[0]
            encoded_video = encoded_video.to("cpu")
            image = image.to("cpu")
            save_file({"encoded_video": encoded_video}, encoded_video_path)
            logger.info(f"Saved encoded video to {encoded_video_path}", main_process_only=False)

        if self.trainer.args.aux_head_enable:
            if not camera_npz_path.exists():
                raise FileNotFoundError(f"Missing npz: {camera_npz_path}")
            with np.load(str(camera_npz_path), mmap_mode='r') as data:
                camera_flow_clip = data['flow']                       # T×H×W×2
            camera_flow_clip = torch.from_numpy(camera_flow_clip).to("cpu")   # T×H×W×2
            camera_flow_clip = camera_flow_clip.permute(3, 0, 1, 2).contiguous()
        else:
            camera_flow_clip = None
        # shape of encoded_video: [C, F, H, W]
        # shape of image: [C, H, W]
        return {
            "image": image,
            "prompt_embedding": prompt_embedding,
            "encoded_video": encoded_video,
            "camera_flow": camera_flow_clip,
            "video_metadata": {
                "num_frames": encoded_video.shape[1],
                "height": encoded_video.shape[2],
                "width": encoded_video.shape[3],
            },
        }

    def preprocess(
        self, video_path: Path | None, image_path: Path | None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads and preprocesses a video and an image.
        If either path is None, no preprocessing will be done for that input.

        Args:
            video_path: Path to the video file to load
            image_path: Path to the image file to load

        Returns:
            A tuple containing:
                - video(torch.Tensor) of shape [F, C, H, W] where F is number of frames,
                  C is number of channels, H is height and W is width
                - image(torch.Tensor) of shape [C, H, W]
        """
        raise NotImplementedError("Subclass must implement this method")

    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Applies transformations to a video.

        Args:
            frames (torch.Tensor): A 4D tensor representing a video
                with shape [F, C, H, W] where:
                - F is number of frames
                - C is number of channels (3 for RGB)
                - H is height
                - W is width

        Returns:
            torch.Tensor: The transformed video tensor
        """
        raise NotImplementedError("Subclass must implement this method")

    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        """
        Applies transformations to an image.

        Args:
            image (torch.Tensor): A 3D tensor representing an image
                with shape [C, H, W] where:
                - C is number of channels (3 for RGB)
                - H is height
                - W is width

        Returns:
            torch.Tensor: The transformed image tensor
        """
        raise NotImplementedError("Subclass must implement this method")


class I2VDatasetWithResize(BaseI2VDataset):
    """
    A dataset class for image-to-video generation that resizes inputs to fixed dimensions.

    This class preprocesses videos and images by resizing them to specified dimensions:
    - Videos are resized to max_num_frames x height x width
    - Images are resized to height x width

    Args:
        max_num_frames (int): Maximum number of frames to extract from videos
        height (int): Target height for resizing videos and images
        width (int): Target width for resizing videos and images
    """

    def __init__(self, max_num_frames: int, height: int, width: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width

        self.__frame_transforms = transforms.Compose(
            [transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)]
        )
        self.__image_transforms = self.__frame_transforms

    @override
    def preprocess(
        self, video_path: Path | None, image_path: Path | None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if video_path is not None:
            video = preprocess_video_with_resize(
                video_path, self.max_num_frames, self.height, self.width
            )
        else:
            video = None
        if image_path is not None:
            image = preprocess_image_with_resize(image_path, self.height, self.width)
        else:
            image = None
        return video, image

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transforms(f) for f in frames], dim=0)

    @override
    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        return self.__image_transforms(image)


class I2VDatasetWithBuckets(BaseI2VDataset):
    def __init__(
        self,
        video_resolution_buckets: List[Tuple[int, int, int]],
        vae_temporal_compression_ratio: int,
        vae_height_compression_ratio: int,
        vae_width_compression_ratio: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.video_resolution_buckets = [
            (
                int(b[0] / vae_temporal_compression_ratio),
                int(b[1] / vae_height_compression_ratio),
                int(b[2] / vae_width_compression_ratio),
            )
            for b in video_resolution_buckets
        ]
        self.__frame_transforms = transforms.Compose(
            [transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)]
        )
        self.__image_transforms = self.__frame_transforms

    @override
    def preprocess(self, video_path: Path, image_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        video = preprocess_video_with_buckets(video_path, self.video_resolution_buckets)
        image = preprocess_image_with_resize(image_path, video.shape[2], video.shape[3])
        return video, image

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transforms(f) for f in frames], dim=0)

    @override
    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        return self.__image_transforms(image)
