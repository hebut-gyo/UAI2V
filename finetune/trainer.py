import hashlib
import json
import logging
import math
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple
import os
import diffusers
import torch
import transformers
import wandb
from accelerate.accelerator import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    gather_object,
    set_seed,
)
from diffusers.optimization import get_scheduler
from diffusers.pipelines import DiffusionPipeline
from diffusers.utils.export_utils import export_to_video
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from finetune.constants import LOG_LEVEL, LOG_NAME
from finetune.datasets import I2VDatasetWithResize, T2VDatasetWithResize
from finetune.datasets.utils import (
    load_images,
    load_prompts,
    load_videos,
    preprocess_image_with_resize,
    preprocess_video_with_resize,
)
from finetune.schemas import Args, Components, State
from finetune.utils import (
    cast_training_params,
    free_memory,
    get_intermediate_ckpt_path,
    get_latest_ckpt_path_to_resume_from,
    get_memory_statistics,
    get_optimizer,
    string_to_filename,
    unload_model,
    unwrap_model,
)
from pathlib import Path
from safetensors.torch import save_file as safe_save_file
from safetensors.torch import load_file as safe_load_file

logger = get_logger(LOG_NAME, LOG_LEVEL)

_DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,  # FP16 is Only Support for CogVideoX-2B
    "bf16": torch.bfloat16,
}


class Trainer:
    # If set, should be a list of components to unload (refer to `Components``)
    UNLOAD_LIST: List[str] = None

    def __init__(self, args: Args) -> None:
        self.args = args
        self.state = State(
            weight_dtype=self.__get_training_dtype(),
            train_frames=self.args.train_resolution[0],
            train_height=self.args.train_resolution[1],
            train_width=self.args.train_resolution[2],
        )

        self.components: Components = self.load_components()
        self.accelerator: Accelerator = None
        self.dataset: Dataset = None
        self.data_loader: DataLoader = None

        self.optimizer = None
        self.lr_scheduler = None

        self._init_distributed()
        self._init_logging()
        self._init_directories()

        self.state.using_deepspeed = self.accelerator.state.deepspeed_plugin is not None

    def _init_distributed(self):
        logging_dir = Path(self.args.output_dir, "logs")
        project_config = ProjectConfiguration(
            project_dir=self.args.output_dir, logging_dir=logging_dir
        )
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_process_group_kwargs = InitProcessGroupKwargs(
            backend="nccl", timeout=timedelta(seconds=self.args.nccl_timeout)
        )
        mixed_precision = "no" if torch.backends.mps.is_available() else self.args.mixed_precision
        report_to = None if self.args.report_to.lower() == "none" else self.args.report_to

        accelerator = Accelerator(
            project_config=project_config,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=report_to,
            kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
        )

        # Disable AMP for MPS.
        if torch.backends.mps.is_available():
            accelerator.native_amp = False

        self.accelerator = accelerator

        if self.args.seed is not None:
            set_seed(self.args.seed)

    def _init_logging(self) -> None:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=LOG_LEVEL,
        )
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        logger.info("Initialized Trainer")
        logger.info(f"Accelerator state: \n{self.accelerator.state}", main_process_only=False)

    def _init_directories(self) -> None:
        if self.accelerator.is_main_process:
            self.args.output_dir = Path(self.args.output_dir)
            self.args.output_dir.mkdir(parents=True, exist_ok=True)

    def check_setting(self) -> None:
        # Check for unload_list
        if self.UNLOAD_LIST is None:
            logger.warning(
                "\033[91mNo unload_list specified for this Trainer. All components will be loaded to GPU during training.\033[0m"
            )
        else:
            for name in self.UNLOAD_LIST:
                if name not in self.components.model_fields:
                    raise ValueError(f"Invalid component name in unload_list: {name}")

    def prepare_models(self) -> None:
        logger.info("Initializing models")

        if self.components.vae is not None:
            if self.args.enable_slicing:
                self.components.vae.enable_slicing()
            if self.args.enable_tiling:
                self.components.vae.enable_tiling()

        self.state.transformer_config = self.components.transformer.config

    def prepare_dataset(self) -> None:
        logger.info("Initializing dataset and dataloader")

        if self.args.model_type == "i2v":
            self.dataset = I2VDatasetWithResize(
                **(self.args.model_dump()),
                device=self.accelerator.device,
                max_num_frames=self.state.train_frames,
                height=self.state.train_height,
                width=self.state.train_width,
                trainer=self,
            )
        elif self.args.model_type == "t2v":
            self.dataset = T2VDatasetWithResize(
                **(self.args.model_dump()),
                device=self.accelerator.device,
                max_num_frames=self.state.train_frames,
                height=self.state.train_height,
                width=self.state.train_width,
                trainer=self,
            )
        else:
            raise ValueError(f"Invalid model type: {self.args.model_type}")

        # Prepare VAE and text encoder for encoding
        self.components.vae.requires_grad_(False)
        self.components.text_encoder.requires_grad_(False)
        self.components.vae = self.components.vae.to(
            self.accelerator.device, dtype=self.state.weight_dtype
        )
        self.components.text_encoder = self.components.text_encoder.to(
            self.accelerator.device, dtype=self.state.weight_dtype
        )

        # Precompute latent for video and prompt embedding
        logger.info("Precomputing latent for video and prompt embedding ...")
        tmp_data_loader = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_size=1,
            num_workers=0,
            pin_memory=self.args.pin_memory,
        )
        tmp_data_loader = self.accelerator.prepare_data_loader(tmp_data_loader)
        for _ in tmp_data_loader:
            ...
        self.accelerator.wait_for_everyone()
        logger.info("Precomputing latent for video and prompt embedding ... Done")

        unload_model(self.components.vae)
        unload_model(self.components.text_encoder)
        free_memory()

        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            shuffle=True,
        )

    def prepare_trainable_parameters(self):
        logger.info("Initializing trainable parameters")

        # For mixed precision training we cast all non-trainable weights to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = self.state.weight_dtype

        if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
            # due to pytorch#99272, MPS does not yet support bfloat16.
            raise ValueError(
                "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
            )

        # For LoRA, we freeze all the parameters
        # For SFT, we train all the parameters in transformer model
        for attr_name, component in vars(self.components).items():
            if hasattr(component, "requires_grad_"):
                if self.args.training_type == "sft" and attr_name == "transformer":
                    component.requires_grad_(True)
                else:
                    component.requires_grad_(False)
        if self.args.training_type == "frozen_backbone":
            if self.args.tfe_mgf_enable:
                tfe_path = "./flow_modulator_epoch_999.pth"
                if os.path.exists(tfe_path) and hasattr(self.components.transformer.traj_extractor, "flow_modulator"):
                    checkpoint = torch.load(str(tfe_path), map_location="cpu")
                    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                        sd = checkpoint["model_state_dict"]
                    else:
                        sd = checkpoint
                    self.components.transformer.traj_extractor.flow_modulator.load_state_dict(sd, strict=True)
                    logger.info(f"[CKPT] load_aux_head done <- {str(tfe_path)}", main_process_only=True)
                self.components.transformer.traj_extractor.requires_grad_(True)
                self.components.transformer.traj_extractor.flow_modulator.requires_grad_(False)
                for mgf in self.components.transformer.fuser:
                    mgf.requires_grad_(True)
            if self.args.aux_head_enable:
                aux_path = "./aux_head_best.pth"
                if os.path.exists(aux_path) and hasattr(self.components.transformer, "aux_head"):
                    sd = torch.load(str(aux_path), map_location="cpu")
                    self.components.transformer.aux_head.load_state_dict(sd, strict=True)
                    self.components.transformer.aux_head.requires_grad_(False)
                    logger.info(f"[CKPT] load_aux_head done <- {str(aux_path)}", main_process_only=True)
            self.__prepare_saving_loading_hooks_no_lora()
        if self.args.training_type == "lora" or self.args.training_type == "lora_flow":
            transformer_lora_config = LoraConfig(
                r=self.args.rank,
                lora_alpha=self.args.lora_alpha,
                init_lora_weights=True,
                target_modules=self.args.target_modules,
            )
            self.components.transformer.add_adapter(transformer_lora_config)
            if self.args.tfe_mgf_enable:
                self.components.transformer.traj_extractor.requires_grad_(True)
                for m in self.components.transformer.fuser:
                    m.requires_grad_(True)
            if self.args.aux_head_enable:
                aux_path = "./aux_head_best.pth"
                if os.path.exists(aux_path) and hasattr(self.components.transformer, "aux_head"):
                    sd = torch.load(str(aux_path), map_location="cpu")
                    self.components.transformer.aux_head.load_state_dict(sd, strict=True)
                    self.components.transformer.aux_head.requires_grad_(False)
                    logger.info(f"[CKPT] load_aux_head done <- {str(aux_path)}", main_process_only=True)
            self.__prepare_saving_loading_hooks(transformer_lora_config)

        # Load components needed for training to GPU (except transformer), and cast them to the specified data type
        ignore_list = ["transformer"] + self.UNLOAD_LIST
        self.__move_components_to_device(dtype=weight_dtype, ignore_list=ignore_list)

        if self.args.gradient_checkpointing:
            self.components.transformer.enable_gradient_checkpointing()

    def prepare_optimizer(self) -> None:
        logger.info("Initializing optimizer and lr scheduler")

        # Make sure the trainable params are in float32
        cast_training_params([self.components.transformer], dtype=torch.float32)

        # 分层学习率：零卷积层用更高学习率
        transformer = self.components.transformer
        
        # 收集不同类型的参数
        zero_conv_params = []  # 零卷积层（gamma/beta 的 temporal 层）
        other_te_mgf_params = []  # 其他 TE/MGF 参数
        lora_params = []  # LoRA 参数
        other_params = []  # 其他可训练参数
        
        # 1. 收集零卷积层参数
        # if hasattr(transformer, 'traj_extractor'):
        #     te = transformer.traj_extractor
        #     if hasattr(te, 'flow_modulator'):
        #         # FlowConditionedTE 的零卷积层
        #         if hasattr(te.flow_modulator, 'flow_temporal_gamma'):
        #             zero_conv_params.extend(te.flow_modulator.flow_temporal_gamma.parameters())
        #         if hasattr(te.flow_modulator, 'flow_temporal_beta'):
        #             zero_conv_params.extend(te.flow_modulator.flow_temporal_beta.parameters())
        
        if hasattr(transformer, 'fuser'):
            # MGF 的零卷积层
            for mgf in transformer.fuser:
                if hasattr(mgf, 'flow_gamma_temporal'):
                    zero_conv_params.extend(mgf.flow_gamma_temporal.parameters())
                if hasattr(mgf, 'flow_beta_temporal'):
                    zero_conv_params.extend(mgf.flow_beta_temporal.parameters())
        
        # 创建零卷积参数的 id 集合，用于后续过滤
        zero_conv_param_ids = {id(p) for p in zero_conv_params}
        
        # 2. 收集其他 TE/MGF 参数（排除零卷积层）
        if hasattr(transformer, 'traj_extractor'):
            te = transformer.traj_extractor
            # ✅ 只收集 body 和 conv_in 的参数（排除 flow_modulator）
            if hasattr(te, 'body'):
                for module in te.body:
                    for param in module.parameters():
                        if param.requires_grad and id(param) not in zero_conv_param_ids:
                            other_te_mgf_params.append(param)
            
            if hasattr(te, 'conv_in'):
                for param in te.conv_in.parameters():
                    if param.requires_grad and id(param) not in zero_conv_param_ids:
                        other_te_mgf_params.append(param)
        
        if hasattr(transformer, 'fuser'):
            for mgf in transformer.fuser:
                for param in mgf.parameters():
                    if param.requires_grad and id(param) not in zero_conv_param_ids:
                        other_te_mgf_params.append(param)
        
        # 创建 TE/MGF 参数的 id 集合
        te_mgf_param_ids = zero_conv_param_ids | {id(p) for p in other_te_mgf_params}
        
        # 3. 收集 LoRA 参数
        for name, param in transformer.named_parameters():
            if param.requires_grad and 'lora_' in name:
                lora_params.append(param)
        
        lora_param_ids = {id(p) for p in lora_params}
        
        # 4. 收集其他可训练参数
        for param in transformer.parameters():
            if param.requires_grad and id(param) not in te_mgf_param_ids and id(param) not in lora_param_ids:
                other_params.append(param)
        
        # 构建参数组（按优先级排序）
        params_to_optimize = []
        
        # 零卷积层：最高学习率
        if zero_conv_params:
            zero_conv_lr = self.args.learning_rate * 10.0  # 3倍基础学习率
            params_to_optimize.append({
                "params": zero_conv_params,
                "lr": zero_conv_lr,
                "name": "zero_conv"
            })
            logger.info(f"Zero-conv layers: {len(zero_conv_params)} params, lr={zero_conv_lr:.2e}")
        
        # 其他 TE/MGF 参数：基础学习率
        if other_te_mgf_params:
            params_to_optimize.append({
                "params": other_te_mgf_params,
                "lr": self.args.learning_rate,
                "name": "te_mgf"
            })
            logger.info(f"Other TE/MGF layers: {len(other_te_mgf_params)} params, lr={self.args.learning_rate:.2e}")
        
        # LoRA 参数：较低学习率
        if lora_params:
            lora_lr = self.args.learning_rate * 0.5  # 0.5倍基础学习率
            params_to_optimize.append({
                "params": lora_params,
                "lr": lora_lr,
                "name": "lora"
            })
            logger.info(f"LoRA layers: {len(lora_params)} params, lr={lora_lr:.2e}")
        
        # 其他参数：基础学习率
        if other_params:
            params_to_optimize.append({
                "params": other_params,
                "lr": self.args.learning_rate,
                "name": "other"
            })
            logger.info(f"Other layers: {len(other_params)} params, lr={self.args.learning_rate:.2e}")
        
        # 如果没有分层参数，回退到原始方式
        if not params_to_optimize:
            trainable_parameters = list(
                filter(lambda p: p.requires_grad, transformer.parameters())
            )
            params_to_optimize = [{
                "params": trainable_parameters,
                "lr": self.args.learning_rate,
            }]
        
        # 计算总可训练参数数量
        self.state.num_trainable_parameters = sum(
            p.numel() for group in params_to_optimize for p in group["params"]
        )

        use_deepspeed_opt = (
            self.accelerator.state.deepspeed_plugin is not None
            and "optimizer" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        optimizer = get_optimizer(
            params_to_optimize=params_to_optimize,
            optimizer_name=self.args.optimizer,
            learning_rate=self.args.learning_rate,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            use_deepspeed=use_deepspeed_opt,
        )

        num_update_steps_per_epoch = math.ceil(
            len(self.data_loader) / self.args.gradient_accumulation_steps
        )
        if self.args.train_steps is None:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
            self.state.overwrote_max_train_steps = True

        use_deepspeed_lr_scheduler = (
            self.accelerator.state.deepspeed_plugin is not None
            and "scheduler" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        total_training_steps = self.args.train_steps * self.accelerator.num_processes
        num_warmup_steps = self.args.lr_warmup_steps * self.accelerator.num_processes

        if use_deepspeed_lr_scheduler:
            from accelerate.utils import DummyScheduler

            lr_scheduler = DummyScheduler(
                name=self.args.lr_scheduler,
                optimizer=optimizer,
                total_num_steps=total_training_steps,
                num_warmup_steps=num_warmup_steps,
            )
        else:
            lr_scheduler = get_scheduler(
                name=self.args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_training_steps,
                num_cycles=self.args.lr_num_cycles,
                power=self.args.lr_power,
            )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def prepare_for_training(self) -> None:
        self.components.transformer, self.optimizer, self.data_loader, self.lr_scheduler = (
            self.accelerator.prepare(
                self.components.transformer, self.optimizer, self.data_loader, self.lr_scheduler
            )
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(
            len(self.data_loader) / self.args.gradient_accumulation_steps
        )
        if self.state.overwrote_max_train_steps:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.args.train_epochs = math.ceil(self.args.train_steps / num_update_steps_per_epoch)
        self.state.num_update_steps_per_epoch = num_update_steps_per_epoch

    def prepare_for_validation(self):
        validation_prompts = load_prompts(self.args.validation_dir / self.args.validation_prompts)

        if self.args.validation_images is not None:
            validation_images = load_images(self.args.validation_dir / self.args.validation_images)
        else:
            validation_images = [None] * len(validation_prompts)

        if self.args.validation_videos is not None:
            validation_videos = load_videos(self.args.validation_dir / self.args.validation_videos)
        else:
            validation_videos = [None] * len(validation_prompts)

        self.state.validation_prompts = validation_prompts
        self.state.validation_images = validation_images
        self.state.validation_videos = validation_videos

    def prepare_trackers(self) -> None:
        logger.info("Initializing trackers")
        tracker_name = self.args.tracker_name or "finetrainers-experiment"

        config = self.args.model_dump()

        safe_config = {}
        for k, v in config.items():
            if isinstance(v, (int, float, str, bool)):
                safe_config[k] = v
            else:
                safe_config[k] = str(v)  # 转成字符串，TensorBoard 可以接受

        self.accelerator.init_trackers(tracker_name, config=safe_config)

    def train(self) -> None:
        logger.info("Starting training")

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        self.state.total_batch_size_count = (
            self.args.batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "total samples": len(self.dataset),
            "train epochs": self.args.train_epochs,
            "train steps": self.args.train_steps,
            "batches per device": self.args.batch_size,
            "total batches observed per epoch": len(self.data_loader),
            "train batch size total count": self.state.total_batch_size_count,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")

        global_step = 0
        first_epoch = 0
        initial_global_step = 0

        # Potentially load in the weights and states from a previous save
        (
            resume_from_checkpoint_path,
            initial_global_step,
            global_step,
            first_epoch,
        ) = get_latest_ckpt_path_to_resume_from(
            resume_from_checkpoint=self.args.resume_from_checkpoint,
            num_update_steps_per_epoch=self.state.num_update_steps_per_epoch,
        )
        if resume_from_checkpoint_path is not None:
            self.accelerator.load_state(resume_from_checkpoint_path)

        progress_bar = tqdm(
            range(0, self.args.train_steps),
            initial=initial_global_step,
            desc="Training steps",
            disable=not self.accelerator.is_local_main_process,
        )

        accelerator = self.accelerator
        generator = torch.Generator(device=accelerator.device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator

        def control_warmup(step, warmup_steps):
            return min(1.0, step / warmup_steps)
        free_memory()
        for epoch in range(first_epoch, self.args.train_epochs):
            logger.debug(f"Starting epoch ({epoch + 1}/{self.args.train_epochs})")

            self.components.transformer.train()
            models_to_accumulate = [self.components.transformer]

            for step, batch in enumerate(self.data_loader):
                logger.debug(f"Starting step {step + 1}")
                logs = {}
                with accelerator.accumulate(models_to_accumulate):
                    # These weighting schemes use a uniform timestep sampling and instead post-weight the loss
                    scale = control_warmup(global_step, self.args.lr_warmup_steps)
                    loss = self.compute_loss(batch, scale, global_step)
                    loss_value = loss.detach().item()
                    accelerator.backward(loss)
                    del loss #切断loss
                    def stat(named_params, filt):
                        s, c, none = 0.0, 0, 0
                        for n, p in named_params:
                            if not filt(n):
                                continue
                            if p.grad is None:
                                none += 1
                                continue
                            s += p.grad.detach().float().abs().mean().item()
                            c += 1
                        return s, c, none

                    # 只看 LoRA
                    lora_s, lora_c, lora_none = stat(
                        self.components.transformer.named_parameters(),
                        lambda n: "lora_" in n
                    )

                    te_s, te_c, te_none = stat(
                        self.components.transformer.named_parameters(),
                        lambda n: "traj_extractor" in n
                    )

                    mgf_s, mgf_c, mgf_none = stat(
                        self.components.transformer.named_parameters(),
                        lambda n: "fuser" in n
                    )
                    logs["lora_grad"] = lora_s
                    logs["te_grad"] = te_s
                    logs["mgf_grad"] = mgf_s
                    print(
                        f"[GRAD] lora mean-sum={lora_s:.2e}, cnt={lora_c}, none={lora_none} \n"
                        f"[GRAD] te_grad mean-sum={te_s:.2e}, cnt={te_c}, none={te_none}\n"
                        f"[GRAD] mgf_grad mean-sum={mgf_s:.2e}, cnt={mgf_c}, none={mgf_none}\n"
                    )

                    if accelerator.sync_gradients:
                        if accelerator.distributed_type == DistributedType.DEEPSPEED:
                            grad_norm = self.components.transformer.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = accelerator.clip_grad_norm_(
                                self.components.transformer.parameters(), self.args.max_grad_norm
                            )
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()

                        logs["grad_norm"] = grad_norm

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    self.__maybe_save_checkpoint(global_step)

                logs["loss"] = loss_value
                logs["lr"] = self.lr_scheduler.get_last_lr()[0]
                progress_bar.set_postfix(logs)

                if accelerator.sync_gradients:
                    torch.cuda.empty_cache()

                # Maybe run validation
                should_run_validation = (
                    self.args.do_validation and global_step % self.args.validation_steps == 0
                )
                if should_run_validation:
                    del loss
                    free_memory()
                    self.validate(global_step)

                accelerator.log(logs, step=global_step)

                if global_step >= self.args.train_steps:
                    break

            memory_statistics = get_memory_statistics()
            logger.info(
                f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}"
            )

        accelerator.wait_for_everyone()
        self.__maybe_save_checkpoint(global_step, must_save=True)
        if self.args.do_validation:
            free_memory()
            self.validate(global_step)

        del self.components
        free_memory()
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")

        accelerator.end_training()

    def validate(self, step: int) -> None:
        logger.info("Starting validation")

        accelerator = self.accelerator
        num_validation_samples = len(self.state.validation_prompts)

        if num_validation_samples == 0:
            logger.warning("No validation samples found. Skipping validation.")
            return

        self.components.transformer.eval()
        torch.set_grad_enabled(False)

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before validation start: {json.dumps(memory_statistics, indent=4)}")

        #####  Initialize pipeline  #####
        pipe = self.initialize_pipeline()

        if self.state.using_deepspeed:
            # Can't using model_cpu_offload in deepspeed,
            # so we need to move all components in pipe to device
            # pipe.to(self.accelerator.device, dtype=self.state.weight_dtype)
            self.__move_components_to_device(
                dtype=self.state.weight_dtype, ignore_list=["transformer"]
            )
        else:
            # if not using deepspeed, use model_cpu_offload to further reduce memory usage
            # Or use pipe.enable_sequential_cpu_offload() to further reduce memory usage
            pipe.enable_model_cpu_offload(device=self.accelerator.device)

            # Convert all model weights to training dtype
            # Note, this will change LoRA weights in self.components.transformer to training dtype, rather than keep them in fp32
            pipe = pipe.to(dtype=self.state.weight_dtype)

        #################################

        all_processes_artifacts = []
        for i in range(num_validation_samples):
            if self.state.using_deepspeed and self.accelerator.deepspeed_plugin.zero_stage != 3:
                # Skip current validation on all processes but one
                if i % accelerator.num_processes != accelerator.process_index:
                    continue

            prompt = self.state.validation_prompts[i]
            image = self.state.validation_images[i]
            video = self.state.validation_videos[i]

            if image is not None:
                image = preprocess_image_with_resize(
                    image, self.state.train_height, self.state.train_width
                )
                # Convert image tensor (C, H, W) to PIL images
                image = image.to(torch.uint8)
                image = image.permute(1, 2, 0).cpu().numpy()
                image = Image.fromarray(image)

            if video is not None:
                video = preprocess_video_with_resize(
                    video, self.state.train_frames, self.state.train_height, self.state.train_width
                )
                # Convert video tensor (F, C, H, W) to list of PIL images
                video = video.round().clamp(0, 255).to(torch.uint8)
                video = [Image.fromarray(frame.permute(1, 2, 0).cpu().numpy()) for frame in video]

            logger.debug(
                f"Validating sample {i + 1}/{num_validation_samples} on process {accelerator.process_index}. Prompt: {prompt}",
                main_process_only=False,
            )
            validation_artifacts = self.validation_step(
                {"prompt": prompt, "image": image, "video": video}, pipe
            )

            if (
                self.state.using_deepspeed
                and self.accelerator.deepspeed_plugin.zero_stage == 3
                and not accelerator.is_main_process
            ):
                continue

            prompt_filename = string_to_filename(prompt)[:25]
            # Calculate hash of reversed prompt as a unique identifier
            reversed_prompt = prompt[::-1]
            hash_suffix = hashlib.md5(reversed_prompt.encode()).hexdigest()[:5]

            artifacts = {
                "image": {"type": "image", "value": image},
                "video": {"type": "video", "value": video},
            }
            for i, (artifact_type, artifact_value) in enumerate(validation_artifacts):
                artifacts.update(
                    {f"artifact_{i}": {"type": artifact_type, "value": artifact_value}}
                )
            logger.debug(
                f"Validation artifacts on process {accelerator.process_index}: {list(artifacts.keys())}",
                main_process_only=False,
            )

            for key, value in list(artifacts.items()):
                artifact_type = value["type"]
                artifact_value = value["value"]
                if artifact_type not in ["image", "video"] or artifact_value is None:
                    continue

                extension = "png" if artifact_type == "image" else "mp4"
                filename = f"validation-{step}-{accelerator.process_index}-{prompt_filename}-{hash_suffix}.{extension}"
                validation_path = self.args.output_dir / "validation_res"
                validation_path.mkdir(parents=True, exist_ok=True)
                filename = str(validation_path / filename)

                if artifact_type == "image":
                    logger.debug(f"Saving image to {filename}")
                    artifact_value.save(filename)
                    artifact_value = wandb.Image(filename)
                elif artifact_type == "video":
                    logger.debug(f"Saving video to {filename}")
                    export_to_video(artifact_value, filename, fps=self.args.gen_fps)
                    artifact_value = wandb.Video(filename, caption=prompt)

                all_processes_artifacts.append(artifact_value)

        all_artifacts = gather_object(all_processes_artifacts)

        if accelerator.is_main_process:
            tracker_key = "validation"
            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    image_artifacts = [
                        artifact for artifact in all_artifacts if isinstance(artifact, wandb.Image)
                    ]
                    video_artifacts = [
                        artifact for artifact in all_artifacts if isinstance(artifact, wandb.Video)
                    ]
                    tracker.log(
                        {
                            tracker_key: {"images": image_artifacts, "videos": video_artifacts},
                        },
                        step=step,
                    )

        ##########  Clean up  ##########
        if self.state.using_deepspeed:
            del pipe
            # Unload models except those needed for training
            self.__move_components_to_cpu(unload_list=self.UNLOAD_LIST)
        else:
            pipe.remove_all_hooks()
            del pipe
            # Load models except those not needed for training
            self.__move_components_to_device(
                dtype=self.state.weight_dtype, ignore_list=self.UNLOAD_LIST
            )
            self.components.transformer.to(self.accelerator.device, dtype=self.state.weight_dtype)

            # Change trainable weights back to fp32 to keep with dtype after prepare the model
            cast_training_params([self.components.transformer], dtype=torch.float32)

        free_memory()
        accelerator.wait_for_everyone()
        ################################

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after validation end: {json.dumps(memory_statistics, indent=4)}")
        torch.cuda.reset_peak_memory_stats(accelerator.device)

        torch.set_grad_enabled(True)
        self.components.transformer.train()

    def fit(self):
        self.check_setting()
        self.prepare_models()
        self.prepare_dataset()
        self.prepare_trainable_parameters()
        self.prepare_optimizer()
        self.prepare_for_training()
        if self.args.do_validation:
            self.prepare_for_validation()
        self.prepare_trackers()
        self.train()

    def collate_fn(self, examples: List[Dict[str, Any]]):
        raise NotImplementedError

    def load_components(self) -> Components:
        raise NotImplementedError

    def initialize_pipeline(self) -> DiffusionPipeline:
        raise NotImplementedError

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # shape of input video: [B, C, F, H, W], where B = 1
        # shape of output video: [B, C', F', H', W'], where B = 1
        raise NotImplementedError

    def encode_text(self, text: str) -> torch.Tensor:
        # shape of output text: [batch size, sequence length, embedding dimension]
        raise NotImplementedError

    def compute_loss(self, batch) -> torch.Tensor:
        raise NotImplementedError

    def validation_step(self) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        raise NotImplementedError

    def __get_training_dtype(self) -> torch.dtype:
        if self.args.mixed_precision == "no":
            return _DTYPE_MAP["fp32"]
        elif self.args.mixed_precision == "fp16":
            return _DTYPE_MAP["fp16"]
        elif self.args.mixed_precision == "bf16":
            return _DTYPE_MAP["bf16"]
        else:
            raise ValueError(f"Invalid mixed precision: {self.args.mixed_precision}")

    def __move_components_to_device(self, dtype, ignore_list: List[str] = []):
        ignore_list = set(ignore_list)
        components = self.components.model_dump()
        for name, component in components.items():
            if not isinstance(component, type) and hasattr(component, "to"):
                if name not in ignore_list:
                    setattr(
                        self.components, name, component.to(self.accelerator.device, dtype=dtype)
                    )

    def __move_components_to_cpu(self, unload_list: List[str] = []):
        unload_list = set(unload_list)
        components = self.components.model_dump()
        for name, component in components.items():
            if not isinstance(component, type) and hasattr(component, "to"):
                if name in unload_list:
                    setattr(self.components, name, component.to("cpu"))

    # 在 Trainer 类中新增方法（放在 __prepare_saving_loading_hooks 之后）

    def __prepare_saving_loading_hooks_no_lora(self):
        """为冻结骨干 + 只训练新模块的场景注册保存/加载钩子"""

        def save_model_hook(models, weights, output_dir):
            output_dir = Path(output_dir)
            if self.accelerator.is_main_process:
                for model in models:
                    model = unwrap_model(self.accelerator, model)
                    # 不保存整个模型，只保存新增模块
                    if weights:
                        weights.pop()

                transformer = unwrap_model(self.accelerator, self.components.transformer)

                if hasattr(transformer, "traj_extractor"):
                    te_path = output_dir / "traj_extractor.safetensors"
                    safe_save_file(
                        transformer.traj_extractor.state_dict(), str(te_path)
                    )

                if hasattr(transformer, "fuser"):
                    mgf_path = output_dir / "fuser.safetensors"
                    safe_save_file(
                        transformer.fuser.state_dict(), str(mgf_path)
                    )

                # if hasattr(transformer, "aux_head"):
                #     aux_path = output_dir / "aux_head.safetensors"
                #     safe_save_file(
                #         transformer.aux_head.state_dict(), str(aux_path)
                #     )

                logger.info(
                    f"[CKPT] save_model_hook (no-lora) done -> {str(output_dir)}",
                    main_process_only=True,
                )

        def load_model_hook(models, input_dir):
            input_dir = Path(input_dir)
            # 弹出 models 以避免 accelerate 尝试加载整个模型
            while len(models) > 0:
                models.pop()

            transformer = unwrap_model(self.accelerator, self.components.transformer)

            te_path = input_dir / "traj_extractor.safetensors"
            if te_path.exists() and hasattr(transformer, "traj_extractor"):
                sd = safe_load_file(str(te_path))
                transformer.traj_extractor.load_state_dict(sd, strict=True)
                logger.info(f"[CKPT] loaded traj_extractor <- {te_path}")

            mgf_path = input_dir / "fuser.safetensors"
            if mgf_path.exists() and hasattr(transformer, "fuser"):
                sd = safe_load_file(str(mgf_path))
                transformer.fuser.load_state_dict(sd, strict=True)
                logger.info(f"[CKPT] loaded fuser <- {mgf_path}")

            aux_path = input_dir / "aux_head.safetensors"
            if aux_path.exists() and hasattr(transformer, "aux_head"):
                sd = safe_load_file(str(aux_path))
                transformer.aux_head.load_state_dict(sd, strict=True)
                logger.info(f"[CKPT] loaded aux_head <- {aux_path}")

            logger.info(
                f"[CKPT] load_model_hook (no-lora) done <- {str(input_dir)}",
                main_process_only=True,
            )

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    def __prepare_saving_loading_hooks(self, transformer_lora_config):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            output_dir = Path(output_dir)
            if self.accelerator.is_main_process:
                transformer_lora_layers_to_save = None

                for model in models:
                    if isinstance(
                        unwrap_model(self.accelerator, model),
                        type(unwrap_model(self.accelerator, self.components.transformer)),
                    ):
                        model = unwrap_model(self.accelerator, model)
                        transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    else:
                        raise ValueError(f"Unexpected save model: {model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

                self.components.pipeline_cls.save_lora_weights(
                    output_dir,
                    transformer_lora_layers=transformer_lora_layers_to_save,
                )
                # 2) 额外保存 TE / MGF（仅当开关开启 + 模块存在）
                if self.args.tfe_mgf_enable and self.components.transformer is not None:
                    if hasattr(self.components.transformer, "traj_extractor"):
                        te_path = output_dir / "traj_extractor.safetensors"
                        safe_save_file(self.components.transformer.traj_extractor.state_dict(), str(te_path))
                    if hasattr(self.components.transformer, "fuser"):
                        mgf_path = output_dir / "fuser.safetensors"
                    # 你的 fuser 可能是 ModuleList / list：统一拿 state_dict
                        safe_save_file(self.components.transformer.fuser.state_dict(), str(mgf_path))

                logger.info(f"[CKPT] save_model_hook done -> {str(output_dir)}", main_process_only=True)

        def load_model_hook(models, input_dir):
            input_dir = Path(input_dir)
            if not self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                while len(models) > 0:
                    model = models.pop()
                    if isinstance(
                        unwrap_model(self.accelerator, model),
                        type(unwrap_model(self.accelerator, self.components.transformer)),
                    ):
                        transformer_ = unwrap_model(self.accelerator, model)
                    else:
                        raise ValueError(
                            f"Unexpected save model: {unwrap_model(self.accelerator, model).__class__}"
                        )
            else:
                transformer_ = unwrap_model(
                    self.accelerator, self.components.transformer
                ).__class__.from_pretrained(self.args.model_path, subfolder="transformer")
                transformer_.add_adapter(transformer_lora_config)

            lora_state_dict = self.components.pipeline_cls.lora_state_dict(input_dir)
            transformer_state_dict = {
                f'{k.replace("transformer.", "")}': v
                for k, v in lora_state_dict.items()
                if k.startswith("transformer.")
            }
            incompatible_keys = set_peft_model_state_dict(
                transformer_, transformer_state_dict, adapter_name="default"
            )
            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )
            # 2) 加载 TE / MGF（可选）
            if self.args.tfe_mgf_enable and transformer_ is not None:
                te_path = input_dir / "traj_extractor.safetensors"
                mgf_path = input_dir / "fuser.safetensors"

                if te_path.exists() and hasattr(transformer_, "traj_extractor"):
                    sd = safe_load_file(str(te_path))
                    transformer_.traj_extractor.load_state_dict(sd, strict=True)

                if mgf_path.exists() and hasattr(transformer_, "fuser"):
                    sd = safe_load_file(str(mgf_path))
                    transformer_.fuser.load_state_dict(sd, strict=True)

            logger.info(f"[CKPT] load_model_hook done <- {str(input_dir)}", main_process_only=True)

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    @torch.no_grad()
    def _visualize_traj_warp(self, traj_latent, flows, step):
        """
        可视化 TrajectoryConstrainedMotion 的效果：对比调制前后的轨迹视频
        traj_latent: [B, 16, T, H_latent, W_latent]  (轨迹视频的 VAE latent)
        flows:       [B, 2, T, H_latent, W_latent]    (全局光流)
        """
        if not self.accelerator.is_main_process or step % 10 != 0:
            return

        import os
        from PIL import Image, ImageDraw, ImageFont

        # 修复：访问正确的属性
        flow_modulator = self.components.transformer.traj_extractor.flow_modulator

        save_dir = os.path.join(str(self.args.output_dir), "traj_flow_vis")
        os.makedirs(save_dir, exist_ok=True)

        vae = self.components.vae
        transformer = unwrap_model(self.accelerator, self.components.transformer)
        traj_extractor = transformer.traj_extractor

        # 统一 dtype
        param_dtype = next(traj_extractor.parameters()).dtype
        traj_input = traj_latent.to(dtype=param_dtype)  # [B, 16, T, H, W]
        flows_vis = flows.to(dtype=param_dtype)

        # 调制后的轨迹（这部分是正确的）
        traj_modulated = traj_extractor.flow_modulator(
            traj_input, flows_vis, control_scale=1.0
        )  # [B, 16, T, H, W]

        B, C, T, H, W = traj_modulated.shape

        def decode_frames(latent_5d):
            """将 [B, C, T, H, W] 的 latent 逐帧解码为 PIL 图像列表"""
            frames_pil = []
            for t in range(latent_5d.shape[2]):
                frame_latent = latent_5d[:1, :, t:t+1, :, :]  # [1, 16, 1, H, W]
                frame_latent = frame_latent / vae.config.scaling_factor
                decoded = vae.decode(frame_latent.to(dtype=vae.dtype)).sample  # [1, 3, 1, H*8, W*8]
                decoded = decoded.squeeze(2)  # [1, 3, H*8, W*8]
                decoded = ((decoded + 1) / 2).clamp(0, 1)
                decoded = (decoded[0] * 255).byte().cpu().permute(1, 2, 0).numpy()
                frames_pil.append(Image.fromarray(decoded))
            return frames_pil

        # 解码原始轨迹视频帧
        frames_original = decode_frames(traj_input)
        # 解码调制后轨迹视频帧
        frames_modulated = decode_frames(traj_modulated)

        # 拼接对比图：上行=原始，下行=调制后
        cols = min(7, T)
        rows_per_group = (T + cols - 1) // cols
        w, h = frames_original[0].size

        # 总高度 = 标签行高 + 原始帧行 + 间隔 + 标签行高 + 调制帧行
        label_h = 30
        gap = 10
        total_h = label_h + h * rows_per_group + gap + label_h + h * rows_per_group
        total_w = w * cols

        grid = Image.new('RGB', (total_w, total_h), (40, 40, 40))
        draw = ImageDraw.Draw(grid)

        # 标签
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        draw.text((10, 5), f"Original Traj Video (step {step})", fill=(255, 255, 100), font=font)
        draw.text((10, label_h + h * rows_per_group + gap + 5),
                f"After Flow Constraint Modulation (step {step})", fill=(100, 255, 100), font=font)

        # 贴原始帧
        y_offset = label_h
        for idx, frame in enumerate(frames_original):
            r, c = divmod(idx, cols)
            grid.paste(frame, (c * w, y_offset + r * h))

        # 贴调制后帧
        y_offset = label_h + h * rows_per_group + gap + label_h
        for idx, frame in enumerate(frames_modulated):
            r, c = divmod(idx, cols)
            grid.paste(frame, (c * w, y_offset + r * h))

        grid.save(os.path.join(save_dir, f"compare_step{step:06d}.png"))

        # 额外：保存差异图（调制后 - 原始，放大显示）
        frames_diff = []
        for orig, mod in zip(frames_original, frames_modulated):
            import numpy as np
            orig_np = np.array(orig).astype(np.float32)
            mod_np = np.array(mod).astype(np.float32)
            diff = np.abs(mod_np - orig_np)
            # 放大差异以便观察（×5 并 clamp）
            diff = np.clip(diff * 5, 0, 255).astype(np.uint8)
            frames_diff.append(Image.fromarray(diff))

        diff_grid = Image.new('RGB', (w * cols, h * rows_per_group), (0, 0, 0))
        for idx, frame in enumerate(frames_diff):
            r, c = divmod(idx, cols)
            diff_grid.paste(frame, (c * w, r * h))
        diff_grid.save(os.path.join(save_dir, f"diff_step{step:06d}.png"))

        print(f"[VIS] Saved traj modulation comparison to {save_dir}/compare_step{step:06d}.png")

    def __maybe_save_checkpoint(self, global_step: int, must_save: bool = False):
        if (
            self.accelerator.distributed_type == DistributedType.DEEPSPEED
            or self.accelerator.is_main_process
        ):
            if must_save or global_step % self.args.checkpointing_steps == 0:
                # for training
                save_path = get_intermediate_ckpt_path(
                    checkpointing_limit=self.args.checkpointing_limit,
                    step=global_step,
                    output_dir=self.args.output_dir,
                )
                self.accelerator.save_state(save_path, safe_serialization=True)
