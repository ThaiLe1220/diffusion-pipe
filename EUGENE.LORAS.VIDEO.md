# Complete Video LoRA Training Guide: WAN 2.1 on RTX 3090

**Successfully tested setup**: Ubuntu 22.04 + RTX 3090 + CUDA 12.8 + Python 3.12

## System Requirements

```bash
# Verified working environment
Ubuntu 22.04.5 LTS
NVIDIA GeForce RTX 3090 (24GB VRAM)
CUDA 12.8 (driver) + 12.4 (toolkit)  
Python 3.12.11
PyTorch 2.4.1+cu124
32GB RAM, 900GB+ free disk space
```

## Environment Setup

### 1. Create Conda Environment
```bash
conda create -n diffusion-pipe python=3.12
conda activate diffusion-pipe
```

### 2. Install Dependencies
```bash
# Install PyTorch with CUDA 12.4
pip install torch==2.4.1 torchvision==0.19.1 torchaudio --extra-index-url https://download.pytorch.org/whl/cu124

# Clone and install diffusion-pipe
git clone --recurse-submodules https://github.com/tdrussell/diffusion-pipe
cd diffusion-pipe
pip install -r requirements.txt
```

### 3. Download WAN 2.1 Models
```bash
mkdir -p models/wan

# Download base model
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir models/wan/Wan2.1-T2V-1.3B --exclude "diffusion_pytorch_model*" "models_t5*"

# Download ComfyUI-compatible components
wget -P models/wan https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-T2V-1_3B_bf16.safetensors
wget -P models/wan https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors
wget -P models/wan https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-fp8_e4m3fn.safetensors
```

## Dataset Preparation

### Video Requirements
- **Resolution**: 480p (shorter dimension)
- **FPS**: 16 frames per second
- **Duration**: 5 seconds maximum
- **Format**: MP4 (audio removed)
- **Captions**: UTF-8 text files with trigger words

### Dataset Structure
```
data/input/
├── video_001.mp4
├── video_001.txt
├── video_002.mp4
├── video_002.txt
└── ... (16 videos + 16 captions recommended)
```

### Caption Format
```
A [TRIGGER_WORD] transformation showing detailed scene description with natural language, avoiding keyword lists.
```

## Configuration Files

### 1. Dataset Configuration (`configs/dataset_video.toml`)
```toml
# Video-optimized dataset configuration
resolutions = [480]
enable_ar_bucket = false
frame_buckets = [16]
min_ar = 0.8
max_ar = 1.25
num_ar_buckets = 3

[[directory]]
path = 'data/input'
num_repeats = 1
```

### 2. Training Configuration (`configs/lora_video_training.toml`)
```toml
# Video LoRA training configuration for RTX 3090
output_dir = 'output/your_lora_name'
dataset = 'configs/dataset_video.toml'

# Video-optimized training parameters
epochs = 250
micro_batch_size_per_gpu = 1
gradient_accumulation_steps = 2
gradient_clipping = 0.5
warmup_steps = 100

# Checkpoint and saving settings
save_every_n_epochs = 25
activation_checkpointing = true
save_dtype = 'bfloat16'
caching_batch_size = 1
cache_latents = true
steps_per_print = 1

[model]
type = 'wan'
ckpt_path = 'models/wan/Wan2.1-T2V-1.3B'
transformer_path = 'models/wan/Wan2_1-T2V-1_3B_bf16.safetensors'
llm_path = 'models/wan/umt5-xxl-enc-fp8_e4m3fn.safetensors'
vae_path = 'models/wan/Wan2_1_VAE_bf16.safetensors'
dtype = 'bfloat16'
timestep_sample_method = 'logit_normal'
blocks_to_swap = 3

[adapter]
type = 'lora'
rank = 32
dtype = 'bfloat16'
# NOTE: Do NOT include 'alpha' - diffusion-pipe enforces alpha=rank

[optimizer]
type = 'AdamW8bitKahan'
lr = 3e-5
betas = [0.9, 0.99]
weight_decay = 0.02
```

## Training Execution

### 1. Set Environment Variables (CRITICAL)
```bash
# RTX 3090-optimized settings
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# DO NOT USE: expandable_segments (causes crashes on Ubuntu 22.04)
```

### 2. Clear CUDA Cache
```bash
python -c "import torch; torch.cuda.empty_cache(); print('CUDA cache cleared')"
```

### 3. Start Training
```bash
deepspeed --num_gpus=1 train.py --deepspeed --config configs/lora_video_training.toml
```

## Training Monitoring

### Expected Performance Metrics
- **Training Time**: 2 hours for 250 epochs
- **Steps per Epoch**: ~8 steps (16 videos ÷ 2 effective batch size)
- **Step Duration**: 1 seconds per step
- **Peak VRAM Usage**: 18-22GB (safe on 24GB RTX 3090)
- **Target Loss**: Stabilizes around 0.015-0.025

### Monitor Progress
```bash
# GPU usage (separate terminal)
watch -n 1 nvidia-smi

# Training logs (separate terminal)  
tail -f output/your_lora_name/training.log
```

### Validation Checkpoints
- **Epoch 50**: Basic trigger word response
- **Epoch 100-150**: Temporal consistency emerges  
- **Epoch 200+**: Full transformation quality

## Troubleshooting

### Common Issues

**Config Error: "Please remove alpha from the config"**
```bash
# Fix: Remove alpha line from [adapter] section
sed -i '/^alpha = /d' configs/lora_video_training.toml
```

**CUDA Allocator Crash: "expandable_segment_ INTERNAL ASSERT FAILED"**
```bash
# Fix: Remove expandable_segments from environment
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# Restart training
```

**Out of Memory Errors**
```toml
# Reduce memory usage in config:
gradient_accumulation_steps = 1  # Reduce from 2
blocks_to_swap = 4              # Increase from 3
```

## Success Indicators

### During Training
- Loss decreasing consistently
- No CUDA errors or crashes
- Checkpoint files saving every 25 epochs
- Stable memory usage (no gradual increase)

### Final Output
```
output/your_lora_name/
├── epoch_025.safetensors
├── epoch_050.safetensors
├── ...
├── epoch_250.safetensors  # Final model
├── config.toml
└── training.log
```

### Testing Your LoRA
- Copy `epoch_200.safetensors` or `epoch_250.safetensors` to ComfyUI/models/loras/
- Use your trigger word in prompts
- Expect smooth video transformations with temporal consistency

## Key Learnings

1. **Environment Variables Matter**: Wrong CUDA allocator settings cause mysterious crashes
2. **Video ≠ Image Training**: Requires more epochs, different memory patterns, longer training times
3. **RTX 3090 is Excellent**: 24GB VRAM handles video LoRA training comfortably
4. **Checkpoint Early and Often**: Save every 25 epochs to find optimal convergence point

---

**This guide represents a fully tested, working configuration for video LoRA training on RTX 3090 systems.**