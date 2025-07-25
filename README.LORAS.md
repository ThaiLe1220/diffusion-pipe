# WAN 2.1 1.3B LoRA Training Guide - Complete Process

## Prerequisites
- diffusion-pipe installed and conda environment activated
- Dataset ready in `data/input/` (paired .jpg and .txt files)
- WAN 2.1 models downloaded in `models/wan/`

## Step 1: Create Dataset Configuration

Create `configs/dataset.toml`:

```toml
# Resolutions to train on
resolutions = [512]

# Enable aspect ratio bucketing
enable_ar_bucket = true

# Aspect ratio settings
min_ar = 0.5
max_ar = 2.0
num_ar_buckets = 7

# Frame buckets (1 for images, higher for videos)
frame_buckets = [1, 16, 24]

[[directory]]
# Path to your dataset
path = 'data/input'
# How many repeats for 1 epoch
num_repeats = 1
```

## Step 2: Create Main Training Configuration

Create `configs/lora_training.toml`:

```toml
# Output directory
output_dir = 'output/lana_del_rey_lora'
# Dataset config file
dataset = 'configs/dataset.toml'

# Training settings
epochs = 150
micro_batch_size_per_gpu = 1
gradient_accumulation_steps = 4
gradient_clipping = 1.0
warmup_steps = 50

# Save settings
save_every_n_epochs = 25
activation_checkpointing = true
save_dtype = 'bfloat16'
caching_batch_size = 1
steps_per_print = 1

[model]
type = 'wan'
ckpt_path = 'models/wan/Wan2.1-T2V-1.3B'
transformer_path = 'models/wan/Wan2_1-T2V-1_3B_bf16.safetensors'
llm_path = 'models/wan/umt5-xxl-enc-fp8_e4m3fn.safetensors'
vae_path = 'models/wan/Wan2_1_VAE_bf16.safetensors'
dtype = 'bfloat16'
timestep_sample_method = 'logit_normal'
blocks_to_swap = 2

[adapter]
type = 'lora'
rank = 64
dtype = 'bfloat16'

[optimizer]
type = 'AdamW8bitKahan'
lr = 5e-5
betas = [0.9, 0.99]
weight_decay = 0.01
```

## Step 3: Set Environment Variables and Start Training

```bash
# Activate environment
conda activate diffusion-pipe

# Set RTX 3090 optimization variables
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Start training
deepspeed --num_gpus=1 train.py --deepspeed --config configs/lora_training.toml
```

## Step 4: Monitor Training (Optional)

In a separate terminal:

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor training progress
tail -f output/lana_del_rey_lora/training.log
```

## Step 5: Check Training Output

After training completes, check the output directory:

```bash
ls -la output/lana_del_rey_lora/
```

**Expected files:**
- `epoch_25.safetensors` - Checkpoint at epoch 25
- `epoch_50.safetensors` - Checkpoint at epoch 50
- `epoch_75.safetensors` - Checkpoint at epoch 75
- `epoch_100.safetensors` - Checkpoint at epoch 100
- `epoch_125.safetensors` - Checkpoint at epoch 125
- `epoch_150.safetensors` - Final trained LoRA
- `config.toml` - Training configuration
- `training.log` - Training logs

## Training Complete

The trained LoRA files (`.safetensors`) are ready to use with ComfyUI or other inference tools. The `epoch_100.safetensors` or `epoch_125.safetensors` typically provide the best results.

**Expected training time on RTX 3090:** 1-2 hours for 150 epochs with ~13 images.