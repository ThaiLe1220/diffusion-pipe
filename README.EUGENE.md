# Complete WAN 2.1 1.3B LoRA Training Guide for RTX 3090

Training LoRAs on WAN (Video Anything Now) 2.1 1.3B model is highly feasible on RTX 3090 systems, with **proven training times of 2.5 hours for 3500 steps** using small datasets. The RTX 3090's 24GB VRAM provides excellent performance for the 1.3B model, utilizing less than 50% of available memory while delivering professional-quality results.

## Foolproof installation sequence

**Complete setup in order - DO NOT SKIP STEPS:**

```bash
# Step 1: Clean environment creation
conda deactivate  # Exit any existing environment
conda env remove -n diffusion-pipe -y  # Remove if exists
conda create -n diffusion-pipe python=3.12 -y
conda activate diffusion-pipe

# Step 2: CUDA toolkit installation
conda install nvidia::cuda-toolkit=12.4 -y

# Step 3: Environment variables setup
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Step 4: PyTorch installation (exact versions)
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --extra-index-url https://download.pytorch.org/whl/cu124

# Step 5: Diffusion-pipe requirements
cd ~/Desktop/your-path/diffusion-pipe  # Adjust path
pip install -r requirements.txt

# Step 6: Compatible flash-attn (NOT latest version)
pip install flash-attn==2.6.3 --no-build-isolation

# Step 7: Verification
python -c "import torch; import flash_attn; print('✅ All working!')"
```

**If ANY step fails, start over from Step 1. Do not attempt partial fixes.**

## Environment setup requirements

**Python version compatibility**
⚠️ **CRITICAL:** Use **Python 3.12 ONLY**. Python 3.13+ causes compilation failures, Python 3.11 has package conflicts, and Python 3.10 lacks modern features.

```bash
# Always create dedicated environment - never use base environment
conda create -n diffusion-pipe python=3.12 -y
conda activate diffusion-pipe
```

**PyTorch and CUDA requirements**
Use **PyTorch 2.4.1 with CUDA 12.4** for RTX 3090. Install CUDA toolkit first to avoid compilation issues:

```bash
# Install CUDA toolkit for compilation
conda install nvidia::cuda-toolkit=12.4 -y

# Set up environment variables
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install exact PyTorch versions
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --extra-index-url https://download.pytorch.org/whl/cu124
```

**Essential dependencies installation**
Install diffusion-pipe requirements BEFORE flash-attn to avoid version conflicts:

```bash
git clone --recurse-submodules https://github.com/tdrussell/diffusion-pipe
cd diffusion-pipe
pip install -r requirements.txt

# Install compatible flash-attn version (NOT latest)
pip install flash-attn==2.6.3 --no-build-isolation
```

**RTX 3090-specific environment variables**
For optimal performance, set these environment variables before training:

```bash
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Dataset preparation requirements

**Video format specifications**
WAN 2.1 1.3B model performs optimally with **832x480 resolution at 16-24 FPS** with video durations of 2-5 seconds. The model supports both MP4 and converted MOV formats, with automatic truncation to approximately 4-5 second clips during preprocessing.

**Dataset structure and organization**
The training system requires a specific directory structure with paired image/video and caption files:

```
/diffusion-pipe/data/input/
├── image_1.jpg
├── image_1.txt
├── image_2.jpg  
├── image_2.txt
├── video_1.mp4
├── video_1.txt
```

**Annotation requirements**
Each media file requires a corresponding UTF-8 encoded `.txt` file containing descriptive captions. Effective captions include consistent trigger words, detailed scene descriptions, and natural language rather than keyword lists. For example: "A [TRIGGER_WORD] model wearing a white tank top, taking professional shoot, having messy brown colored hair, outdoor setting with soft lighting."

**Dataset size recommendations**
For RTX 3090 training, **15-30 high-quality images** provide optimal results for character or style LoRAs. Minimum viable datasets contain 7-15 images, while complex concept training may require 50-100 samples.

**Video preprocessing pipeline**
Essential preprocessing uses FFmpeg for format conversion, resolution adjustment, and frame rate optimization:

```bash
# Resize to optimal resolution
ffmpeg -i input.mp4 -vf scale=832:480 -c:a copy output_480p.mp4

# Convert to 16 FPS (native training rate)
ffmpeg -i input.mp4 -r 16 -c:v libx264 -crf 18 output_16fps.mp4

# Trim to 5-second clips
ffmpeg -i input.mp4 -t 5 -c copy output_5sec.mp4
```

## Working codebase and installation instructions

**Primary recommendation: diffusion-pipe**
The most reliable and actively maintained repository is **tdrussell/diffusion-pipe**, updated through April 2025 with comprehensive WAN 2.1 support and RTX 3090 optimizations.

```bash
git clone --recurse-submodules https://github.com/tdrussell/diffusion-pipe
conda create -n diffusion-pipe python=3.12
conda activate diffusion-pipe
pip install torch==2.4.1 torchvision==0.19.1 torchaudio --extra-index-url https://download.pytorch.org/whl/cu124
conda install nvidia::cuda-nvcc
pip install -r requirements.txt
```

**GUI alternative: musubi-tuner-wan-gui**
For users preferring graphical interfaces, **Kvento/musubi-tuner-wan-gui** provides Windows-compatible training with reported times of 35 minutes for 32 epochs on RTX 4090.

**Model downloads**
Required models include the base WAN 2.1 model and ComfyUI-compatible components:

```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir models/wan/Wan2.1-T2V-1.3B --exclude "diffusion_pytorch_model*" "models_t5*"

wget -P models/wan https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-T2V-1_3B_bf16.safetensors
wget -P models/wan https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors
wget -P models/wan https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-fp8_e4m3fn.safetensors
```

## Step-by-step training process

**Configuration optimization for RTX 3090**
The training configuration requires careful memory management to maximize RTX 3090's 24GB VRAM efficiency:

```toml
[model]
type = 'wan'
ckpt_path = 'models/wan/Wan2.1-T2V-1.3B'
transformer_path = 'models/wan/Wan2_1-T2V-1_3B_bf16.safetensors'
llm_path = 'models/wan/umt5-xxl-enc-fp8_e4m3fn.safetensors'
dtype = 'bfloat16'
timestep_sample_method = 'logit_normal'
blocks_to_swap = 2  # Memory optimization

[adapter]
type = 'lora'
rank = 64
dtype = 'bfloat16'

[optimizer]
type = 'AdamW8bitKahan'
lr = 5e-5
betas = [0.9, 0.99]
weight_decay = 0.01

[training]
num_epochs = 100
save_every_n_epochs = 10
train_batch_size = 1
gradient_accumulation_steps = 4
mixed_precision = 'bf16'
cache_latents = true
```

**Training execution command**
Launch training with proper environment variables for RTX 3090 optimization:

```bash
NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config examples/wan_video.toml
```

**Memory optimization techniques**
RTX 3090 training benefits from several optimization strategies: **block swapping** (blocks_to_swap = 2-4), **gradient checkpointing** with activation_checkpointing = 'unsloth', and **8-bit optimizer** (AdamW8bitKahan) providing 40% memory savings.

**Training monitoring**
Monitor progress using Tensorboard launched in a separate terminal:

```bash
tensorboard --logdir ./output/[training_run_directory]/tensorboard_logs --port 6006
```

Key metrics include loss values targeting ~0.02xxx for quality LoRA results, training speed of 2-4 seconds per step, and VRAM usage monitored with `watch -n 0.5 nvidia-smi`.

## RTX 3090-specific performance optimizations

**Memory utilization patterns**
RTX 3090's 24GB VRAM provides excellent headroom for WAN 1.3B training, with peak usage typically reaching only 12GB (50% capacity). This allows comfortable training with optimal batch sizes and reduced risk of out-of-memory errors.

**Power and thermal management**
During sustained training, RTX 3090 operates at 350W TDP with temperatures reaching 70-74°C under proper cooling. **Power limiting to 280W provides 93% performance** while significantly improving thermal characteristics and system stability.

**Optimal hardware configuration**
For sustained training workloads, ensure adequate cooling (dual-fan or liquid cooling recommended) and a minimum 750W PSU, with 850W+ preferred for overclocking or multi-component systems.

## Expected training times and resource usage

**Concrete performance benchmarks**
RTX 3090 achieves **2.5 hours training time for 3500 steps** using 14 images at 512 resolution. This represents excellent performance for a 24GB GPU, significantly faster than alternatives like HunyuanVideo LoRA training.

**Dataset scaling expectations**
Training duration scales with dataset size: 100 epochs provide quick testing (~500 steps), while production quality typically requires 200+ epochs (1000+ steps). The optimal sweet spot for RTX 3090 is 14-30 images providing professional results within reasonable timeframes.

**Resource efficiency**
Training speed averages 2-4 seconds per step, with memory usage remaining well below VRAM capacity. The effective batch size of 4 (1 physical × 4 gradient accumulation steps) provides optimal training stability while maintaining memory efficiency.

## Common installation issues and solutions

**Flash-attn import errors**
If you encounter `undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationE` errors:

```bash
# Remove problematic version
pip uninstall flash-attn -y

# Install compatible version
pip install flash-attn==2.6.3 --no-build-isolation

# Verify working installation
python -c "import torch; import flash_attn; print('✅ PyTorch:', torch.__version__); print('✅ CUDA available:', torch.cuda.is_available()); print('✅ flash-attn working')"
```

**PyTorch version conflicts**
If flash-attn installation upgrades PyTorch to 2.7.x causing torchvision/torchaudio conflicts:

```bash
# Fix version conflicts
pip uninstall torch torchvision torchaudio flash-attn -y
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --extra-index-url https://download.pytorch.org/whl/cu124
pip install flash-attn==2.6.3 --no-build-isolation
```

**Python 3.13+ compilation failures**
If using Python 3.13+ and getting compilation errors, recreate environment:

```bash
conda deactivate
conda env remove -n diffusion-pipe -y
conda create -n diffusion-pipe python=3.12 -y
conda activate diffusion-pipe
# Follow installation steps again
```

**CUDA version mismatches**
Ensure your driver CUDA version supports the toolkit version:

```bash
# Check driver version
nvidia-smi | grep "CUDA Version"
# Should show 12.4+ for CUDA 12.4 toolkit compatibility
```

## Validation and testing procedures

**Environment verification**
Before training, verify complete setup:

```bash
# Test all critical components
python -c "
import torch
import flash_attn
print('✅ PyTorch:', torch.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
print('✅ GPU detected:', torch.cuda.get_device_name(0))
print('✅ flash-attn working')
print('✅ VRAM available:', torch.cuda.get_device_properties(0).total_memory // 1024**3, 'GB')
"
```

**Package version compatibility check**
Ensure no version conflicts:

```bash
pip list | grep -E "(torch|flash|cuda)" | sort
# Should show consistent cu124 versions and flash-attn 2.6.3
```

**Training readiness test**
Verify diffusion-pipe can load WAN models:

```bash
python -c "
import sys
sys.path.append('.')
try:
    from src.models.wan import WanModel
    print('✅ WAN model loading capability verified')
except ImportError as e:
    print('❌ Missing dependencies:', e)
"
```

This comprehensive guide provides everything needed to successfully implement WAN 2.1 1.3B LoRA training on Ubuntu with RTX 3090, leveraging proven configurations and optimization techniques for professional-quality results within practical timeframes.