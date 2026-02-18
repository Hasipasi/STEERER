# STEERER Setup Guide

This guide documents the complete setup process for the STEERER repository.

## Prerequisites

- Ubuntu/Linux system with NVIDIA GPU (CUDA 11.3+)
- Python 3.10+
- Git
- Raw JHU dataset at `/home/bodis/data/jhu_crowd_v2.0/`

## Directory Structure

```
/home/bodis/
├── STEERER/                    # This repository
├── ProcessedData/              # Preprocessed datasets
│   ├── JHU/
│   ├── SHHB/
│   ├── SHHA/
│   └── ...
└── PretrainedModels/          # Model weights
    └── JHU.pth
```

## Step-by-Step Setup

### 1. Clone Repository
```bash
cd /home/bodis
git clone https://github.com/taohan10200/STEERER.git
cd STEERER
```

### 2. Create Python Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install PyTorch with CUDA 11.3
```bash
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

### 4. Install Additional Dependencies
```bash
pip install matplotlib opencv-python
pip install -r requirements.txt
```

### 5. Create Directory Structure
```bash
cd /home/bodis
mkdir -p ProcessedData/{SHHB,SHHA,NWPU,QNRF,JHU,MTC,JHUTRANCOS_v3,TREE,FDST}
mkdir -p PretrainedModels
```

### 6. Download JHU Pretrained Model
```bash
cd /home/bodis/PretrainedModels
wget -O JHU.pth "https://pjlab-my.sharepoint.cn/:u:/g/personal/hantao_dispatch_pjlab_org_cn/EYjeF4H3Xw9GlYvtYOhygCEBS7N39Si_izSr9jRH2Pslfg?e=KgIgbe&download=1"
```

### 7. Update Preprocessing Script Paths

Edit `/home/bodis/STEERER/lib/datasets/prepare/prepare_JHU.py`:

```python
# Change these lines (around line 13-14):
Root = '/home/bodis/data/jhu_crowd_v2.0'
dst_Root = '/home/bodis/ProcessedData/JHU'
```

Also update line 37 to filter only .jpg files:
```python
file_list = [f for f in os.listdir(imgs_path) if f.endswith('.jpg')]
```

And uncomment the localization GT generation (lines 360-373):
```python
if __name__ == '__main__':
    #================1. resize images and gt===================
    resize_images('train')
    resize_images('val')
    resize_images('test')

    # ================3. train test val id==================
    JHU_list_make('test')
    JHU_list_make('val')
    JHU_list_make('train')

    # ================4. generate val_loc_gt.txt and test_loc_gt.txt==================
    loc_gt_make(mode = 'test')
    loc_gt_make(mode='val')
```

### 8. Run JHU Dataset Preprocessing
```bash
cd /home/bodis/STEERER/lib/datasets/prepare
/home/bodis/STEERER/.venv/bin/python prepare_JHU.py
```

This will:
- Resize all images to be compatible with the network (multiples of 16)
- Generate JSON annotation files
- Create train/val/test split files
- Generate localization ground truth files

### 9. Create Combined Test Set
```bash
cd /home/bodis/ProcessedData/JHU
cat test.txt val.txt > test_val.txt
```

### 10. Fix Shell Scripts

Update test.sh and train.sh to use bash:
```bash
cd /home/bodis/STEERER
sed -i '1s|#!/usr/bin/env sh|#!/bin/bash|' test.sh
sed -i '1s|#!/usr/bin/env sh|#!/bin/bash|' train.sh
```

### 11. Configure JHU Test Settings

Edit `configs/JHU_final.py` for GPU memory constraints:

```python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

gpus = (0,)  # Single GPU

test = dict(
    image_size=(384, 512),  # Reduced for 11GB GPU
    base_size=None,
    loc_base_size=None,
    loc_threshold = 0.10,
    batch_size_per_gpu=1,
    patch_batch_size=1,  # Minimum for memory
    flip_test=False,
    multi_scale=False,
    model_file='',
)
```

### 12. Run Test
```bash
cd /home/bodis/STEERER
bash test.sh configs/JHU_final.py ../PretrainedModels/JHU.pth 0
```

## Expected Results

According to the paper, the JHU model should achieve:
- **MAE**: 54.5
- **MSE**: 240.6
- **F1-measure**: 65.6%
- **Precision**: 66.7%
- **Recall**: 64.6%

## Dataset Statistics

**JHU Dataset (Preprocessed):**
- Total images: 4,372
- Training: 2,272 images
- Validation: 501 images
- Testing: 1,600 images

## Troubleshooting

### CUDA Out of Memory
If you get OOM errors, further reduce test settings in `configs/JHU_final.py`:
```python
test = dict(
    image_size=(256, 384),  # Even smaller
    patch_batch_size=1,
)
```

### Missing Files
Ensure the raw JHU dataset is at `/home/bodis/data/jhu_crowd_v2.0/` with this structure:
```
jhu_crowd_v2.0/
├── train/
│   ├── images/
│   └── gt/
├── val/
│   ├── images/
│   └── gt/
└── test/
    ├── images/
    └── gt/
```

### Python Environment
Always activate the virtual environment before running:
```bash
source /home/bodis/STEERER/.venv/bin/activate
```

## Training (Optional)

To train from scratch:
```bash
bash train.sh configs/JHU_final.py 0
```

Note: Training requires significant time and GPU resources.
