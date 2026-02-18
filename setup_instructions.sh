#!/bin/bash
# STEERER Repository Setup Instructions
# Run these commands in order to set up the repository

# 1. Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install PyTorch 1.12.0 with CUDA 11.3
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# 3. Install additional dependencies for preprocessing
pip install matplotlib opencv-python

# 4. Install requirements
pip install -r requirements.txt

# 5. Create directory structure
cd ..
mkdir -p ProcessedData/{SHHB,SHHA,NWPU,QNRF,JHU,MTC,JHUTRANCOS_v3,TREE,FDST}
mkdir -p PretrainedModels

# 6. Download JHU pretrained model (248 MB)
cd PretrainedModels
wget -O JHU.pth "https://pjlab-my.sharepoint.cn/:u:/g/personal/hantao_dispatch_pjlab_org_cn/EYjeF4H3Xw9GlYvtYOhygCEBS7N39Si_izSr9jRH2Pslfg?e=KgIgbe&download=1"
cd ..

# 7. Preprocess JHU dataset (assumes raw data is at ~/data/jhu_crowd_v2.0)
# Edit the paths in the preprocessing script first:
cd STEERER/lib/datasets/prepare
# Run preprocessing (this will take some time)
python prepare_JHU.py
cd ../../..

# 8. Create combined test_val.txt file
cat ProcessedData/JHU/test.txt ProcessedData/JHU/val.txt > ProcessedData/JHU/test_val.txt

# 9. Fix the test.sh script (replace shell with bash)
cd STEERER
sed -i '1s|#!/usr/bin/env sh|#!/bin/bash|' test.sh
sed -i '1s|#!/usr/bin/env sh|#!/bin/bash|' train.sh

# 10. Run the test
bash test.sh configs/JHU_final.py ../PretrainedModels/JHU.pth 0
