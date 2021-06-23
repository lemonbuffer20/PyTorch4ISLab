# Day 1: Environment Setting

## Key

1. You should be familiar with CLI
2. You should be familiar with Pycharm & Conda
3. You should be familiar with Github & Git

## Environment Setting

1. Install Git (Windows: https://git-scm.com)
2. Install miniconda Python=3.8 (https://docs.conda.io/en/latest/miniconda.html)
3. Check if current machine supports NVIDIA CUDA (nvidia-smi) and check driver version.
4. Open up Anaconda Powershell Prompt (Windows) or Terminal (Linux)
5. Follow below

```shell
# 1. create new environment and activate
conda create --name torch python=3.8
conda activate torch

# 2. install packages in conda
# (search package in conda first, then pip)
conda install numpy scipy matplotlib pytest pylint imageio pillow pyarrow python-lmdb yaml tqdm

# 3. install pytorch, see pytorch.org
# (CPU Only)
conda install pytorch torchvision torchaudio cpuonly -c pytorch
# (GPU Windows)
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
# (GPU Linux)
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia

# 4. install packages that are not supported by conda via pip
pip install hydra-core fire wandb

# 5. check packages
conda list
```

## Pycharm Setting

1. Install Pycharm (good to make JetBrain account with @snu.ac.kr)
2. Set interpreter (don't forget to check on 'visible for all projects')

## Github Setting

1. Make account for Github (https://github.com)
2. Make your first private repository (check README and .gitignore)
3. Clone it to your local machine

```shell
# clone from origin (=github)
git clone https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY_NAME
# pull from origin (PyCharm will do)
git pull origin
# push to origin (PyCharm will do) 
git push origin
```

## WanDB Setting

1. Make account for WanDB (https://wandb.ai)
2. Copy your API key (settings - API keys)
3. Set to your machine

```shell
wandb login
```

## First Commit

1. Open your repository in PyCharm
2. Change README.md file (file should be 'green', which means it is changed.)

```shell
Author: YOUR_EMAIL
```

3. Change your .gitignore file

```shell
# these files will be not tracked by Git.

# IDE
.idea/
.vs/

# Data
*.jpg
*.png
*.jpeg
*.txt
*.zip
*.tar.gz
*.tar
*.pth
*.ckpt

# Log
result/
results/
output/
outputs/
wandb/
logs/
lightning_logs/
checkpoints/
```

4. Your first commit & first push

* (menu) Git - Commit (Ctrl + K)
* (menu) Git - Push (Ctrl + Shift + K)
* You don't have to start from Github, but it is ALWAYS good to track by Git.

5. See your file history.

* (right click on file) Local history - Show history
* You can revert it to any place you were in.

## First Python Run

1. Understand hello_world.py (try to type it by yourself.)
2. In terminal, move to current folder.
3. Run hello_world.py in terminal.
4. ... or, run in Pycharm (Shift + F10 or Shift + F9)

```shell
cd Pytorch4ISLab
# We should use -m flag because
# hello_world.py cannot find 'torch4is' because it is not in PYTHONPATH
# You may set PYTHONPATH manually.
python -m day01.hello_world
```

## Pycharm is Best

* Autocomplete (Ctrl + Space)
* Undo, Cut, Copy, Paste (Ctrl + Z, X, C, V)
* Duplicate line (Ctrl + D)
* Remove line (Ctrl + Y)
* Move line up/down (Ctrl + Shift + UP/DOWN)
* Goto declaration (Ctrl + B)
* Comment/Uncomment line (Ctrl + /)
* Expand/Collapse paragraph (Ctrl + .)
* Find (Ctrl + F)
* Find and replace (Ctrl + FR)
* Global find (Ctrl + Shift + F)
* Global find and replace (Ctrl + Shift + FR)
* Rename (Shift + F6)
* Reformat (Ctrl + Alt + L)
