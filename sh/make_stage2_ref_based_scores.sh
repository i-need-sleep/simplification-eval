#!/bin/bash
#SBATCH --job-name=stage_2_ref_free       # 任务名
#SBATCH --nodes=1                   # 这里不用动 多节点脚本请查官方文档
#SBATCH --ntasks=1                  # 这里不用动 多任务脚本请查官方文档
#SBATCH --cpus-per-task=4           # 要几块CPU (一般4块就够用了)
#SBATCH --mem=64GB                 # 最大内存
#SBATCH --time=23:00:00           # 运行时间上限
#SBATCH --mail-type=END             # ALL / END
#SBATCH --mail-user=yh2689@nyu.edu  # 结束之后给哪里发邮件
#SBATCH --output=./logs/stage2/%x%A.out           # 正常输出写入的文件
#SBATCH --error=./logs/stage2/%x%A.err            # 报错信息写入的文件
#SBATCH -q cpu-512                  

nvidia-smi
nvcc --version
cd /l/users/yichen.huang/simplification-eval/code   # 切到程序目录

echo "START"               # 输出起始信息
source /apps/local/anaconda3/bin/activate tim          # 调用 virtual env
python -u make_ref_based_supervision.py
echo "FINISH"                       # 输出起始信息
