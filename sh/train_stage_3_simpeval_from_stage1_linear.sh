#!/bin/bash
#SBATCH --job-name=train_stage_3_simpeval_from_stage1_linear       # 任务名
#SBATCH --nodes=1                   # 这里不用动 多节点脚本请查官方文档
#SBATCH --ntasks=1                  # 这里不用动 多任务脚本请查官方文档
#SBATCH --cpus-per-task=4           # 要几块CPU (一般4块就够用了)
#SBATCH --mem=256GB                 # 最大内存
#SBATCH --time=12:00:00           # 运行时间上限
#SBATCH --mail-type=END             # ALL / END
#SBATCH --mail-user=yh2689@nyu.edu  # 结束之后给哪里发邮件
#SBATCH --output=./logs/stage3/%x%A.out           # 正常输出写入的文件
#SBATCH --error=./logs/stage3/%x%A.err            # 报错信息写入的文件
#SBATCH --gres=gpu:1                # 需要几块GPU (同时最多8块)
#SBATCH -p gpu                   # 有GPU的partition
#SBATCH --qos=gpu-8                 # To enable the use of up to 8 GPUs

nvidia-smi
nvcc --version
cd /l/users/yichen.huang/simplification-eval/code   # 切到程序目录

echo "START"               # 输出起始信息
source /apps/local/anaconda3/bin/activate tim          # 调用 virtual env
python -u train_deberta.py \
    --name train_stage_3_simpeval_from_stage1_linear \
    --stage finetune_simpeval \
    --lr 3e-6 \
    --batch_size 10 \
    --batch_size_dev 5 \
    --n_epoch 20 \
    --checkpoint ../results/checkpoints/stage_1_3e-5_linear/lr3e-05_3_5758_0.10180610418319702.bin \
    --head_type linear
echo "FINISH"                       # 输出起始信息
