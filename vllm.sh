#!/bin/bash

#SBATCH --account=rbwei                     # 账户名
#SBATCH --job-name=qwen-inference         # 作业名称
#SBATCH --partition=RTX3090               # 分区
#SBATCH --nodes=1                            # 申请的节点数量
#SBATCH --ntasks-per-node=8                  # 每个节点的任务数
#SBATCH --cpus-per-task=12                   # 每个进程的CPU数量
#SBATCH --gres=gpu:8                         # 使用的GPU数量
#SBATCH --mail-type=end                      # 设置邮件通知类型
#SBATCH --nodelist=node2                        # 使用的GPU数量
#SBATCH --mail-user=weirubinn@gmail.com      # 通知邮箱
#SBATCH --output=%j.out                      # 标准输出文件路径
#SBATCH --error=%j.err                       # 标准错误文件路径

# torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 load_model.py
# OMP_NUM_THREADS=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=4 --nnodes=1 fsdp.py
python run_vllm.py