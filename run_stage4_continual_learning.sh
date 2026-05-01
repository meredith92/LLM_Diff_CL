#!/bin/bash
# Stage 4: Domain B 持续学习脚本
# 使用方法: bash run_stage4_continual_learning.sh

set -e

# ============ 配置参数 ============
BASE_WORK_DIR="work_dirs/continual_experiment"
DOMAIN_A_WORK_DIR="${BASE_WORK_DIR}/stage2_train_domain_a"
DOMAIN_B_WORK_DIR="${BASE_WORK_DIR}/stage4_continual_finetune_domain_b"
CONFIG_FILE="projects/pcb_conductor/configs/segformer_mt_vb.py"

# 持续学习参数
MAX_ITERS=1000          # Domain B微调迭代数
VAL_INTERVAL=200        # 验证间隔
USE_LWF=True            # 启用Learning Without Forgetting
LAM_LWF=1.0             # LWF权重

# 模型参数
BATCH_SIZE=2
LR=6e-5

echo "=========================================="
echo "Stage 4: Domain B 持续学习"
echo "=========================================="
echo ""
echo "配置信息:"
echo "  - Domain A 工作目录: ${DOMAIN_A_WORK_DIR}"
echo "  - Domain B 工作目录: ${DOMAIN_B_WORK_DIR}"
echo "  - 配置文件: ${CONFIG_FILE}"
echo "  - 最大迭代数: ${MAX_ITERS}"
echo "  - 使用 LWF: ${USE_LWF}"
echo "  - LWF 权重: ${LAM_LWF}"
echo ""

# ============ 检查前置条件 ============
echo "检查前置条件..."

if [ ! -d "${DOMAIN_A_WORK_DIR}" ]; then
    echo "❌ 错误: Domain A 工作目录不存在!"
    echo "   ${DOMAIN_A_WORK_DIR}"
    exit 1
fi

echo "✅ Domain A 工作目录存在"

# 检查是否有Domain A的checkpoint
DOMAIN_A_CKPT=$(find "${DOMAIN_A_WORK_DIR}" -name "best_*.pth" -o -name "latest.pth" | head -1)
if [ -z "$DOMAIN_A_CKPT" ]; then
    echo "⚠️  警告: 未找到Domain A的checkpoint"
    echo "   将从头开始训练Domain B"
else
    echo "✅ 找到Domain A checkpoint: ${DOMAIN_A_CKPT}"
fi

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "❌ 错误: 配置文件不存在!"
    echo "   ${CONFIG_FILE}"
    exit 1
fi

echo "✅ 配置文件存在"
echo ""

# ============ 创建工作目录 ============
mkdir -p "${DOMAIN_B_WORK_DIR}"
echo "✅ 工作目录已就绪: ${DOMAIN_B_WORK_DIR}"
echo ""

# ============ 运行训练 ============
echo "=========================================="
echo "开始 Domain B 持续学习训练"
echo "=========================================="
echo ""

python tools/train.py \
    --config ${CONFIG_FILE} \
    --work-dir ${DOMAIN_B_WORK_DIR} \
    --resume \
    --cfg-options \
        train_cfg.max_iters=${MAX_ITERS} \
        train_cfg.val_interval=${VAL_INTERVAL} \
        model.use_lwf=${USE_LWF} \
        model.lam_lwf=${LAM_LWF} \
        model.batch_size=${BATCH_SIZE} \
        optim_wrapper.optimizer.lr=${LR} \
    --amp

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Stage 4 完成!"
    echo "=========================================="
    echo "结果保存在: ${DOMAIN_B_WORK_DIR}"
    echo ""
    echo "下一步: 评估持续学习效果"
    echo "  python projects/pcb_conductor/tools/visualize_results.py"
else
    echo ""
    echo "❌ 训练失败!"
    exit 1
fi
