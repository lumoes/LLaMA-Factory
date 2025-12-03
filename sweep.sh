#!/bin/bash

# ================= 配置区 =================
# 激活环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate v1  # 你的环境名

# 基础模型路径
MODEL_PATH="/root/autodl-tmp/models/llama3-8b"

# 结果保存根目录
BASE_OUTPUT_DIR="saves/llama3-8b/layer_scan"

# 扫描的层号列表 (步长为 4，加上最后一层 31)
# 对应: 0, 4, 8,10, 12, 16, 20, 24, 28, 31
LAYERS_TO_SCAN=( 0 4 8 10 12 16 20 24 28 31)

# 统一参数
LR="1e-4"           # 假设这是你测出的最佳 LR
RANK="256"          # High-Rank 模拟知识注入
STEPS="200"         # 快速扫描，200步看 Loss 足够了
# =========================================

echo "🚀 Starting Layer Sensitivity Scan..."
echo "Layers to scan: ${LAYERS_TO_SCAN[@]}"

for LAYER_ID in "${LAYERS_TO_SCAN[@]}"; do
    echo "----------------------------------------------------"
    echo "🧪 Processing Layer: $LAYER_ID"
    echo "----------------------------------------------------"

    # [关键技术] 动态构建 lora_target 字符串
    # LLaMA-Factory 支持后缀匹配，这里我们构造唯一的后缀来锁定该层
    # Llama-3 结构: model.layers.16.mlp.gate_proj
    TARGET="layers.${LAYER_ID}.mlp.gate_proj,layers.${LAYER_ID}.mlp.up_proj,layers.${LAYER_ID}.mlp.down_proj"
    
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/layer_${LAYER_ID}"

    # 启动训练
    # 注意：我们关闭了 do_eval 以节省时间，直接看 training loss 或者最后跑一次 eval
    # 如果你想看 eval loss，把 --do_eval true 加上，并设置 val_size
    CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
        --stage sft \
        --do_train \
        --model_name_or_path $MODEL_PATH \
        --template llama3 \
        --dataset math \
        --finetuning_type lora \
        --lora_rank $RANK \
        --lora_alpha 512 \
        --lora_target "$TARGET" \
        --output_dir $OUTPUT_DIR \
        --overwrite_output_dir \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 4 \
        --learning_rate $LR \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.1 \
        --max_steps $STEPS \
        --logging_steps 10 \
        --save_steps $STEPS \
        --save_total_limit 1 \
        --gradient_checkpointing true \
        --bf16 \
        --trust_remote_code true

    echo "✅ Layer $LAYER_ID finished. Saved to $OUTPUT_DIR"
done

echo "🎉 All Scans Completed!"