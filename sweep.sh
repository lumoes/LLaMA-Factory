#!/bin/bash

# ================= 配置区 =================
source /root/miniconda3/etc/profile.d/conda.sh
conda activate v1

MODEL_PATH="/root/autodl-tmp/models/llama3-8b"
BASE_OUTPUT_DIR="saves/llama3-8b/layer_scan"
SUMMARY_FILE="${BASE_OUTPUT_DIR}/scan_results.csv"  # 结果汇总文件

# 扫描列表
LAYERS_TO_SCAN=( 0 4 8 10 12 16 20 24 28 31 )

LR="2e-4"           # 稍微加大一点，因为只有200步，要让它显形
RANK="256"
STEPS="300"         # 稍微加长一点
# =========================================

# 初始化汇总文件头
mkdir -p $BASE_OUTPUT_DIR
echo "Layer_ID,Training_Loss,Eval_Loss" > $SUMMARY_FILE

echo "🚀 Starting Layer Sensitivity Scan..."

for LAYER_ID in "${LAYERS_TO_SCAN[@]}"; do
    echo "----------------------------------------------------"
    echo "🧪 Processing Layer: $LAYER_ID"
    echo "----------------------------------------------------"

    TARGET="layers.${LAYER_ID}.mlp.gate_proj,layers.${LAYER_ID}.mlp.up_proj,layers.${LAYER_ID}.mlp.down_proj"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/layer_${LAYER_ID}"

    # 训练 + 评估
    CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
        --stage sft \
        --do_train \
        --do_eval \
        --model_name_or_path $MODEL_PATH \
        --template llama3 \
        --dataset math \
        --val_size 0.1 \
        --finetuning_type lora \
        --lora_rank $RANK \
        --lora_alpha 512 \
        --lora_target "$TARGET" \
        --output_dir $OUTPUT_DIR \
        --overwrite_output_dir \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --learning_rate $LR \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.05 \
        --max_steps $STEPS \
        --logging_steps 10 \
        --save_steps $STEPS \
        --save_total_limit 1 \
        --eval_steps $STEPS \
        --bf16 \
        --trust_remote_code true

    # [关键] 自动抓取结果
    # 从 trainer_state.json 中提取最后的 loss (需要 python 或 jq，这里用简单的 grep 提取逻辑)
    # 如果没有 jq，手动看也行，但建议用 python one-liner 提取
    
    # 这里是一个简单的 Python 提取脚本
    TRAIN_LOSS=$(python3 -c "import json; print(json.load(open('${OUTPUT_DIR}/trainer_state.json'))['log_history'][-2]['loss'])" 2>/dev/null || echo "N/A")
    EVAL_LOSS=$(python3 -c "import json; print(json.load(open('${OUTPUT_DIR}/trainer_state.json'))['log_history'][-1]['eval_loss'])" 2>/dev/null || echo "N/A")
    
    echo "${LAYER_ID},${TRAIN_LOSS},${EVAL_LOSS}" >> $SUMMARY_FILE
    
    echo "✅ Layer $LAYER_ID finished. Train Loss: $TRAIN_LOSS | Eval Loss: $EVAL_LOSS"
done

echo "🎉 All Scans Completed! Check results at: $SUMMARY_FILE"