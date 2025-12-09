#!/bin/bash

# ================= 1. 环境配置区 =================
# 请确保你的 conda 环境名正确

# 模型和数据路径 (根据你的实际情况修改)
MODEL_PATH="/home/chenzhican/LLaMA-Factory/Llama-3-8B-Base/LLM-Research/Meta-Llama-3-8B"
BASE_OUTPUT_DIR="saves/llama3-8b/layer_scan_qlora"
SUMMARY_FILE="${BASE_OUTPUT_DIR}/scan_results.csv"

# 扫描的层号列表
# 建议扫描: 0(底层), 4, 8, 12, 16(中层), 20, 24, 28, 31(顶层)
LAYERS_TO_SCAN=( 0 4 8 12 16 20 24 28 31 )

# ================= 2. 16GB 显存优化参数 =================
# [显存救星] 开启 4bit 量化后，Rank 64 是比较平衡的选择
# 如果显存依然紧张，可以将 RANK 降为 32
RANK="32"
ALPHA="64"

# [训练时长] 200步足够看清 Loss 下降趋势
STEPS="200"

# [显存控制] 单卡 BS=1，累积 16 次 => 等效 BS=16
BATCH_SIZE="1"
GRAD_ACCUM="16"

# [学习率] QLoRA 通常需要稍微大一点的 LR
LR="2e-4"
# =======================================================

# 初始化结果文件
mkdir -p $BASE_OUTPUT_DIR
echo "Layer_ID,Training_Loss,Eval_Loss" > $SUMMARY_FILE

echo "🚀 Starting 4-bit QLoRA Layer Scan on 16GB GPU..."
echo "Target Layers: ${LAYERS_TO_SCAN[@]}"

for LAYER_ID in "${LAYERS_TO_SCAN[@]}"; do
    echo "----------------------------------------------------"
    echo "🧪 Processing Layer: $LAYER_ID"
    echo "----------------------------------------------------"

    # 动态构建目标层 (LocFFN 模式：只微调 MLP)
    # 如果你想微调该层所有参数(包括Attention)，请在字符串里加上:
    # ,layers.${LAYER_ID}.self_attn.q_proj,layers.${LAYER_ID}.self_attn.v_proj
    TARGET="layers.${LAYER_ID}.mlp.gate_proj,layers.${LAYER_ID}.mlp.up_proj,layers.${LAYER_ID}.mlp.down_proj"
    
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/layer_${LAYER_ID}"

    # 启动训练
    CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
        --stage sft \
        --do_train \
        --do_eval \
        --model_name_or_path $MODEL_PATH \
        --template llama3 \
        --dataset alpaca_gpt4_en \
        --val_size 0.1 \
        --finetuning_type lora \
        --quantization_bit 4 \
        --lora_rank $RANK \
        --lora_alpha $ALPHA \
        --lora_target "$TARGET" \
        --output_dir $OUTPUT_DIR \
        --overwrite_output_dir \
        --per_device_train_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
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

    # 结果提取逻辑 (自动
    TRAIN_LOSS=$(python3 -c "import json; print(json.load(open('${OUTPUT_DIR}/trainer_state.json'))['log_history'][-2].get('loss', 'N/A'))" 2>/dev/null || echo "N/A")
    EVAL_LOSS=$(python3 -c "import json; print(json.load(open('${OUTPUT_DIR}/trainer_state.json'))['log_history'][-1].get('eval_loss', 'N/A'))" 2>/dev/null || echo "N/A")
    
    echo "${LAYER_ID},${TRAIN_LOSS},${EVAL_LOSS}" >> $SUMMARY_FILE
    
    echo "✅ Layer $LAYER_ID Done. (Train: $TRAIN_LOSS | Eval: $EVAL_LOSS)"
done

echo "🎉 All Scans Completed! Results saved to: $SUMMARY_FILE"