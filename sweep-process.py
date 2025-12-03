import json
import matplotlib.pyplot as plt
import os
import glob

# 配置你的扫描结果路径
base_dir = "saves/llama3-8b/layer_scan"
layers = [0, 4, 8,10, 12, 16, 20, 24, 28, 31]

layer_losses = []

print("正在提取数据...")
for layer in layers:
    json_path = os.path.join(base_dir, f"layer_{layer}", "trainer_state.json")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            # 获取最后一步的 loss
            final_loss = data['log_history'][-1]['train_loss']
            layer_losses.append(final_loss)
            print(f"Layer {layer}: Loss = {final_loss:.4f}")
    except Exception as e:
        print(f"Layer {layer}: 数据缺失 ({e})")
        layer_losses.append(None)
# 1. 创建画板
plt.figure(figsize=(10, 6))

# 2. 绘图
plt.plot(layers, layer_losses, marker='o', linestyle='-', linewidth=2, color='#1f77b4', label='Llama-3-8B FFN Fine-tuning')

# 3. 标注和美化
plt.title("Layer Sensitivity Analysis: GSM8K Fine-tuning", fontsize=16, fontweight='bold')
plt.xlabel("Transformer Layer Index", fontsize=14)
plt.ylabel("Final Training Loss (Lower is Better)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(layers)
plt.legend() # 加个图例显得更专业

# --- 关键修改：保存图片 ---

# 保存为 PNG (用于 PPT 演示，网页展示)
# dpi=300: 设置高分辨率（学术标准）
# bbox_inches='tight': 自动剪裁周围的白边，防止标题或坐标轴被切掉
plt.savefig("layer_sensitivity_result.png", dpi=300, bbox_inches='tight')

# 保存为 PDF (强烈推荐用于 LaTeX 论文)
# 矢量图，无限放大不失真
plt.savefig("layer_sensitivity_result.pdf", format='pdf', bbox_inches='tight')

print("✅ 图片已保存为 layer_sensitivity_result.png 和 .pdf")

# 4. 最后再显示
plt.show()