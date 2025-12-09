import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ================= 配置区 =================
# 你的结果文件路径
FILE_PATH = "saves/llama3-8b/layer_scan_qlora/scan_results.csv"
# 图片保存名称
OUTPUT_IMG = "layer_sensitivity_heatmap.png"
# =========================================

def draw_plots():
    try:
        # 1. 读取数据
        df = pd.read_csv(FILE_PATH)
        
        # 确保数据是数字类型 (处理可能的 N/A)
        df['Eval_Loss'] = pd.to_numeric(df['Eval_Loss'], errors='coerce')
        df = df.dropna(subset=['Eval_Loss']) # 删掉没跑完的空行
        
        # 按 Layer_ID 排序
        df = df.sort_values('Layer_ID')

    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        print("请检查 csv 路径是否正确，或者文件内容是否为空。")
        return

    # 设置画图风格
    sns.set_theme(style="whitegrid")
    
    # 创建画布：上面是柱状图，下面是热力图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    # -------------------------------------------------------
    # 图 1: 柱状图 (Bar Chart) - 精确对比
    # -------------------------------------------------------
    # 颜色映射：Loss 越低，颜色越深/越显眼 (Reverse colormap)
    norm = plt.Normalize(df['Eval_Loss'].min(), df['Eval_Loss'].max())
    colors = plt.cm.viridis_r(norm(df['Eval_Loss'].values)) # _r 表示反转颜色，让低 Loss 变亮/深色

    bars = ax1.bar(
        df['Layer_ID'].astype(str), # X轴变字符串，防止自动补全中间的层
        df['Eval_Loss'], 
        color=colors,
        alpha=0.9,
        edgecolor='black'
    )
    
    # 标出最低点
    min_loss_idx = df['Eval_Loss'].idxmin()
    best_layer = df.loc[min_loss_idx, 'Layer_ID']
    min_loss = df.loc[min_loss_idx, 'Eval_Loss']
    
    ax1.axhline(y=min_loss, color='red', linestyle='--', alpha=0.5)
    ax1.text(0, min_loss*1.01, f'Best Loss: {min_loss:.4f} (Layer {best_layer})', color='red', fontweight='bold')

    ax1.set_title(f'Layer Sensitivity Analysis (Llama-3-8B on Math)\nLOWER Loss = BETTER Performance', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Eval Loss')
    ax1.set_xlabel('Layer ID')
    ax1.set_ylim(df['Eval_Loss'].min() * 0.95, df['Eval_Loss'].max() * 1.05) # 动态缩放Y轴，让差异更明显

    # -------------------------------------------------------
    # 图 2: 一维热力图 (Heatmap) - 视觉直观
    # -------------------------------------------------------
    # 准备热力图数据：需要转成 (1, N) 的矩阵
    heatmap_data = df['Eval_Loss'].values.reshape(1, -1)
    
    # 绘制
    # cmap="RdYlGn_r": 红(Red)代表高Loss(坏)，绿(Green)代表低Loss(好)。_r 是反转的意思。
    sns.heatmap(
        heatmap_data, 
        annot=True,              # 显示数值
        fmt=".3f",               # 小数点位数
        cmap="RdYlGn_r",         # 红色=高Loss(差)，绿色=低Loss(好)
        xticklabels=df['Layer_ID'], 
        yticklabels=["Eval Loss"],
        cbar_kws={'label': 'Loss Value'},
        ax=ax2
    )
    
    ax2.set_title('Sensitivity Heatmap (Green is Better)', fontsize=12)
    ax2.set_xlabel('Layer ID')

    # -------------------------------------------------------
    # 保存与显示
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=300)
    print(f"✅ 图片已保存为: {OUTPUT_IMG}")