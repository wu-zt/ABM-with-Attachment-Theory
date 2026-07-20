import pandas as pd
import matplotlib.pyplot as plt

# 1. 读取数据（跳过前6行 NetLogo 表头）
file_path = 'AT disaster_duration_test-table.csv'
df = pd.read_csv(file_path, skiprows=6)

# 2. 重命名列，让代码更好看
cols = {
    'count turtles with [color = green]': 'Secure (Green)',
    'count turtles with [color = yellow]': 'Avoidant (Yellow)',
    'count turtles with [color = blue]': 'Anxious (Blue)',
    'count turtles with [color = red]': 'Fearful (Red)',
    '[step]': 'Step'
}
df = df.rename(columns=cols)

# 3. 核心统计：按 Step 分组，计算 5 次重复实验的【平均值】和【标准差】
# numeric_only=True 防止把 run number 等非数值列算错
grouped = df.groupby('Step')
means = grouped.mean(numeric_only=True)
stds = grouped.std(numeric_only=True)

# 4. 开始画图
plt.figure(figsize=(12, 6))

# 定义颜色映射 (为了在白底上看得清，把黄色稍微调深一点)
colors_map = {
    'Secure (Green)': '#2ca02c',  # 经典绿
    'Avoidant (Yellow)': '#d4b000',  # 深黄色
    'Anxious (Blue)': '#1f77b4',  # 经典蓝
    'Fearful (Red)': '#d62728'  # 经典红
}

# 循环绘制四根曲线
for col, color in colors_map.items():
    # 绘制平均值折线
    plt.plot(means.index, means[col], label=col, color=color, linewidth=2)

    # 绘制标准差阴影（展示 5 次实验的波动范围）
    plt.fill_between(means.index,
                     means[col] - stds[col],
                     means[col] + stds[col],
                     color=color, alpha=0.2)  # alpha=0.2 是阴影的透明度

# 5. 图表美化（加上标题、标签、网格）
plt.title('Evolution of Attachment Styles over 30 Generations (Epoch = 100)', fontsize=14, pad=15)
plt.xlabel('Simulation Steps (101 steps = 1 Generation)', fontsize=12)
plt.ylabel('Population Count (Mean ± Std, n=5)', fontsize=12)

# 设置图例位置和边框
plt.legend(loc='upper right', frameon=True, shadow=True)

# 添加背景网格线，方便对齐看数据
plt.grid(True, linestyle='--', alpha=0.6)

# 紧凑布局
plt.tight_layout()

# 6. 保存高清图片并显示
plt.savefig('evolution_curve_30gens.png', dpi=300)
print("图表已成功保存为 'evolution_curve_30gens.png'")

# 弹出窗口展示
plt.show()