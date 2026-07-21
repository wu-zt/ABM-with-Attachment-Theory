import pandas as pd
import matplotlib.pyplot as plt

# 1. 读取新的 400 步数据文件
file_path = 'AT_disaster_duration_test-table.csv' # 👉 重点：这里换成你刚跑出来的新文件名！
df = pd.read_csv(file_path, skiprows=6)

# 2. 重命名列
cols = {
    'count turtles with [color = green]': 'Secure (Green)',
    'count turtles with [color = yellow]': 'Avoidant (Yellow)',
    'count turtles with [color = blue]': 'Anxious (Blue)',
    'count turtles with [color = red]': 'Fearful (Red)',
    '[step]': 'Step'
}
df = df.rename(columns=cols)

# 3. 计算均值和标准差
n_runs = df['[run number]'].nunique() # 自动获取重复次数
grouped = df.groupby('Step')
means = grouped.mean(numeric_only=True)
stds = grouped.std(numeric_only=True)

# 4. 画图
plt.figure(figsize=(12, 6))
colors_map = {
    'Secure (Green)': '#2ca02c',
    'Avoidant (Yellow)': '#d4b000',
    'Anxious (Blue)': '#1f77b4',
    'Fearful (Red)': '#d62728'
}

for col, color in colors_map.items():
    plt.plot(means.index, means[col], label=col, color=color, linewidth=2)
    plt.fill_between(means.index,
                     means[col] - stds[col],
                     means[col] + stds[col],
                     color=color, alpha=0.2)

# 5. 图表美化 👉 重点：标题改成 Epoch = 400
plt.title('Evolution of Attachment Styles over 10 Generations (Epoch = 400)', fontsize=14, pad=15)
plt.xlabel('Simulation Steps (401 steps = 1 Generation)', fontsize=12) # 注意 400 的代长是 401
plt.ylabel(f'Population Count (Mean ± Std, n={n_runs})', fontsize=12)

plt.legend(loc='upper right', frameon=True, shadow=True)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# 保存新图
plt.savefig('evolution_curve_10gens_epoch400.png', dpi=300)
print("图表已成功保存为 'evolution_curve_10gens_epoch400.png'")
plt.show()