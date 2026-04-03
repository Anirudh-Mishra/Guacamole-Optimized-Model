import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'Helvetica'
matplotlib.rcParams['font.size'] = 10

BLUE = '#0f3460'
RED = '#e94560'
TEAL = '#00b4d8'
GRAY = '#94a3b8'
GREEN = '#16a34a'
AMBER = '#d97706'

import os
os.makedirs("charts", exist_ok=True)

# ═══════════════════════════════════════════
# 1. Vicuna: Our Model vs ChatGPT
# ═══════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))
categories = ['Common\nsense', 'Knowledge', 'Counter\nfactual', 'Generic', 'Coding', 'Writing', 'Roleplay', 'Fermi', 'Math']
ours =    [7.30, 7.40, 6.60, 7.30, 7.00, 6.60, 6.30, 5.40, 5.00]
chatgpt = [7.50, 7.70, 6.50, 8.00, 7.86, 7.80, 7.60, 7.30, 7.00]

x = np.arange(len(categories))
w = 0.35
bars1 = ax.bar(x - w/2, ours, w, label='Our Model (Guanaco-8B)', color=BLUE, zorder=3, edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + w/2, chatgpt, w, label='ChatGPT (GPT-3.5)', color=RED, zorder=3, edgecolor='white', linewidth=0.5)

ax.set_ylabel('Score (1-10)')
ax.set_title('Vicuna Benchmark: Our Model vs ChatGPT\n80 prompts · GPT-4o-mini as judge · Our model = 88.8% of ChatGPT (Paper: 87.0%)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=9)
ax.set_ylim(0, 10)
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add ratio labels
for i, (o, c) in enumerate(zip(ours, chatgpt)):
    ratio = o / c * 100
    color = GREEN if ratio >= 90 else (AMBER if ratio >= 80 else RED)
    ax.text(i, max(o, c) + 0.2, f'{ratio:.0f}%', ha='center', fontsize=8, color=color, fontweight='bold')

plt.tight_layout()
plt.savefig('charts/01_vicuna_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ 01_vicuna_comparison.png")

# ═══════════════════════════════════════════
# 2. Vicuna Radar Chart
# ═══════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
categories_radar = ['Common-sense', 'Knowledge', 'Counterfactual', 'Generic', 'Coding', 'Writing', 'Roleplay', 'Fermi', 'Math']
ours_radar = [7.30, 7.40, 6.60, 7.30, 7.00, 6.60, 6.30, 5.40, 5.00]
chatgpt_radar = [7.50, 7.70, 6.50, 8.00, 7.86, 7.80, 7.60, 7.30, 7.00]

angles = np.linspace(0, 2 * np.pi, len(categories_radar), endpoint=False).tolist()
ours_radar += ours_radar[:1]
chatgpt_radar += chatgpt_radar[:1]
angles += angles[:1]

ax.plot(angles, ours_radar, 'o-', linewidth=2, label='Our Model', color=BLUE, markersize=5)
ax.fill(angles, ours_radar, alpha=0.15, color=BLUE)
ax.plot(angles, chatgpt_radar, 'o-', linewidth=2, label='ChatGPT', color=RED, markersize=5)
ax.fill(angles, chatgpt_radar, alpha=0.1, color=RED)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories_radar, fontsize=9)
ax.set_ylim(0, 10)
ax.set_title('Capability Profile: Our Model vs ChatGPT', fontsize=13, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1), fontsize=10)

plt.tight_layout()
plt.savefig('charts/02_vicuna_radar.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ 02_vicuna_radar.png")

# ═══════════════════════════════════════════
# 3. MMLU Categories
# ═══════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
mmlu_cats = ['Social Sciences', 'Other', 'Humanities', 'STEM']
mmlu_scores = [69.81, 67.14, 54.62, 51.25]
colors_mmlu = [GREEN if s >= 60 else AMBER for s in mmlu_scores]

bars = ax.barh(mmlu_cats, mmlu_scores, color=colors_mmlu, edgecolor='white', linewidth=0.5, height=0.6, zorder=3)
ax.axvline(x=66, color=RED, linestyle='--', linewidth=1.5, alpha=0.7, label='Untuned baseline (~66%)', zorder=2)
ax.axvline(x=59.96, color=BLUE, linestyle='-', linewidth=2, alpha=0.5, label='Overall: 59.96%', zorder=2)

for bar, score in zip(bars, mmlu_scores):
    ax.text(score + 0.5, bar.get_y() + bar.get_height()/2, f'{score:.1f}%', va='center', fontsize=11, fontweight='bold')

ax.set_xlim(0, 85)
ax.set_xlabel('Accuracy (%)')
ax.set_title('MMLU 5-Shot by Category\nOverall: 59.96% (Untuned baseline: ~66%)', fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(axis='x', alpha=0.3, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('charts/03_mmlu_categories.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ 03_mmlu_categories.png")

# ═══════════════════════════════════════════
# 4. MMLU Top 10 vs Bottom 10
# ═══════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

top_subjects = ['Sociology', 'Management', 'Marketing', 'HS Gov\'t & Politics', 'World Religions',
                'HS Psychology', 'HS US History', 'Int\'l Law', 'Miscellaneous', 'US Foreign Policy']
top_scores = [86.57, 85.44, 82.91, 81.35, 80.12, 80.55, 79.41, 79.34, 79.31, 78.00]

bottom_subjects = ['College Math', 'Abstract Algebra', 'HS Mathematics', 'Global Facts', 'Moral Scenarios',
                   'Formal Logic', 'HS Physics', 'Prof. Accounting', 'College Physics', 'Prof. Law']
bottom_scores = [30.00, 31.00, 33.33, 34.00, 35.75, 37.30, 39.07, 41.13, 42.16, 42.24]

ax1.barh(top_subjects, top_scores, color=GREEN, edgecolor='white', linewidth=0.5, height=0.7)
for i, s in enumerate(top_scores):
    ax1.text(s + 0.5, i, f'{s:.1f}%', va='center', fontsize=9)
ax1.set_xlim(0, 100)
ax1.set_title('Top 10 Subjects', fontsize=12, fontweight='bold', color=GREEN)
ax1.grid(axis='x', alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.invert_yaxis()

ax2.barh(bottom_subjects, bottom_scores, color=RED, edgecolor='white', linewidth=0.5, height=0.7)
for i, s in enumerate(bottom_scores):
    ax2.text(s + 0.5, i, f'{s:.1f}%', va='center', fontsize=9)
ax2.set_xlim(0, 100)
ax2.set_title('Bottom 10 Subjects', fontsize=12, fontweight='bold', color=RED)
ax2.grid(axis='x', alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.invert_yaxis()

fig.suptitle('MMLU 5-Shot: Best vs Worst Subjects (out of 57)', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('charts/04_mmlu_best_worst.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ 04_mmlu_best_worst.png")

# ═══════════════════════════════════════════
# 5. Training Loss Curve
# ═══════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))
steps = [25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500,
         525,550,575,600,625,650,675,700,725,750,775,800,825,850,875,900,925,950,975,1000,
         1025,1050,1075,1100,1125,1150,1175,1200,1225,1250,1275,1300,1325,1350,1375,1400,
         1425,1450,1475,1500,1525,1550,1575,1600,1625,1650,1675,1700,1725,1750,1775,1800,
         1825,1850,1875]
losses = [1.2173,1.4871,1.2065,1.5109,1.2315,1.5078,1.2236,1.5484,1.161,1.5464,1.1994,
          1.5191,1.2412,1.4925,1.2235,1.524,1.1814,1.5076,1.184,1.4678,1.192,1.5549,
          1.2133,1.49,1.2557,1.1881,1.2457,1.153,1.244,1.1604,1.2069,1.1287,1.179,1.1547,
          1.2739,1.1072,1.2084,1.1874,1.2456,1.1485,1.225,1.1738,1.2628,1.1823,1.2161,
          1.1517,1.2135,1.1382,1.2086,1.0476,0.9344,0.9002,0.9317,0.9504,0.9531,0.9408,
          0.9352,0.9302,0.9448,0.9697,1.008,0.9433,0.9211,0.9241,0.9482,0.9774,0.9816,
          0.9393,0.9737,0.9646,0.9252,0.9145,0.9899,0.9125,0.7524]

ax.plot(steps, losses, color=BLUE, linewidth=1.2, alpha=0.9)

# Shade epochs
ax.axvspan(0, 625, alpha=0.05, color='blue', label='Epoch 1')
ax.axvspan(625, 1250, alpha=0.05, color='green', label='Epoch 2')
ax.axvspan(1250, 1875, alpha=0.05, color='red', label='Epoch 3')

# Epoch boundary lines
ax.axvline(x=625, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
ax.axvline(x=1250, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

# Annotations
ax.annotate('Epoch 3 begins\n(sharp drop)', xy=(1250, 1.05), xytext=(1050, 1.45),
            fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'), color='gray')
ax.annotate(f'Final: {losses[-1]:.4f}', xy=(1875, losses[-1]), xytext=(1750, 0.65),
            fontsize=10, fontweight='bold', color=BLUE,
            arrowprops=dict(arrowstyle='->', color=BLUE))

ax.set_xlabel('Training Step')
ax.set_ylabel('Training Loss')
ax.set_title('Training Loss Curve (1,875 steps)\nOscillation from group_by_length=True is expected behavior', fontsize=12, fontweight='bold')
ax.set_ylim(0.6, 1.65)
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('charts/05_training_loss.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ 05_training_loss.png")

# ═══════════════════════════════════════════
# 6. Paper Comparison
# ═══════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Vicuna comparison
models_v = ['Paper:\nGuanaco-7B', 'Ours:\nGuanaco-8B']
vicuna_scores = [87.0, 88.8]
bars = ax1.bar(models_v, vicuna_scores, color=[TEAL, BLUE], width=0.5, edgecolor='white', linewidth=0.5, zorder=3)
for bar, score in zip(bars, vicuna_scores):
    ax1.text(bar.get_x() + bar.get_width()/2, score + 0.3, f'{score}%', ha='center', fontsize=12, fontweight='bold')
ax1.set_ylim(80, 95)
ax1.set_ylabel('% of ChatGPT Performance')
ax1.set_title('Vicuna Benchmark', fontsize=12, fontweight='bold')
ax1.axhline(y=100, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax1.grid(axis='y', alpha=0.3, zorder=0)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# MMLU comparison
models_m = ['Paper Base\n(LLaMA-1-7B)', 'Paper\nGuanaco-7B', 'Our Base\n(LLaMA-3-8B)', 'Ours\nGuanaco-8B']
mmlu_scores_comp = [35.1, 36.6, 66.0, 59.96]
bar_colors = [GRAY, TEAL, GRAY, BLUE]
bars = ax2.bar(models_m, mmlu_scores_comp, color=bar_colors, width=0.6, edgecolor='white', linewidth=0.5, zorder=3)
for bar, score in zip(bars, mmlu_scores_comp):
    ax2.text(bar.get_x() + bar.get_width()/2, score + 1, f'{score}%', ha='center', fontsize=10, fontweight='bold')

# Draw arrows showing delta
ax2.annotate('', xy=(1, 36.6), xytext=(0, 35.1), arrowprops=dict(arrowstyle='->', color=GREEN, lw=2))
ax2.text(0.5, 33, '+1.5pp', ha='center', fontsize=9, color=GREEN, fontweight='bold')
ax2.annotate('', xy=(3, 59.96), xytext=(2, 66.0), arrowprops=dict(arrowstyle='->', color=RED, lw=2))
ax2.text(2.5, 68, '-6.0pp', ha='center', fontsize=9, color=RED, fontweight='bold')

ax2.set_ylim(0, 80)
ax2.set_ylabel('MMLU 5-shot Accuracy (%)')
ax2.set_title('MMLU 5-Shot', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3, zorder=0)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig.suptitle('Comparison with QLoRA Paper (Dettmers et al., 2023)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('charts/06_paper_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ 06_paper_comparison.png")

# ═══════════════════════════════════════════
# 7. Memory Footprint
# ═══════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Pie chart - Edge deployment
edge_sizes = [4.2, 0.64, 0.22, 18.94]
edge_labels = ['Base Model\n4.2 GB', 'Adapter\n0.64 GB', 'Runtime\n0.22 GB', 'Free\n18.94 GB']
edge_colors = [BLUE, TEAL, AMBER, '#e8f0fe']
explode = (0.03, 0.03, 0.03, 0)
wedges, texts, autotexts = ax1.pie(edge_sizes, labels=edge_labels, colors=edge_colors, explode=explode,
                                     autopct='%1.0f%%', pctdistance=0.75, startangle=90,
                                     textprops={'fontsize': 9})
for autotext in autotexts:
    autotext.set_fontsize(8)
ax1.set_title('Apple M4: Memory Usage\n(24 GB unified memory)', fontsize=12, fontweight='bold')

# Bar chart - Cloud vs Edge
components = ['Base Model', 'Adapter', 'Runtime\nOverhead', 'Total Peak']
cloud_vals = [5.0, 0.65, 7.85, 13.5]
edge_vals = [4.2, 0.64, 0.22, 5.06]

x = np.arange(len(components))
w = 0.35
ax2.bar(x - w/2, cloud_vals, w, label='AWS A10G (VRAM)', color=BLUE, zorder=3, edgecolor='white')
ax2.bar(x + w/2, edge_vals, w, label='Apple M4 (Unified)', color=GREEN, zorder=3, edgecolor='white')

for i, (c, e) in enumerate(zip(cloud_vals, edge_vals)):
    ax2.text(i - w/2, c + 0.2, f'{c}', ha='center', fontsize=8, fontweight='bold', color=BLUE)
    ax2.text(i + w/2, e + 0.2, f'{e}', ha='center', fontsize=8, fontweight='bold', color=GREEN)

ax2.axhline(y=23.7, color=BLUE, linestyle='--', linewidth=0.8, alpha=0.4, label='A10G VRAM limit (23.7 GB)')
ax2.axhline(y=24.0, color=GREEN, linestyle='--', linewidth=0.8, alpha=0.4, label='M4 memory (24 GB)')

ax2.set_ylabel('GB')
ax2.set_xticks(x)
ax2.set_xticklabels(components, fontsize=9)
ax2.set_title('Memory Comparison:\nCloud Training vs Edge Inference', fontsize=12, fontweight='bold')
ax2.legend(loc='upper left', fontsize=8)
ax2.grid(axis='y', alpha=0.3, zorder=0)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('charts/07_memory_footprint.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ 07_memory_footprint.png")

print("\n═══════════════════════════════════════")
print("All 7 charts saved to ./charts/")
print("═══════════════════════════════════════")
