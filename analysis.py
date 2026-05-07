import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')
os.makedirs('images', exist_ok=True)

# ===================== НАСТРОЙКА СТИЛЯ =====================
plt.rcParams.update({
    'figure.facecolor': '#0F1F3D',
    'axes.facecolor': '#0F1F3D',
    'axes.edgecolor': '#4A6080',
    'text.color': '#FFFFFF',
    'xtick.color': '#CCCCCC',
    'ytick.color': '#CCCCCC',
    'axes.labelcolor': '#CCCCCC',
    'axes.titlecolor': '#FFFFFF',
    'grid.color': '#1E3A5F',
    'grid.linestyle': '--',
    'grid.alpha': 0.5,
})
ACCENT = '#0A7EA4'
ACCENT2 = '#F4A233'

# ===================== ЗАГРУЗКА ДАННЫХ =====================
FILE = 'data/VK_Trafic.xlsx'
df = pd.read_excel(FILE, sheet_name='Лист1')

# Очистка: если есть столбцы с пробелами, убираем их
df.columns = df.columns.str.strip()

# Переводим TimeSpent в секунды, если он object
if df['TimeSpent'].dtype == 'object':
    def time_to_sec(t):
        try:
            parts = str(t).split(':')
            return int(parts[-1]) + int(parts[-2])*60 + int(parts[-3])*3600 if len(parts)==3 else np.nan
        except:
            return np.nan
    df['TimeSpentSec'] = df['TimeSpent'].apply(time_to_sec)
else:
    df['TimeSpentSec'] = df['TimeSpent']

# Заполняем пропуски в сегментах
df['Segments'] = df['Segments'].fillna('undefined')

print("Данные загружены")

# ===================== БЛОК РАСЧЁТОВ =====================
total_clicks = df['Clicks'].sum()
total_impressions = df['Impressions'].sum()
avg_position = df['Position'].mean()
avg_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
avg_bounce = df['BounceRate'].mean() * 100
avg_depth = df['ViewDepth'].mean()
avg_time = df['TimeSpentSec'].mean()

# Топ-5 страниц по кликам
top_pages = df.nlargest(10, 'Clicks')[['Clicks', 'Impressions', 'Position', 'Segments']]

# Распределение по сегментам
segment_counts = df['Segments'].value_counts()

print(f"Общее количество кликов: {total_clicks}")
print(f"Общее количество показов: {total_impressions}")
print(f"Средняя позиция: {avg_position:.2f}")
print(f"Средний CTR: {avg_ctr:.2f}%")
print(f"Средняя глубина просмотра: {avg_depth:.2f} стр.")
print(f"Среднее время на сайте: {avg_time:.0f} сек.")

# ===================== ГРАФИК 1: Распределение кликов =====================
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df['Clicks'], bins=20, color=ACCENT, edgecolor='white', alpha=0.8)
ax.axvline(df['Clicks'].median(), color=ACCENT2, linestyle='--', label=f"Медиана: {df['Clicks'].median():.0f}")
ax.set_title('Распределение количества кликов по страницам')
ax.set_xlabel('Клики')
ax.set_ylabel('Количество страниц')
ax.legend(facecolor='#1a2f4e', edgecolor='#4A6080')
ax.grid(axis='y', alpha=0.4)
fig.tight_layout()
plt.savefig('images/01_clicks_hist.png', dpi=150, bbox_inches='tight')
plt.close()

# ===================== ГРАФИК 2: Клики vs Позиция (scatter) =====================
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(df['Position'], df['Clicks'], c=df['Impressions'], cmap='coolwarm', s=60, edgecolor='white')
ax.set_title('Зависимость кликов от позиции')
ax.set_xlabel('Позиция в поиске')
ax.set_ylabel('Клики')
ax.grid(alpha=0.4)
fig.colorbar(sc, ax=ax, label='Показы')
fig.tight_layout()
plt.savefig('images/02_clicks_vs_position.png', dpi=150, bbox_inches='tight')
plt.close()

# ===================== ГРАФИК 3: Тепловая карта корреляции =====================
corr_columns = ['Clicks', 'Impressions', 'Position', 'BounceRate', 'ViewDepth',
                'TimeSpentSec', 'RobotsVisits', 'Mobility', 'Title Length',
                'Meta Description Length', 'H1 Length', 'Word Count',
                'Sentence Count', 'Folder Depth', 'Link Score', 'Inlinks',
                'Outlinks', 'Response Time']
corr = df[corr_columns].corr()

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.index)))
ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
ax.set_yticklabels(corr.index, fontsize=8)
ax.set_title('Корреляционная матрица SEO-метрик')
fig.colorbar(im, ax=ax, label='Коэффициент корреляции')
fig.tight_layout()
plt.savefig('images/03_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# ===================== ГРАФИК 4: Топ-10 страниц по кликам =====================
top10 = df.nlargest(10, 'Clicks')[['Clicks', 'Segments']].reset_index(drop=True)
top10.index = [f'Стр.{i+1}' for i in range(10)]

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.barh(top10.index[::-1], top10['Clicks'][::-1], color=ACCENT)
max_val = top10['Clicks'].max()
offset = max_val * 0.03
for bar, val in zip(bars, top10['Clicks'][::-1]):
    ax.text(val + offset, bar.get_y() + bar.get_height()/2, str(val), va='center', color='white', fontsize=9)
ax.set_xlim(right=ax.get_xlim()[1] * 1.12)
ax.set_title('Топ-10 страниц по кликам')
ax.grid(axis='x', alpha=0.4)
fig.tight_layout()
plt.savefig('images/04_top10_clicks.png', dpi=150, bbox_inches='tight')
plt.close()

# ===================== ГРАФИК 5: Распределение по сегментам =====================
seg_data = segment_counts.copy()
total_seg = seg_data.sum()
seg_pct = (seg_data / total_seg * 100).round(1)
seg_colors = ['#0A7EA4', '#F4A233', '#7BC8A4', '#E87E6B'][:len(seg_data)]

fig, ax = plt.subplots(figsize=(7, 5))
wedges, _ = ax.pie(seg_data.values, labels=None, colors=seg_colors,
                   startangle=90, wedgeprops=dict(edgecolor='#0F1F3D', linewidth=2))
legend_labels = [f'{idx}: {val} стр. ({pct}%)' for idx, val, pct in zip(seg_data.index, seg_data.values, seg_pct)]
ax.legend(wedges, legend_labels, title="Сегменты", loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1), facecolor='#1a2f4e', edgecolor='#4A6080',
          labelcolor='white', title_fontsize=11)
ax.set_title('Распределение страниц по сегментам')
fig.tight_layout()
plt.savefig('images/05_segment_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# ===================== ГРАФИК 6: KPI-карточки =====================
metrics = [
    ('Всего кликов', f'{total_clicks:,}'.replace(',', ' '), ACCENT),
    ('Всего показов', f'{total_impressions:,}'.replace(',', ' '), ACCENT2),
    ('Средняя позиция', f'{avg_position:.1f}', '#7BC8A4'),
    ('Средний CTR', f'{avg_ctr:.2f}%', '#F472B6'),
    ('Средняя глубина', f'{avg_depth:.2f} стр.', '#A78BFA'),
]
fig, axes = plt.subplots(1, 5, figsize=(16, 3))
for ax, (label, value, color) in zip(axes, metrics):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.add_patch(plt.Rectangle((0.05, 0.1), 0.9, 0.8, facecolor='#1a2f4e', edgecolor=color, linewidth=2))
    ax.text(0.5, 0.63, value, ha='center', va='center', fontsize=16, fontweight='bold', color=color)
    ax.text(0.5, 0.28, label, ha='center', va='center', fontsize=9, color='#AAAAAA')
fig.suptitle('Сводные показатели трафика VK', fontsize=14, y=1.05)
fig.tight_layout()
plt.savefig('images/06_kpi_summary.png', dpi=150, bbox_inches='tight')
plt.close()

# ===================== ГРАФИК 7: Boxplot глубины просмотра по сегментам =====================
fig, ax = plt.subplots(figsize=(10, 6))
df.boxplot(column='ViewDepth', by='Segments', ax=ax, patch_artist=True,
           boxprops=dict(facecolor=ACCENT, color='white'),
           whiskerprops=dict(color='white'),
           capprops=dict(color='white'),
           medianprops=dict(color=ACCENT2))
ax.set_title('Глубина просмотра по сегментам')
ax.set_xlabel('Сегмент')
ax.set_ylabel('Глубина просмотра')
ax.grid(axis='y', alpha=0.4)
fig.suptitle('')  # убираем автоматический suptitle
fig.tight_layout()
plt.savefig('images/07_depth_boxplot.png', dpi=150, bbox_inches='tight')
plt.close()

# ===================== ГРАФИК 8: Время на сайте vs Глубина просмотра =====================
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(df['ViewDepth'], df['TimeSpentSec'], c=df['BounceRate'], cmap='coolwarm',
                s=60, edgecolor='white')
ax.set_title('Зависимость времени на сайте от глубины просмотра')
ax.set_xlabel('Глубина просмотра (стр.)')
ax.set_ylabel('Время на сайте (сек.)')
ax.grid(alpha=0.4)
fig.colorbar(sc, ax=ax, label='Показатель отказов')
fig.tight_layout()
plt.savefig('images/08_time_vs_depth.png', dpi=150, bbox_inches='tight')
plt.close()

# ===================== ГРАФИК 9: Длина заголовка и клики =====================
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(df['Title Length'], df['Clicks'], c=df['Position'], cmap='coolwarm',
                s=60, edgecolor='white')
ax.set_title('Влияние длины заголовка на клики')
ax.set_xlabel('Длина Title')
ax.set_ylabel('Клики')
ax.grid(alpha=0.4)
fig.colorbar(sc, ax=ax, label='Позиция')
fig.tight_layout()
plt.savefig('images/09_title_length_clicks.png', dpi=150, bbox_inches='tight')
plt.close()

print("Все графики сохранены в папку images/")
