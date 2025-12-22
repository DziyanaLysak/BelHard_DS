# ФИНАЛЬНЫЙ ПРОЕКТ: Кластеризация модных товаров Myntra

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

#+ ШАГ 1: ЗАГРУЗКА ДАННЫХ

print("=" * 60)
print("ШАГ 1: ЗАГРУЗКА ДАННЫХ")
print("=" * 60)

def prepare_data():
    """Загружаем данные"""
    df = pd.read_csv("../data/styles.csv", on_bad_lines='skip')
    print(f"✅ Данные загружены: {df.shape[0]} строк, {df.shape[1]} колонок")
    return df

df = prepare_data()
print("Первые 3 строки:")
print(df.head(3))

#+ ШАГ 2: ИССЛЕДОВАТЕЛЬСКИЙ АНАЛИЗ (EDA)

print("\n" + "=" * 60)
print("ШАГ 2: ИССЛЕДОВАТЕЛЬСКИЙ АНАЛИЗ (EDA)")
print("=" * 60)

# 2.1 Проверка пропусков
print("1. Проверка пропусков в данных:")
print(df.isnull().sum())

# 2.2 Распределение по полу
print("\n2. Распределение по целевой аудитории:")
gender_counts = df['gender'].value_counts()
for gender, count in gender_counts.items():
    percentage = (count / len(df)) * 100
    print(f"   {gender}: {count} товаров ({percentage:.1f}%)")

# 2.3 График для всех категорий
plt.figure(figsize=(8, 5))
colors = ['blue', 'pink', 'gray', 'lightblue', 'lightpink']
bars = plt.bar(gender_counts.index, gender_counts.values, color=colors)

# Добавляем значения на столбцы
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom')

plt.title('Распределение товаров по целевой аудитории', fontsize=14)
plt.xlabel('Целевая аудитория')
plt.ylabel('Количество товаров')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gender_distribution.png', dpi=100)
# plt.show()

#+ ШАГ 3: АНАЛИЗ ТИПОВ ОДЕЖДЫ И ЦВЕТОВ

print("\n" + "=" * 60)
print("ШАГ 3: ТОЧНЫЙ АНАЛИЗ ДАТАСЕТА")
print("=" * 60)

# 3.1 Топ-5 типов одежды
print("1. ТОП-5 ТИПОВ ОДЕЖДЫ:")
top_articles = df['articleType'].value_counts().head(5)
total_items = len(df)

for i, (type_name, count) in enumerate(top_articles.items(), 1):
    percentage = (count / total_items) * 100
    print(f"   {i}. {type_name}: {count} товаров ({percentage:.1f}%)")

# 3.2 Топ-5 цветов
print("\n2. ТОП-5 ЦВЕТОВ:")
top_colors = df['baseColour'].value_counts().head(5)

for i, (color, count) in enumerate(top_colors.items(), 1):
    percentage = (count / total_items) * 100
    print(f"   {i}. {color}: {count} товаров ({percentage:.1f}%)")

# 3.3 График топ-5 типов одежды
plt.figure(figsize=(10, 5))
bars_articles = plt.barh(range(len(top_articles)), top_articles.values, color='#FF6B8B')  # Розовый

plt.yticks(range(len(top_articles)), top_articles.index, fontsize=11)
plt.title('Топ-5 типов одежды', fontsize=14, pad=15)
plt.xlabel('Количество товаров', fontsize=12)
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.2, axis='x')
plt.tight_layout()
plt.savefig('top_5_article_types.png', dpi=100, bbox_inches='tight')
# plt.show()

# 3.4 График топ-5 цветов
print("\n3. Визуализация топ-5 цветов:")

color_map = {
    'Black': 'black',
    'Blue': 'blue',
    'White': 'white',
    'Grey': 'grey',
    'Brown': 'brown'
}

bar_colors = []
edge_colors = []
for color_name in top_colors.index:
    bar_colors.append(color_map.get(color_name, '#CCCCCC'))
    # Для белого цвета делаем чёрную рамку
    if color_name == 'White':
        edge_colors.append('black')
    else:
        edge_colors.append(color_map.get(color_name, '#CCCCCC'))

plt.figure(figsize=(10, 5))
bars_colors = plt.bar(range(len(top_colors)), top_colors.values,
                     color=bar_colors, edgecolor=edge_colors, linewidth=2)

plt.xticks(range(len(top_colors)), top_colors.index, rotation=45, ha='right', fontsize=11)
plt.title('Топ-5 цветов)', fontsize=14, pad=15)
plt.xlabel('Цвет', fontsize=12)
plt.ylabel('Количество товаров', fontsize=12)
plt.grid(True, alpha=0.2, axis='y')

# Белый фон для всего графика
fig = plt.gcf()
fig.patch.set_facecolor('white')
ax = plt.gca()
ax.set_facecolor('white')

plt.tight_layout()
plt.savefig('top_5_colors.png', dpi=100, bbox_inches='tight', facecolor='white')
# plt.show()

print(f"\n✅ Всего в выборке: {total_items} товаров")

#+ ШАГ 4: АНАЛИЗ ПО СЕЗОНАМ"

print("\n" + "=" * 60)
print("ШАГ 4: АНАЛИЗ ПО СЕЗОНАМ")
print("=" * 60)

# 4.1 Заменяем Fall на Autumn
df['season'] = df['season'].replace({'Fall': 'Autumn'})

# 4.2 Распределение по сезонам
print("Распределение товаров по сезонам:")
season_counts = df['season'].value_counts()
total_items = len(df)

season_order = ['Summer', 'Autumn', 'Winter', 'Spring']
season_counts = season_counts.reindex(season_order).fillna(0)

for season, count in season_counts.items():
    percentage = (count / total_items) * 100
    print(f"   {season}: {count} товаров ({percentage:.1f}%)")

# 4.3 График распределения по сезонам с цветами
plt.figure(figsize=(8, 5))
colors = {'Summer': 'yellow', 'Autumn': 'orange', 'Winter': 'lightblue', 'Spring': 'green'}

bar_colors = [colors.get(season, 'gray') for season in season_counts.index]
bars = plt.bar(season_counts.index, season_counts.values, color=bar_colors, edgecolor='black')

plt.title('Распределение товаров по сезонам', fontsize=14, pad=15)
plt.xlabel('Сезон', fontsize=12)
plt.ylabel('Количество товаров', fontsize=12)
plt.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
plt.savefig('seasons_distribution.png', dpi=100, bbox_inches='tight')
# plt.show()

#+ ШАГ 5: ПОДГОТОВКА ДАННЫХ ДЛЯ КЛАСТЕРИЗАЦИИ

print("\n" + "=" * 60)
print("ШАГ 5: ПОДГОТОВКА ДАННЫХ ДЛЯ КЛАСТЕРИЗАЦИИ")
print("=" * 60)

# 5.1 Выбираем признаки для кластеризации
print("1. Выбор признаков для кластеризации:")
features_for_clustering = ['gender', 'masterCategory', 'articleType', 'baseColour', 'season', 'usage']
print(f"   Признаки: {', '.join(features_for_clustering)}")

# Проверяем, что все признаки есть в данных
missing_features = [f for f in features_for_clustering if f not in df.columns]
if missing_features:
    print(f"   ⚠️ Отсутствуют: {missing_features}")
    # Используем только те, что есть
    features_for_clustering = [f for f in features_for_clustering if f in df.columns]
    print(f"   Будем использовать: {', '.join(features_for_clustering)}")

# 5.2 Создаём DataFrame с выбранными признаками
df_cluster = df[features_for_clustering].copy()
print(f"\n2. Размер данных для кластеризации: {df_cluster.shape}")

# 5.3 Кодируем категориальные признаки (Label Encoding)
print("\n3. Кодирование категориальных признаков:")
label_encoders = {}

for column in df_cluster.columns:
    le = LabelEncoder()
    df_cluster[column] = le.fit_transform(df_cluster[column].astype(str))
    label_encoders[column] = le
    print(f"   {column}: {len(le.classes_)} уникальных значений")

# 5.4 Нормализация данных
print("\n4. Нормализация данных (min-max scaling):")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)

print(f"   Размер после нормализации: {X_scaled.shape}")
print(f"   Диапазон значений: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")

# 5.5 Сохраняем обработанные данные
df_processed = pd.DataFrame(X_scaled, columns=df_cluster.columns)
df_processed.to_csv('../data/processed_data.csv', index=False)  # ../data/
print("\n✅ Обработанные данные сохранены: '../data/processed_data.csv'")

print("   • Все категории закодированы числами")
print("   • Данные нормализованы")
print("   • Готовы для PCA и кластеризации")

#+ ШАГ 6: УМЕНЬШЕНИЕ РАЗМЕРНОСТИ (PCA)

print("\n" + "=" * 60)
print("ШАГ 6: УМЕНЬШЕНИЕ РАЗМЕРНОСТИ (PCA)")
print("=" * 60)

# 6.1 Применяем PCA для 2 компонент (как в примере с ирисами)
print("1. Применение PCA для 2 главных компонент:")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"   Размер до PCA: {X_scaled.shape}")
print(f"   Размер после PCA: {X_pca.shape}")

# 6.2 Анализ объяснённой дисперсии
print("\n2. Объяснённая дисперсия компонент:")
explained_variance = pca.explained_variance_ratio_
for i, variance in enumerate(explained_variance, 1):
    print(f"   Компонента {i}: {variance:.3f} ({variance*100:.1f}%)")

total_variance = explained_variance.sum()
print(f"   Суммарная объяснённая дисперсия: {total_variance:.3f} ({total_variance*100:.1f}%)")

# 6.3 Визуализация PCA
print("\n3. Визуализация данных в пространстве PCA:")
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, c='purple', s=10)

plt.xlabel(f'Первая главная компонента ({explained_variance[0]*100:.1f}% дисперсии)')
plt.ylabel(f'Вторая главная компонента ({explained_variance[1]*100:.1f}% дисперсии)')
plt.title('Данные в пространстве двух главных компонент (PCA)', fontsize=14, pad=15)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pca_visualization.png', dpi=100, bbox_inches='tight')
# plt.show()

print("\n✅ PCA применён успешно")
print(f"   • 2 компоненты сохраняют {total_variance*100:.1f}% дисперсии")
print("   • Данные готовы для кластеризации K-means и сети Кохонена")

#+ ШАГ 7: КЛАСТЕРИЗАЦИЯ K-MEANS (3 кластера)

print("\n" + "=" * 60)
print("ШАГ 7: КЛАСТЕРИЗАЦИЯ K-MEANS (3 кластера)")
print("=" * 60)

# 7.1 Применяем K-means с 3 кластерами
print("1. Применение алгоритма K-means:")
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)  # Используем нормализованные данные

print(f"   Количество кластеров: 3")
print(f"   Размеры кластеров:")
for cluster_num in range(3):
    cluster_size = (kmeans_labels == cluster_num).sum()
    percentage = (cluster_size / len(kmeans_labels)) * 100
    print(f"   Кластер {cluster_num}: {cluster_size} товаров ({percentage:.1f}%)")

# 7.2 Визуализация кластеров в пространстве PCA
print("\n2. Визуализация кластеров K-means:")
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green']

for cluster_num in range(3):
    cluster_points = X_pca[kmeans_labels == cluster_num]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
               alpha=0.6, s=20, c=colors[cluster_num],
               label=f'Кластер {cluster_num}')

# Центроиды в пространстве PCA
centroids_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
           marker='X', s=200, c='black', label='Центроиды')

plt.xlabel(f'Первая главная компонента ({explained_variance[0]*100:.1f}% дисперсии)')
plt.ylabel(f'Вторая главная компонента ({explained_variance[1]*100:.1f}% дисперсии)')
plt.title('Кластеризация K-means (3 кластера) в пространстве PCA', fontsize=14, pad=15)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('kmeans_clusters.png', dpi=100, bbox_inches='tight')
# plt.show()

print("\n✅ K-means кластеризация завершена")

#+ ШАГ 8: КЛАСТЕРИЗАЦИЯ СЕТЬЮ КОХОНЕНА

print("\n" + "=" * 60)
print("ШАГ 8: КЛАСТЕРИЗАЦИЯ СЕТЬЮ КОХОНЕНА")
print("=" * 60)

# 8.1 Подготовка данных
print("1. Подготовка данных для сети Кохонена:")
sample_size = 5000
np.random.seed(42)
sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)
X_sample = X_scaled[sample_indices]
print(f"   Для обучения взято {sample_size} случайных товаров")


# 8.2 Класс KohonenMap
class KohonenMap:
    def __init__(self, grid_size=3, input_dim=6, learning_rate=0.5, sigma=1.0):
        self.grid_size = grid_size
        self.weights = np.random.randn(grid_size, grid_size, input_dim) * 0.1
        self.lr = learning_rate
        self.sigma = sigma

    def train(self, data, epochs=50):
        for epoch in range(epochs):
            for x in data:
                winner = self.find_winner(x)
                self.update_weights(x, winner, epoch, epochs)

    def find_winner(self, x):
        distances = np.linalg.norm(self.weights - x, axis=2)
        return np.unravel_index(np.argmin(distances), distances.shape)

    def update_weights(self, x, winner, epoch, total_epochs):
        lr = self.lr * (1 - epoch / total_epochs)
        sigma = self.sigma * (1 - epoch / total_epochs)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                d = np.linalg.norm(np.array([i, j]) - np.array(winner))
                neighborhood = np.exp(-d ** 2 / (2 * sigma ** 2))
                self.weights[i, j] += lr * neighborhood * (x - self.weights[i, j])

    def predict(self, data):
        labels = []
        for x in data:
            winner = self.find_winner(x)
            cluster_id = winner[0] * self.grid_size + winner[1]
            labels.append(cluster_id)
        return np.array(labels)


# 8.3 Создание, обучение и предсказание
print("\n2. Создание и обучение сети Кохонена 3x3:")
som = KohonenMap(grid_size=3, input_dim=X_sample.shape[1])
som.train(X_sample, epochs=50)
som_labels_sample = som.predict(X_sample)
X_pca_sample = pca.transform(X_sample)

print("3. Распределение по кластерам сети Кохонена:")
unique_labels, counts = np.unique(som_labels_sample, return_counts=True)
for label, count in zip(unique_labels, counts):
    percentage = (count / sample_size) * 100
    print(f"   Кластер {label}: {count} товаров ({percentage:.1f}%)")

# 8.4 Градиентная визуализация
print("\n4. Градиентная визуализация карты Кохонена:")


def get_neuron_color(i, j):
    y_norm = i / 2
    x_norm = j / 2
    r = 0.9 - y_norm * 0.3
    g = 0.6 - x_norm * 0.3
    b = 0.1 + y_norm * 0.3
    return (r, g, b)


neuron_colors = {}
for i in range(3):
    for j in range(3):
        neuron_id = i * 3 + j
        neuron_colors[neuron_id] = get_neuron_color(i, j)

point_colors = [neuron_colors[label] for label in som_labels_sample]

plt.figure(figsize=(10, 6))
plt.scatter(X_pca_sample[:, 0], X_pca_sample[:, 1],
            alpha=0.6, s=15, c=point_colors)

plt.xlabel(f'Первая главная компонента ({explained_variance[0] * 100:.1f}% дисперсии)')
plt.ylabel(f'Вторая главная компонента ({explained_variance[1] * 100:.1f}% дисперсии)')
plt.title('Карта Кохонена: градиентная визуализация', fontsize=14, pad=15)
plt.grid(True, alpha=0.2)

plt.text(0.02, 0.98, 'Градиент:\n• Светлые → летние/светлые товары\n• Тёмные → зимние/тёмные товары',
         transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('som_gradient.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n✅ Сеть Кохонена обучена")
print("   • 9 кластеров (карта 3x3)")
print("   • Градиент: от светлого к тёмному")
print("   • Выборка: 5,000 товаров")



