import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Загружаем данные
df = pd.read_csv('datasets/heart_disease.csv')

#+ Базовая информация о данных

# Посмотрим на структуру данных
print("Размер данных (строки, столбцы):", df.shape)

# Посмотрим на названия всех столбцов
print("Названия столбцов:")
print(df.columns.tolist())

# Посмотрим на первые несколько строк данных
print("Первые 5 пациентов:")
print(df.head())

# Проверим пропущенные значения
print("Пропущенные значения в каждом столбце:")
print(df.isnull().sum())

#+ 2. Построим столбчатую диаграмму, сравнивающую количество здоровых и больных пациентов.

# Считаем количество здоровых и больных пациентов
healthy_count = len(df[df['target'] == 0])    # target == 0 → True если пациент здоров
sick_count = len(df[df['target'] == 1])     # target == 1 → True если пациент болен

print(f"Здоровых пациентов: {healthy_count}")
print(f"Больных пациентов: {sick_count}")

# Создаем фигуру для графика
plt.figure(figsize=(8, 6))   # рисунок(фигура) размером 8x6 дюймов

# Задаем названия категорий для столбцов
categories = ['Здоровые', 'Больные']

# Создаем столбчатую диаграмму, окрашиваем столбцы в разные цвета
plt.bar(categories, [healthy_count, sick_count],
        color=['lightgreen', 'salmon'])

# Добавляем подписи
plt.title('Количество здоровых и больных пациентов')
plt.xlabel('Группа пациентов')
plt.ylabel('Количество')

# Показываем диаграмму
# plt.show()

#+ 3. Диаграмма рассеяния: пульс / возраст

# Создаем новую фигуру для диаграммы рассеяния
plt.figure(figsize=(10, 6))

# Здоровые пациенты (target == 0)
healthy = df[df['target'] == 0]   #  выбираем только здоровых пациентов
plt.scatter(healthy['age'], healthy['thalach'],
           color='lightgreen', label='Здоров', alpha=0.7)

# Больные пациенты (target == 1)
sick = df[df['target'] == 1]    # выбираем только больных пациентов
plt.scatter(sick['age'], sick['thalach'],
           color='salmon', label='Болен', alpha=0.7)

# Добавляем подписи осей и заголовок
plt.xlabel('Возраст (age)')
plt.ylabel('Максимальный пульс (thalach)')
plt.title('Зависимость пульса от возраста')

# Добавляем легенду (объяснение цветов)
plt.legend()

# Добавляем сетку для удобства чтения
plt.grid(True, linestyle='--', alpha=0.7)

# plt.show()

#+ 4. Преобразование признака sex

# Создаем копию данных чтобы не испортить оригинал
df_processed = df.copy()

# Заменяем 0 на 'female', 1 на 'male'
df_processed['sex'] = df_processed['sex'].replace({0: 'female', 1: 'male'})

print("После преобразования:")
print(df_processed['sex'].value_counts())

# Применяем One-Hot Encoding к столбцу sex
sex_encoded = pd.get_dummies(df_processed['sex'], prefix='sex')

print("Закодированные данные:")
print(sex_encoded.head())

# Удаляем старый столбец sex и добавляем новые закодированные столбцы
df_processed = df_processed.drop('sex', axis=1)
df_processed = pd.concat([df_processed, sex_encoded], axis=1)

print("Таблица после One-Hot Encoding:")
print(df_processed[['sex_female', 'sex_male']].head())

#+ 5. Средний уровень холестерина

# Средний холестерин для здоровых пациентов (target = 0)
mean_chol_healthy = df_processed[df_processed['target'] == 0]['chol'].mean()

# Средний холестерин для больных пациентов (target = 1)
mean_chol_sick = df_processed[df_processed['target'] == 1]['chol'].mean()

print("Средний уровень холестерина")
print(f"Здоровые пациенты: {mean_chol_healthy:.2f}")
print(f"Больные пациенты: {mean_chol_sick:.2f}")

#+ 6. Нормализация признаков

# Выбираем столбцы для нормализации
columns_to_normalize = ['age', 'trestbps', 'chol', 'thalach']
features_to_normalize = df_processed[columns_to_normalize]

print("Данные для нормализации:")
print(features_to_normalize.head())

# Создаем "нормализатор"
scaler = MinMaxScaler()  # ← здесь формула!

# Обучаем его на наших данных (запоминает минимумы и максимумы)
scaler.fit(features_to_normalize)

# Применяем формулу ко всем данным
normalized_features = scaler.transform(features_to_normalize)

print("Минимумы которые запомнил нормализатор:")
print(scaler.data_min_)
print("Максимумы которые запомнил нормализатор:")
print(scaler.data_max_)

print("Данные после нормализации:")
print(normalized_features[:5])  # первые 5 строк

# Заменяем исходные столбцы на нормализованные
df_processed[columns_to_normalize] = normalized_features

print("Финальные данные (первые 5 строк):")
print(df_processed[columns_to_normalize].head())