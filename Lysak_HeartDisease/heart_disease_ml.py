# heart_disease_ml.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score,
                           precision_score, f1_score, recall_score,
                           roc_curve, auc)

# 1. Загрузка данных
df = pd.read_csv("datasets/heart_disease.csv")
print("Данные загружены. Размер:", df.shape)


# 2. Разделяем: что предсказываем (y) и по каким признакам (X)
y = df["target"]                    # Цель: есть болезнь или нет
X = df.drop("target", axis=1)       # Признаки: возраст, пол, холестерин и т.д.

print("Признаки (X):", X.shape)
print("Цель (y):", y.shape)


# 3. Разделяем данные на обучение и тест
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Обучающая выборка:", X_train.shape, y_train.shape)
print("Тестовая выборка:", X_test.shape, y_test.shape)


# 4. Наивный Байесовский классификатор
gnb = GaussianNB()                      # Создаём модель
gnb.fit(X_train, y_train)               # Обучаем модель на обучающих данных
y_pred = gnb.predict(X_test)            # Предсказываем на тестовых данных

print("Неправильно классифицировано:", (y_test != y_pred).sum(), "из", len(y_test))


# 5. Оценка модели Наивного Байеса
print("\n=== Наивный Байес ===")
print("Матрица ошибок:")
print(confusion_matrix(y_test, y_pred))

print("Accuracy (точность):", accuracy_score(y_test, y_pred))
print("Precision (точность предсказаний):", precision_score(y_test, y_pred))
print("Recall (полнота):", recall_score(y_test, y_pred))
print("F1-score (баланс):", f1_score(y_test, y_pred))

# 6. Логистическая регрессия
print("\n=== Логистическая регрессия ===")

logreg = LogisticRegression(random_state=42, max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_log = logreg.predict(X_test)

print("Матрица ошибок:")
print(confusion_matrix(y_test, y_pred_log))

print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Precision:", precision_score(y_test, y_pred_log))
print("Recall:", recall_score(y_test, y_pred_log))
print("F1-score:", f1_score(y_test, y_pred_log))


# 7. K-ближайших соседей - ищем лучшее k
print("\n=== K-ближайших соседей ===")

acc_series = []
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_knn)
    acc_series.append(accuracy)
    print(f"k={k:2d}: Accuracy={accuracy:.4f}")

best_k = acc_series.index(max(acc_series)) + 1
print(f"\nЛучшее k = {best_k} с Accuracy = {max(acc_series):.4f}")

# 8.1 График подбора k для KNN
plt.figure(figsize=(8, 6))
plt.plot(range(1, 21), acc_series, marker='o', linestyle='-', color='blue')
plt.xlabel('Количество соседей (k)')
plt.ylabel('Accuracy (Точность)')
plt.title('Зависимость точности от k для KNN')
plt.grid(True)
plt.xticks(range(1, 21))
plt.show()

# 8.2 ROC для Наивного Байеса
plt.figure(figsize=(8, 6))

# Получаем вероятности
y_probs_nb = gnb.predict_proba(X_test)[:, 1]

# Считаем ROC
fpr_nb, tpr_nb, _ = roc_curve(y_test, y_probs_nb)
roc_auc_nb = auc(fpr_nb, tpr_nb)

# Рисуем
plt.plot(fpr_nb, tpr_nb, color='darkorange', lw=2,
         label=f'ROC кривая (AUC = {roc_auc_nb:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
         label='Случайный классификатор')

# Настройки
plt.xlabel('False Positive Rate (1 - Специфичность)')
plt.ylabel('True Positive Rate (Чувствительность)')
plt.title('ROC-анализ: Наивный Байес')
plt.legend(loc="lower right")
plt.grid(True)

plt.show()

print(f"AUC для Наивного Байеса: {roc_auc_nb:.4f}")


# 8.3 ROC для Логистической регрессии
plt.figure(figsize=(8, 6))

# Получаем вероятности
y_probs_lr = logreg.predict_proba(X_test)[:, 1]

# Считаем ROC
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_probs_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

# Рисуем
plt.plot(fpr_lr, tpr_lr, color='darkorange', lw=2,
         label=f'ROC кривая (AUC = {roc_auc_lr:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
         label='Случайный классификатор')

# Настройки
plt.xlabel('False Positive Rate (1 - Специфичность)')
plt.ylabel('True Positive Rate (Чувствительность)')
plt.title('ROC-анализ: Логистическая регрессия')
plt.legend(loc="lower right")
plt.grid(True)

plt.show()

print(f"AUC для Логистической регрессии: {roc_auc_lr:.4f}")


# 8.4 ROC для K-ближайших соседей (k=11)
plt.figure(figsize=(8, 6))

# Создаем и обучаем KNN с лучшим k
knn_best = KNeighborsClassifier(n_neighbors=11)
knn_best.fit(X_train, y_train)

# Получаем вероятности
y_probs_knn = knn_best.predict_proba(X_test)[:, 1]

# Считаем ROC
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_probs_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)

# Рисуем
plt.plot(fpr_knn, tpr_knn, color='darkorange', lw=2,
         label=f'ROC кривая (AUC = {roc_auc_knn:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
         label='Случайный классификатор')

# Настройки
plt.xlabel('False Positive Rate (1 - Специфичность)')
plt.ylabel('True Positive Rate (Чувствительность)')
plt.title('ROC-анализ: K-ближайших соседей (k=11)')
plt.legend(loc="lower right")
plt.grid(True)

plt.show()

print(f"AUC для K-соседей (k=11): {roc_auc_knn:.4f}")

# Итог всех моделей:
# Логистическая регрессия: AUC 0.927 (лучшая!)
# Наивный Байес: AUC 0.894 (хорошая)
# K-соседи: AUC 0.807 (требует нормализации)