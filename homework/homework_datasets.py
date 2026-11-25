import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)
torch.manual_seed(42)

# ==============================================================================
# 2.1 Кастомный класс Dataset для работы с CSV файлами
# ==============================================================================
class CSVDataset(Dataset):
    def __init__(self, csv_path, target_col, is_classification=False):
        df = pd.read_csv(csv_path)

        # 1. Отделяем правильные ответы (y) от таблицы
        y_raw = df[target_col].values
        if is_classification:
            y_raw = LabelEncoder().fit_transform(y_raw)
        self.y = torch.tensor(y_raw, dtype=torch.float32).unsqueeze(1)

        # 2. Берем только признаки (X), удалив колонку с ответами
        X_df = df.drop(columns=[target_col])

        # 3. Переводим текстовые категории в 0 и 1 (One-Hot Encoding)
        cat_cols = X_df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        if cat_cols:
            X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=True)

        # 4. Приводим все числа к одному масштабу (StandardScaler)
        X_scaled = StandardScaler().fit_transform(X_df)
        self.X = torch.tensor(X_scaled, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==============================================================================
# 2.2 Эксперимент 1: Линейная регрессия на датасете Diabetes (CSV)
# ==============================================================================
print("=" * 60)
print("2.2 Эксперимент 1: Регрессия (Diabetes CSV)")
print("=" * 60)

reg_ds = CSVDataset("data/diabetes_regression.csv", target_col="target", is_classification=False)
n_train = int(0.8 * len(reg_ds))

X_train, y_train = reg_ds.X[:n_train], reg_ds.y[:n_train]
X_test, y_test = reg_ds.X[n_train:], reg_ds.y[n_train:]

model_reg = nn.Linear(reg_ds.X.shape[1], 1) # 11 параметров из табл
criterion_reg = nn.MSELoss()
optimizer_reg = optim.Adam(model_reg.parameters(), lr=1.0)

reg_losses = []
for epoch in range(250):
    optimizer_reg.zero_grad()
    loss = criterion_reg(model_reg(X_train), y_train)
    loss.backward()
    optimizer_reg.step()
    reg_losses.append(loss.item())

with torch.no_grad():
    preds = model_reg(X_test).numpy()
    targets = y_test.numpy()

print(f"Строк: {len(reg_ds)} | Признаков: {reg_ds.X.shape[1]}")
print(f"Test MSE: {mean_squared_error(targets, preds):.2f}")
print(f"Test R^2: {r2_score(targets, preds):.4f}")

plt.figure(figsize=(6, 3.5))
plt.plot(reg_losses, color='blue')
plt.title('Обучение линейной регрессии (Diabetes CSV)')
plt.xlabel('Эпоха')
plt.ylabel('MSE')
plt.grid(True)
plt.savefig("plots/dataset_regression_loss.png", dpi=150)
plt.close()


# ==============================================================================
# 2.2 Эксперимент 2: Логистическая регрессия на датасете Cancer (CSV)
# ==============================================================================
print("\n" + "=" * 60)
print("2.2 Эксперимент 2: Бинарная классификация (Cancer CSV)")
print("=" * 60)

clf_ds = CSVDataset("data/cancer_classification.csv", target_col="target", is_classification=True)
n_train_c = int(0.8 * len(clf_ds))

X_c_train, y_c_train = clf_ds.X[:n_train_c], clf_ds.y[:n_train_c]
X_c_test, y_c_test = clf_ds.X[n_train_c:], clf_ds.y[n_train_c:]

model_clf = nn.Linear(clf_ds.X.shape[1], 1)
criterion_clf = nn.BCEWithLogitsLoss()
optimizer_clf = optim.Adam(model_clf.parameters(), lr=0.1)

clf_losses = []
for epoch in range(150):
    optimizer_clf.zero_grad()
    logits = model_clf(X_c_train)
    loss = criterion_clf(logits, y_c_train)
    loss.backward()
    optimizer_clf.step()
    clf_losses.append(loss.item())

with torch.no_grad():
    probs = torch.sigmoid(model_clf(X_c_test)).numpy().ravel()
    preds_bin = (probs >= 0.5).astype(int)
    targets_c = y_c_test.numpy().ravel()

print(f"Строк: {len(clf_ds)} | Признаков: {clf_ds.X.shape[1]}")
print(f"Accuracy:  {accuracy_score(targets_c, preds_bin):.4f}")
print(f"Precision: {precision_score(targets_c, preds_bin):.4f}")
print(f"Recall:    {recall_score(targets_c, preds_bin):.4f}")
print(f"F1-score:  {f1_score(targets_c, preds_bin):.4f}")
print(f"ROC-AUC:   {roc_auc_score(targets_c, probs):.4f}")

plt.figure(figsize=(6, 3.5))
plt.plot(clf_losses, color='green')
plt.title('Обучение логистической регрессии (Cancer CSV)')
plt.xlabel('Эпоха')
plt.ylabel('BCE Loss')
plt.grid(True)
plt.savefig("plots/dataset_classification_loss.png", dpi=150)
plt.close()
print("\nГрафики сохранены в plots/")
