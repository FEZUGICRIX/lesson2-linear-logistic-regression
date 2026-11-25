import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)
torch.manual_seed(42)

# Загрузка и базовая подготовка данных (Diabetes)
df = pd.read_csv("data/diabetes_regression.csv")
X_raw = df.drop(columns=['target']).select_dtypes(include=[np.number]).values
y_raw = df['target'].values.reshape(-1, 1)

X_base = StandardScaler().fit_transform(X_raw)
n_train = int(0.8 * len(X_base))

X_train = torch.tensor(X_base[:n_train], dtype=torch.float32)
y_train = torch.tensor(y_raw[:n_train], dtype=torch.float32)
X_test = torch.tensor(X_base[n_train:], dtype=torch.float32)
y_test = torch.tensor(y_raw[n_train:], dtype=torch.float32)


def run_experiment(opt_name="Adam", lr=1.0, batch_size=None, epochs=250):
    """Универсальная функция для одного эксперимента"""
    model = nn.Linear(X_train.shape[1], 1)
    criterion = nn.MSELoss()

    if opt_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif opt_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []
    n_samples = len(X_train)
    b_size = n_samples if batch_size is None else batch_size

    for _ in range(epochs):
        epoch_loss = 0.0
        steps = 0
        # Проходим по данным пачками (батчами) размера b_size
        for i in range(0, n_samples, b_size):
            bx = X_train[i:i + b_size]
            by = y_train[i:i + b_size]
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1
        losses.append(epoch_loss / steps)

    with torch.no_grad():
        test_mse = mean_squared_error(y_test.numpy(), model(X_test).numpy())
    return losses, test_mse


# ==============================================================================
# 3.1 Исследование гиперпараметров (Learning Rate, Batch Size, Оптимизаторы)
# ==============================================================================
print("=" * 65)
print("3.1 Эксперимент 1: Скорость обучения (Learning Rate) для Adam")
print("=" * 65)
for lr in [0.01, 0.1, 1.0]:
    losses, test_mse = run_experiment(opt_name="Adam", lr=lr, epochs=200)
    print(f"lr = {lr:<5} | Train MSE = {losses[-1]:<10.2f} | Test MSE = {test_mse:.2f}")

print("\n" + "=" * 65)
print("3.1 Эксперимент 2: Размер батча (Batch Size) для Adam (lr=0.2)")
print("=" * 65)
for bs in [16, 32, 64]:
    losses, test_mse = run_experiment(opt_name="Adam", lr=0.2, batch_size=bs, epochs=100)
    print(f"batch_size = {bs:<3} | Train MSE = {losses[-1]:<10.2f} | Test MSE = {test_mse:.2f}")

print("\n" + "=" * 65)
print("3.1 Эксперимент 3: Сравнение оптимизаторов (SGD, RMSprop, Adam)")
print("=" * 65)
opt_configs = [("SGD", 0.1), ("RMSprop", 0.8), ("Adam", 1.5)]
history = {}
for name, lr in opt_configs:
    losses, test_mse = run_experiment(opt_name=name, lr=lr, epochs=250)
    history[name] = losses
    print(f"{name:<10} (lr={lr}) | Train MSE = {losses[-1]:<10.2f} | Test MSE = {test_mse:.2f}")

plt.figure(figsize=(7, 4))
for name, _ in opt_configs:
    plt.plot(history[name], label=name)
plt.title('Сравнение сходимости оптимизаторов (SGD vs RMSprop vs Adam)')
plt.xlabel('Эпоха')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.savefig("plots/optimizers_comparison.png", dpi=150)
plt.close()


# ==============================================================================
# 3.2 Feature Engineering (Полиномы, Взаимодействия и Статистика)
# ==============================================================================
print("\n" + "=" * 65)
print("3.2 Feature Engineering: Создание новых признаков")
print("=" * 65)

# 1. Полиномы (x^2) и взаимодействия (x1 * x2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_base)

# 2. Статистики по строкам (среднее и дисперсия)
row_means = np.mean(X_base, axis=1, keepdims=True)
row_vars = np.var(X_base, axis=1, keepdims=True)

# 3. Объединяем и масштабируем
X_eng = np.hstack([X_poly, row_means, row_vars])
X_eng = StandardScaler().fit_transform(X_eng)

print(f"Признаков в базовой модели:          {X_base.shape[1]}")
print(f"Признаков после Feature Engineering: {X_eng.shape[1]}")

def train_eval(X_mat):
    Xt_train = torch.tensor(X_mat[:n_train], dtype=torch.float32)
    Xt_test = torch.tensor(X_mat[n_train:], dtype=torch.float32)

    model = nn.Linear(X_mat.shape[1], 1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1.0, weight_decay=0.01)

    for _ in range(250):
        optimizer.zero_grad()
        loss = criterion(model(Xt_train), y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = model(Xt_test).numpy()
    return mean_squared_error(y_test.numpy(), preds), r2_score(y_test.numpy(), preds)


mse_base, r2_base = train_eval(X_base)
mse_eng, r2_eng = train_eval(X_eng)

print(f"\nБазовая модель:               MSE = {mse_base:.2f} | R^2 = {r2_base:.4f}")
print(f"Модель с Feature Engineering: MSE = {mse_eng:.2f} | R^2 = {r2_eng:.4f}")

plt.figure(figsize=(6.5, 4))
bars = plt.bar(['Базовая модель', 'Feature Engineering'], [r2_base, r2_eng], color=['skyblue', 'coral'], width=0.45)
plt.ylabel('R^2 Score (чем выше, тем лучше)')
plt.title('Сравнение качества до и после Feature Engineering')
plt.ylim(0, 1.0)
for bar in bars:
    val = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.4f}", ha='center', fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("plots/feature_engineering_comparison.png", dpi=150)
plt.close()
print("Графики сохранены в plots/")
