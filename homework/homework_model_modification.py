import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import make_classification, make_regression

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
torch.manual_seed(42)

# ==============================================================================
# 1.1 Линейная регрессия (L1/L2 регуляризация + Early Stopping)
# ==============================================================================
print("=" * 60)
print("1.1 Линейная регрессия: L1/L2 и Early Stopping")
print("=" * 60)

# 1. Данные: 300 примеров, делим на 240 (учеба) и 60 (проверка)
X_raw, y_raw = make_regression(n_samples=300, n_features=5, noise=15.0, random_state=42)
X = torch.tensor(X_raw, dtype=torch.float32)
y = torch.tensor(y_raw, dtype=torch.float32).unsqueeze(1)

X_train, y_train = X[:240], y[:240]
X_val, y_val = X[240:], y[240:]

# 2. Модель, ошибка MSE и оптимизатор
model_reg = nn.Linear(5, 1)
criterion_mse = nn.MSELoss()
optimizer = optim.Adam(model_reg.parameters(), lr=1.0)

lambda_l1 = 0.01  # коэффициент L1 штрафа
lambda_l2 = 0.02  # коэффициент L2 штрафа

patience = 8 # сколько эпох подряд мы готовы ждать улучшения ошибки, прежде чем нажать стоп
best_val_loss = float('inf')
patience_counter = 0
train_losses, val_losses = [], []

# 3. Цикл обучения
for epoch in range(1, 301):
    # Учеба на Train
    optimizer.zero_grad()
    y_pred = model_reg(X_train)
    mse_loss = criterion_mse(y_pred, y_train)

    # Штрафы L1 (модули) и L2 (квадраты)
    l1 = lambda_l1 * sum(p.abs().sum() for p in model_reg.parameters())
    l2 = lambda_l2 * sum((p ** 2).sum() for p in model_reg.parameters())
    total_loss = mse_loss + l1 + l2

    total_loss.backward()
    optimizer.step()
    train_losses.append(mse_loss.item())

    # Проверка на Val
    with torch.no_grad():
        val_loss = criterion_mse(model_reg(X_val), y_val).item()
    val_losses.append(val_loss)

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model_reg.state_dict(), "models/best_linreg_model.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"-> Early stopping на эпохе {epoch}! Лучший Val MSE: {best_val_loss:.2f}")
            break

# Загружаем лучшую модель и рисуем график
model_reg.load_state_dict(torch.load("models/best_linreg_model.pth"))
plt.figure(figsize=(7, 4))
plt.plot(train_losses, label='Train MSE')
plt.plot(val_losses, label='Val MSE')
plt.title('Линейная регрессия (L1/L2 + Early Stopping)')
plt.xlabel('Эпоха')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.savefig("plots/linreg_early_stopping.png", dpi=150)
plt.close()


# ==============================================================================
# 1.2 Многоклассовая логистическая регрессия и метрики
# ==============================================================================
print("\n" + "=" * 60)
print("1.2 Многоклассовая классификация (3 класса)")
print("=" * 60)

# 1. Данные: 500 примеров, делим на 400 (учеба) и 100 (тест)
X_c, y_c = make_classification(n_samples=500, n_features=6, n_informative=4, n_classes=3, random_state=42)
X_c = torch.tensor(X_c, dtype=torch.float32)
y_c = torch.tensor(y_c, dtype=torch.long)

X_c_train, y_c_train = X_c[:400], y_c[:400]
X_c_test, y_c_test = X_c[400:], y_c[400:]

# 2. Модель на 3 класса, ошибка CrossEntropy и оптимизатор
model_clf = nn.Linear(6, 3)
criterion_ce = nn.CrossEntropyLoss()
optimizer_clf = optim.Adam(model_clf.parameters(), lr=0.05)

# 3. Обучение
for epoch in range(100):
    optimizer_clf.zero_grad()
    logits = model_clf(X_c_train)
    loss = criterion_ce(logits, y_c_train)
    loss.backward()
    optimizer_clf.step()

torch.save(model_clf.state_dict(), "models/multiclass_logreg_model.pth")

# 4. Проверка на тесте и расчёт метрик
with torch.no_grad():
    logits_test = model_clf(X_c_test)
    probs = torch.softmax(logits_test, dim=1).numpy()  # вероятности для 3 классов
    preds = probs.argmax(axis=1)                       # выбранный класс (где вероятность выше)
    targets = y_c_test.numpy()

# если модель назвала объект классом Х, как часто она не врёт
print(f"Precision: {precision_score(targets, preds, average='macro'):.4f}") 
# какую долю всех реальных объектов класса Х модель нашла и не пропустила
print(f"Recall:    {recall_score(targets, preds, average='macro'):.4f}")  
#среднее гармоническое между Precision и Recall
print(f"F1-score:  {f1_score(targets, preds, average='macro'):.4f}")
# насколько хорошо модель в целом разделяет классы (0.5 — угадывает как монетка, 1.0 — идеал).
print(f"ROC-AUC:   {roc_auc_score(targets, probs, multi_class='ovr'):.4f}")

# 5. Матрица ошибок (Confusion Matrix)
cm = confusion_matrix(targets, preds)
ConfusionMatrixDisplay(cm, display_labels=["Класс 0", "Класс 1", "Класс 2"]).plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("plots/confusion_matrix.png", dpi=150)
plt.close()
print("Графики сохранены в plots/, модели в models/.")
