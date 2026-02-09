import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ========== 1. 数据加载与预处理 ==========
train_path = '/home/huazhao/DL-HW/ML-2025/mini-aide/input/train.csv'
test_path = '/home/huazhao/DL-HW/ML-2025/mini-aide/input/test.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# 提取目标列并确保列名完全一致
target_col = 'tested_positive_day3'
y = train_df[target_col].values

# 特征列：排除id和目标列
feature_cols = [col for col in train_df.columns if col not in ['id', target_col]]
X_train_raw = train_df[feature_cols].values

# 测试集特征（注意：测试集没有目标列）
X_test_raw = test_df[feature_cols].values

print(f"训练集形状: {X_train_raw.shape}, 测试集形状: {X_test_raw.shape}")

# ========== 2. 时间序列划分验证集 ==========
# 取最后20%作为验证集，不shuffle
val_size = int(0.2 * len(X_train_raw))
X_train_split = X_train_raw[:-val_size]
y_train_split = y[:-val_size]
X_val_split = X_train_raw[-val_size:]
y_val_split = y[-val_size:]

print(f"训练集分割后: {X_train_split.shape}, 验证集: {X_val_split.shape}")

# ========== 3. 特征工程 ==========
# 3.1 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_split)
X_val_scaled = scaler.transform(X_val_split)
X_test_scaled = scaler.transform(X_test_raw)

# 3.2 特征选择 (k=15)
selector = SelectKBest(score_func=f_regression, k=15)
X_train_selected = selector.fit_transform(X_train_scaled, y_train_split)
X_val_selected = selector.transform(X_val_scaled)
X_test_selected = selector.transform(X_test_scaled)

print(f"特征选择后维度: {X_train_selected.shape}")

# 3.3 简单交互特征（示例：cli_day1 * wearing_mask_7d_day1）
# 这里我们直接在标准化后的数据上构造，但要注意特征选择可能已改变列顺序
# 为简化，我们跳过这一步，让模型自行学习特征关系

# ========== 4. 转换为PyTorch张量 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

X_train_tensor = torch.FloatTensor(X_train_selected).to(device)
y_train_tensor = torch.FloatTensor(y_train_split).to(device).view(-1, 1)
X_val_tensor = torch.FloatTensor(X_val_selected).to(device)
y_val_tensor = torch.FloatTensor(y_val_split).to(device).view(-1, 1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ========== 5. 定义DNN模型 ==========
class COVIDPredictor(nn.Module):
    def __init__(self, input_dim):
        super(COVIDPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)

model = COVIDPredictor(X_train_selected.shape[1]).to(device)

# ========== 6. 训练设置 ==========
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

# ========== 7. 训练循环 ==========
epochs = 100
best_val_loss = float('inf')

for epoch in range(epochs):
    # 训练阶段
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # 验证阶段
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val_tensor)
        val_loss = criterion(val_predictions, y_val_tensor)
    
    # 学习率调度
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss:.4f}')

# ========== 8. 在验证集上评估 ==========
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
with torch.no_grad():
    val_predictions = model(X_val_tensor)
    val_mse = criterion(val_predictions, y_val_tensor).item()
    val_rmse = np.sqrt(val_mse)

# 计算Score
score = 1.0 / (1.0 + val_rmse)
print(f'Score: {score:.6f}')
print(f'Score = (1.0 / (1.0 + RMSE)) = {score:.6f}')

# ========== 9. 对测试集进行预测 ==========
X_test_tensor = torch.FloatTensor(X_test_selected).to(device)
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor).cpu().numpy()

# ========== 10. 生成提交文件 ==========
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'tested_positive': test_predictions.flatten()
})
submission_df.to_csv('submission.csv', index=False)
print(f"提交文件已保存，形状: {submission_df.shape}")