import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. 数据加载 ====================
train_df = pd.read_csv('/home/huazhao/DL-HW/ML-2025/mini-aide/input/train.csv')
test_df = pd.read_csv('/home/huazhao/DL-HW/ML-2025/mini-aide/input/test.csv')

# 目标列名
TARGET = 'tested_positive_day3'

# ==================== 2. 数据准备 ====================
# 准备训练数据
X_train_all = train_df.drop(columns=['id', TARGET])
y_train_all = train_df[TARGET]

# 准备测试数据
X_test = test_df.drop(columns=['id'])

# 确保训练集和测试集的特征列顺序一致
common_cols = [col for col in X_train_all.columns if col in X_test.columns]
X_train_all = X_train_all[common_cols]
X_test = X_test[common_cols]

# ==================== 3. 时间序列划分验证集 ====================
# 取最后20%作为验证集，不打乱
val_size = int(len(X_train_all) * 0.2)
X_train = X_train_all.iloc[:-val_size]
X_val = X_train_all.iloc[-val_size:]
y_train = y_train_all.iloc[:-val_size]
y_val = y_train_all.iloc[-val_size:]

print(f"训练集大小: {len(X_train)}")
print(f"验证集大小: {len(X_val)}")

# ==================== 4. 特征工程 ====================
# 4.1 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 4.2 特征选择
selector = SelectKBest(score_func=f_regression, k=15)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_val_selected = selector.transform(X_val_scaled)
X_test_selected = selector.transform(X_test_scaled)

print(f"特征选择后维度: {X_train_selected.shape[1]}")

# ==================== 5. 转换为PyTorch张量 ====================
X_train_tensor = torch.FloatTensor(X_train_selected)
y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
X_val_tensor = torch.FloatTensor(X_val_selected)
y_val_tensor = torch.FloatTensor(y_val.values).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_selected)

# 创建DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ==================== 6. 定义DNN模型 ====================
class COVIDPredictor(nn.Module):
    def __init__(self, input_dim):
        super(COVIDPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.model(x)

# ==================== 7. 训练模型 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = COVIDPredictor(X_train_selected.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# 训练参数
epochs = 100
best_val_loss = float('inf')
patience = 10
patience_counter = 0

# 训练循环
for epoch in range(epochs):
    # 训练阶段
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # 验证阶段
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor.to(device))
        val_loss = criterion(val_outputs, y_val_tensor.to(device))
        val_rmse = torch.sqrt(val_loss).item()
    
    # 学习率调度
    scheduler.step(val_loss)
    
    # 早停检查
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"早停在 epoch {epoch+1}")
        break
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val RMSE: {val_rmse:.4f}')

# 加载最佳模型
model.load_state_dict(best_model_state)

# ==================== 8. 自我评估 ====================
model.eval()
with torch.no_grad():
    val_preds = model(X_val_tensor.to(device)).cpu().numpy()
    val_rmse = np.sqrt(np.mean((val_preds - y_val.values.reshape(-1, 1)) ** 2))
    score = 1.0 / (1.0 + val_rmse)
    print(f"Score = (1.0 / (1.0 + RMSE)) = {score:.6f}")

# ==================== 9. 生成提交文件 ====================
with torch.no_grad():
    test_preds = model(X_test_tensor.to(device)).cpu().numpy().flatten()

# 创建提交文件
submission_df = pd.DataFrame({
    'id': test_df.index,
    'tested_positive': test_preds
})

# 保存提交文件
submission_df.to_csv('submission.csv', index=False)
print(f"提交文件已保存: submission.csv")
print(f"预测值范围: [{test_preds.min():.2f}, {test_preds.max():.2f}]")