rmse = np.sqrt(np.mean((val_pred - y_val.numpy())**2))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 1. 数据加载
train_path = '/home/huazhao/DL-HW/ML-2025/mini-aide/input/train.csv'
test_path = '/home/huazhao/DL-HW/ML-2025/mini-aide/input/test.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print(f"训练集形状: {train_df.shape}")
print(f"测试集形状: {test_df.shape}")

# 2. 特征工程
# 2.1 分离特征和目标
target_col = 'tested_positive_day3'
X_train_raw = train_df.drop(columns=[target_col, 'id'])
y_train_raw = train_df[target_col].values

X_test_raw = test_df.drop(columns=['id'])

# 2.2 特征选择
# 使用SelectKBest选择前k个特征
k = 15  # 选择前15个特征
selector = SelectKBest(score_func=f_regression, k=k)
X_train_selected = selector.fit_transform(X_train_raw, y_train_raw)
X_test_selected = selector.transform(X_test_raw)

print(f"特征选择后训练集形状: {X_train_selected.shape}")
print(f"特征选择后测试集形状: {X_test_selected.shape}")

# 2.3 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# 3. 划分训练集和验证集（注意：时间序列任务，不能打乱数据）
# 取最后20%作为验证集
val_size = int(0.2 * len(X_train_scaled))
X_train = X_train_scaled[:-val_size]
X_val = X_train_scaled[-val_size:]
y_train = y_train_raw[:-val_size]
y_val = y_train_raw[-val_size:]

print(f"训练集大小: {X_train.shape}")
print(f"验证集大小: {X_val.shape}")

# 转换为PyTorch Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

# 4. 定义神经网络模型
class COVIDPredictor(nn.Module):
    def __init__(self, input_dim):
        super(COVIDPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.network(x)

# 5. 初始化模型、损失函数和优化器
input_dim = X_train.shape[1]
model = COVIDPredictor(input_dim)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# 6. 训练模型
epochs = 200
batch_size = 32
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 注意：验证集不要打乱，保持时间顺序
val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print("开始训练...")
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # 验证集评估
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            val_loss += loss.item()
            val_preds.append(predictions.numpy())
            val_targets.append(batch_y.numpy())
    
    # 计算平均损失
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    
    # 每50个epoch打印一次损失
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

print("训练完成！")

# 7. 在验证集上进行最终评估
model.eval()
with torch.no_grad():
    val_pred = model(X_val_tensor).numpy()

# 计算RMSE - 修复错误：直接使用NumPy数组计算
rmse = np.sqrt(np.mean((val_pred - y_val)**2))
print(f"验证集RMSE: {rmse:.6f}")

# 计算Score
score = 1.0 / (1.0 + rmse)
print(f"Score = (1.0 / (1.0 + RMSE)) = {score:.6f}")

# 8. 在测试集上进行预测
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor).numpy().flatten()

# 9. 生成提交文件
submission = pd.DataFrame({
    'id': test_df['id'],
    'tested_positive': test_predictions
})

# 确保预测值为非负数
submission['tested_positive'] = submission['tested_positive'].clip(lower=0)

# 保存提交文件
submission_path = 'submission.csv'
submission.to_csv(submission_path, index=False)
print(f"提交文件已保存到: {submission_path}")
print(f"提交文件形状: {submission.shape}")