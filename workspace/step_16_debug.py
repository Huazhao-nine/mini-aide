import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. 加载数据
train_path = '/home/huazhao/DL-HW/ML-2025/mini-aide/input/train.csv'
test_path = '/home/huazhao/DL-HW/ML-2025/mini-aide/input/test.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# 2. 分离特征和目标
# 训练集: 注意目标列是 'tested_positive_day3'
target_col = 'tested_positive_day3'
if target_col in train_df.columns:
    X_train_full = train_df.drop(columns=[target_col, 'id'])
    y_train_full = train_df[target_col].values
else:
    # 检查是否有'tested_positive'列（可能重命名）
    alt_target = 'tested_positive'
    if alt_target in train_df.columns:
        X_train_full = train_df.drop(columns=[alt_target, 'id'])
        y_train_full = train_df[alt_target].values
        target_col = alt_target
    else:
        raise KeyError(f"目标列 {target_col} 或 'tested_positive' 在训练集中不存在")

# 测试集: 没有目标列，直接删除id列
X_test = test_df.drop(columns=['id'])

print(f"训练集形状: {X_train_full.shape}")
print(f"测试集形状: {X_test.shape}")

# 3. 时间序列切分训练集和验证集（不shuffle）
val_size = int(0.2 * len(X_train_full))
X_train = X_train_full[:-val_size]
y_train = y_train_full[:-val_size]
X_val = X_train_full[-val_size:]
y_val = y_train_full[-val_size:]

print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}")

# 4. 特征工程 - 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 5. 特征选择
selector = SelectKBest(score_func=f_regression, k=15)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_val_selected = selector.transform(X_val_scaled)
X_test_selected = selector.transform(X_test_scaled)

print(f"特征选择后: 训练集 {X_train_selected.shape}, 验证集 {X_val_selected.shape}")

# 6. 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train_selected).to(device)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
X_val_tensor = torch.FloatTensor(X_val_selected).to(device)
y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(device)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)  # 时间序列不shuffle

# 7. 定义神经网络模型
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
            
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.model(x)

# 8. 模型初始化
input_dim = X_train_selected.shape[1]
model = COVIDPredictor(input_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# 9. 训练模型
num_epochs = 100
train_losses = []
val_losses = []

for epoch in range(num_epochs):
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
        val_loss = criterion(val_predictions, y_val_tensor).item()
    
    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss)
    
    # 学习率调度
    scheduler.step(val_loss)
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_loss:.4f}")

# 10. 在验证集上评估
model.eval()
with torch.no_grad():
    val_predictions = model(X_val_tensor)
    val_mse = criterion(val_predictions, y_val_tensor).item()
    val_rmse = np.sqrt(val_mse)
    
    # 计算Score
    score = 1.0 / (1.0 + val_rmse)
    print(f"\n验证集评估结果:")
    print(f"MSE: {val_mse:.4f}")
    print(f"RMSE: {val_rmse:.4f}")
    print(f"Score = (1.0 / (1.0 + RMSE)) = {score:.4f}")

# 11. 对测试集进行预测
X_test_tensor = torch.FloatTensor(X_test_selected).to(device)
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor).cpu().numpy()

# 12. 生成提交文件
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'tested_positive': test_predictions.flatten()
})

# 确保预测值非负（病例数不能为负）
submission_df['tested_positive'] = submission_df['tested_positive'].clip(lower=0)

# 保存提交文件
submission_path = 'submission.csv'
submission_df.to_csv(submission_path, index=False)
print(f"\n提交文件已保存到: {submission_path}")
print(f"提交文件形状: {submission_df.shape}")