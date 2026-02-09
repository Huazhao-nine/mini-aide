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

# 1. 数据加载
train_df = pd.read_csv('/home/huazhao/DL-HW/ML-2025/mini-aide/input/train.csv')
test_df = pd.read_csv('/home/huazhao/DL-HW/ML-2025/mini-aide/input/test.csv')

print(f"训练集形状: {train_df.shape}")
print(f"测试集形状: {test_df.shape}")

# 2. 分离特征和目标
# 训练集：分离特征和目标列
y_train = train_df['tested_positive_day3'].values.reshape(-1, 1)

# 从训练集中删除目标列
X_train_raw = train_df.drop(columns=['tested_positive_day3', 'id'])
# 从测试集中删除id列
X_test_raw = test_df.drop(columns=['id'])

# 确保训练集和测试集特征列一致
# 按测试集的特征列顺序重新排列训练集特征
X_train_raw = X_train_raw[X_test_raw.columns]

print(f"训练集特征形状: {X_train_raw.shape}")
print(f"测试集特征形状: {X_test_raw.shape}")

# 3. 特征工程
# 3.1 特征选择（使用SelectKBest）
selector = SelectKBest(score_func=f_regression, k=min(20, X_train_raw.shape[1]))
X_train_selected = selector.fit_transform(X_train_raw, y_train.flatten())
X_test_selected = selector.transform(X_test_raw)

print(f"特征选择后训练集形状: {X_train_selected.shape}")
print(f"特征选择后测试集形状: {X_test_selected.shape}")

# 3.2 标准化（使用StandardScaler而不是RobustScaler）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# 4. 划分训练集和验证集（时间序列划分，不shuffle）
val_size = int(0.2 * len(X_train_scaled))
X_train = X_train_scaled[:-val_size]
y_train_split = y_train[:-val_size]
X_val = X_train_scaled[-val_size:]
y_val = y_train[-val_size:]

print(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}")

# 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train_split)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.FloatTensor(y_val)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)  # 时间序列不shuffle

# 5. 定义DNN模型
class COVID19Predictor(nn.Module):
    def __init__(self, input_dim):
        super(COVID19Predictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.model(x)

# 6. 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = COVID19Predictor(X_train.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # 验证
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_tensor.to(device))
        val_loss = criterion(val_pred, y_val_tensor.to(device))
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss.item():.4f}')

# 7. 在验证集上评估
model.eval()
with torch.no_grad():
    val_pred = model(X_val_tensor.to(device))
    val_pred = val_pred.cpu().numpy()

# 计算RMSE
rmse = np.sqrt(np.mean((val_pred - y_val.numpy())**2))
score = 1.0 / (1.0 + rmse)
print(f"验证集RMSE: {rmse:.6f}")
print(f"Score = (1.0 / (1.0 + RMSE)) = {score:.6f}")

# 8. 对测试集进行预测
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
with torch.no_grad():
    test_predictions = model(X_test_tensor)
test_predictions = test_predictions.cpu().numpy()

# 9. 生成提交文件
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'tested_positive': test_predictions.flatten()
})

# 确保没有负值（实际病例数不能为负）
submission_df['tested_positive'] = submission_df['tested_positive'].clip(lower=0)

submission_df.to_csv('submission.csv', index=False)
print("提交文件已保存为 submission.csv")
print(f"提交文件形状: {submission_df.shape}")
print("前几行预测值:")
print(submission_df.head())