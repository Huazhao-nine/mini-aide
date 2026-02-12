import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# 1. 数据加载
train_path = '/home/huazhao/DL-HW/ML-2025/mini-aide/input/train.csv'
test_path = '/home/huazhao/DL-HW/ML-2025/mini-aide/input/test.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print(f"训练集形状: {train_df.shape}")
print(f"测试集形状: {test_df.shape}")

# 2. 准备数据
# 分离特征和目标
target_col = 'tested_positive_day3'
y = train_df[target_col].values

# 排除目标列和ID列
feature_cols = [col for col in train_df.columns if col not in ['id', target_col]]
X_train = train_df[feature_cols].values
X_test = test_df[feature_cols].values

# 3. 创建交互特征
def create_interaction_features(X, feature_names):
    """创建交互特征"""
    # 获取症状和行为特征的索引
    symptom_features = [i for i, name in enumerate(feature_names) 
                       if any(keyword in name for keyword in ['cli', 'ili', 'cmnty'])]
    behavior_features = [i for i, name in enumerate(feature_names) 
                        if any(keyword in name for keyword in ['mask', 'shop', 'restaurant', 'transit', 'event'])]
    
    interaction_features = []
    for sf in symptom_features:
        for bf in behavior_features:
            # 只在同一天的特征之间创建交互
            if 'day' in feature_names[sf] and 'day' in feature_names[bf]:
                sf_day = feature_names[sf].split('_day')[1][0]
                bf_day = feature_names[bf].split('_day')[1][0]
                if sf_day == bf_day:  # 确保是同一天的特征
                    interaction_features.append(X[:, sf] * X[:, bf])
    
    if interaction_features:
        interaction_features = np.column_stack(interaction_features)
        X_enhanced = np.hstack([X, interaction_features])
        return X_enhanced
    return X

# 4. 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 创建交互特征
X_train_enhanced = create_interaction_features(X_train_scaled, feature_cols)

# 特征选择
k = 15
selector = SelectKBest(f_regression, k=k)
X_train_selected = selector.fit_transform(X_train_enhanced, y)

print(f"特征选择后训练集形状: {X_train_selected.shape}")

# 5. 划分训练集和验证集（时间序列，不shuffle）
train_size = int(len(X_train_selected) * 0.8)
X_train_final = X_train_selected[:train_size]
y_train = y[:train_size]
X_val = X_train_selected[train_size:]
y_val = y[train_size:]

print(f"训练集大小: {X_train_final.shape}, 验证集大小: {X_val.shape}")

# 6. 准备测试集
X_test_scaled = scaler.transform(X_test)
X_test_enhanced = create_interaction_features(X_test_scaled, feature_cols)
X_test_selected = selector.transform(X_test_enhanced)

# 7. 创建PyTorch数据集
class COVIDDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx].unsqueeze(0)
        return self.X[idx]

train_dataset = COVIDDataset(X_train_final, y_train)
val_dataset = COVIDDataset(X_val, y_val)
test_dataset = COVIDDataset(X_test_selected)

# 8. 创建数据加载器（注意：shuffle=False 对于训练集和验证集）
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 9. 定义神经网络模型
class COVIDNet(nn.Module):
    def __init__(self, input_dim):
        super(COVIDNet, self).__init__()
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

# 10. 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = COVIDNet(input_dim=X_train_selected.shape[1]).to(device)
print(f"使用设备: {device}")

# 11. 定义损失函数和优化器
criterion = nn.L1Loss()  # MAE损失
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

# 12. 训练模型
num_epochs = 200
best_val_loss = float('inf')
patience = 20
patience_counter = 0

for epoch in range(num_epochs):
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
    val_loss = 0
    val_preds = []
    val_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            
            val_preds.extend(outputs.cpu().numpy())
            val_targets.extend(batch_y.cpu().numpy())
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    
    # 学习率调度
    scheduler.step(val_loss)
    
    # 早停
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # 保存最佳模型
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
    
    if epoch % 20 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch}')
        break

# 13. 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))

# 14. 在验证集上评估
model.eval()
val_preds = []
val_targets = []

with torch.no_grad():
    for batch_X, batch_y in val_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        val_preds.extend(outputs.cpu().numpy().flatten())
        val_targets.extend(batch_y.cpu().numpy().flatten())

# 计算RMSE
val_preds = np.array(val_preds)
val_targets = np.array(val_targets)
rmse = np.sqrt(np.mean((val_preds - val_targets) ** 2))

# 计算评估分数
score = 1.0 / (1.0 + rmse)

print(f"验证集RMSE: {rmse:.6f}")
print(f"Score= (1.0 / (1.0 + RMSE)) = {score:.6f}")

# 15. 在测试集上进行预测
model.eval()
test_preds = []

with torch.no_grad():
    for batch_X in test_loader:
        batch_X = batch_X.to(device)
        outputs = model(batch_X)
        test_preds.extend(outputs.cpu().numpy().flatten())

test_preds = np.array(test_preds)

# 确保没有负值（根据约束，只处理负值）
test_preds = np.maximum(test_preds, 0)

# 16. 生成提交文件
submission = pd.DataFrame({
    'id': test_df['id'],
    'tested_positive': test_preds
})

submission.to_csv('submission.csv', index=False)
print("提交文件已保存为 'submission.csv'")

# 打印最终的评估分数（按要求的格式）
print(f"Score= (1.0 / (1.0 + RMSE)) = {score:.6f}")