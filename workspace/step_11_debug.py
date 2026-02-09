import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# ========== 1. 数据加载 ==========
train_path = '/home/huazhao/DL-HW/ML-2025/mini-aide/input/train.csv'
test_path = '/home/huazhao/DL-HW/ML-2025/mini-aide/input/test.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print(f"训练集形状: {train_df.shape}")
print(f"测试集形状: {test_df.shape}")

# ========== 2. 数据预处理 ==========
# 分离特征和目标
TARGET_COL = 'tested_positive_day3'

# 保存测试集ID用于最终提交
test_ids = test_df['id'].copy()

# 训练集：分离特征和目标
X_train_raw = train_df.drop(['id', TARGET_COL], axis=1)
y_train = train_df[TARGET_COL].values

# 测试集：分离特征
X_test_raw = test_df.drop(['id'], axis=1)

# 确保训练集和测试集列顺序一致
X_test_raw = X_test_raw[X_train_raw.columns]

print(f"训练特征形状: {X_train_raw.shape}")
print(f"测试特征形状: {X_test_raw.shape}")

# ========== 3. 特征工程 ==========
def create_features(df):
    """创建衍生特征，只针对特征列，避免目标列"""
    df = df.copy()
    
    # 1. 基本统计特征（仅使用特征列）
    feature_columns = [col for col in df.columns if 'tested_positive' not in col]
    
    # 为每个测量类型（cli, ili等）创建统计特征
    for base_feature in ['cli', 'ili', 'wnohh_cmnty_cli', 'wbelief_masking_effective', 
                         'wbelief_distancing_effective', 'wcovid_vaccinated_friends',
                         'wlarge_event_indoors', 'wothers_masked_public', 
                         'wothers_distanced_public', 'wshop_indoors', 
                         'wrestaurant_indoors', 'wworried_catch_covid',
                         'hh_cmnty_cli', 'nohh_cmnty_cli', 'wearing_mask_7d',
                         'public_transit', 'worried_finances']:
        
        day_cols = [col for col in df.columns if col.startswith(f'{base_feature}_day')]
        if len(day_cols) >= 2:
            # 创建平均值
            df[f'{base_feature}_mean'] = df[day_cols].mean(axis=1)
            # 创建标准差
            df[f'{base_feature}_std'] = df[day_cols].std(axis=1)
            # 创建趋势（只使用day1和day2）
            if f'{base_feature}_day1' in df.columns and f'{base_feature}_day2' in df.columns:
                df[f'{base_feature}_trend'] = (df[f'{base_feature}_day2'] - df[f'{base_feature}_day1']) / (df[f'{base_feature}_day1'] + 1e-6)
    
    # 2. 交互特征（仅使用特征列）
    if 'cli_day1' in df.columns and 'wearing_mask_7d_day1' in df.columns:
        df['cli_mask_interaction'] = df['cli_day1'] * df['wearing_mask_7d_day1']
    
    if 'ili_day1' in df.columns and 'public_transit_day1' in df.columns:
        df['ili_transit_interaction'] = df['ili_day1'] * df['public_transit_day1']
    
    return df

# 应用特征工程
X_train_fe = create_features(X_train_raw)
X_test_fe = create_features(X_test_raw)

print(f"特征工程后训练特征形状: {X_train_fe.shape}")
print(f"特征工程后测试特征形状: {X_test_fe.shape}")

# ========== 4. 数据标准化 ==========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_fe)
X_test_scaled = scaler.transform(X_test_fe)

# ========== 5. 特征选择 ==========
# 选择与目标相关性最高的20个特征
k = min(20, X_train_scaled.shape[1])
selector = SelectKBest(score_func=f_regression, k=k)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

print(f"特征选择后训练特征形状: {X_train_selected.shape}")
print(f"特征选择后测试特征形状: {X_test_selected.shape}")

# ========== 6. 划分训练集和验证集（时间顺序） ==========
# 按时间顺序划分：前80%训练，后20%验证
val_size = int(0.2 * len(X_train_selected))
X_train_final = X_train_selected[:-val_size]
y_train_final = y_train[:-val_size]
X_val_final = X_train_selected[-val_size:]
y_val_final = y_train[-val_size:]

print(f"最终训练集大小: {X_train_final.shape}")
print(f"最终验证集大小: {X_val_final.shape}")

# ========== 7. 转换为PyTorch张量 ==========
X_train_tensor = torch.FloatTensor(X_train_final)
y_train_tensor = torch.FloatTensor(y_train_final).view(-1, 1)
X_val_tensor = torch.FloatTensor(X_val_final)
y_val_tensor = torch.FloatTensor(y_val_final).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_selected)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)  # 注意：shuffle=False

# ========== 8. 定义神经网络模型 ==========
class COVIDPredictor(nn.Module):
    def __init__(self, input_dim):
        super(COVIDPredictor, self).__init__()
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
            nn.Dropout(0.1),
            
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.model(x)

# ========== 9. 训练模型 ==========
model = COVIDPredictor(X_train_final.shape[1])
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

num_epochs = 100
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # 训练模式
    model.train()
    train_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # 验证模式
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
    
    # 打印进度
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss.item():.4f}')
    
    # 保存最佳模型
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        torch.save(model.state_dict(), 'best_model.pth')

# 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))

# ========== 10. 评估模型 ==========
model.eval()
with torch.no_grad():
    # 在验证集上计算预测
    val_predictions = model(X_val_tensor)
    
    # 计算RMSE
    val_rmse = torch.sqrt(criterion(val_predictions, y_val_tensor)).item()
    
    # 计算分数
    score = 1.0 / (1.0 + val_rmse)
    
    print(f"验证集RMSE: {val_rmse:.4f}")
    print(f"Score = (1.0 / (1.0 + RMSE)) = {score:.4f}")

# ========== 11. 在测试集上进行预测 ==========
with torch.no_grad():
    test_predictions = model(X_test_tensor).numpy().flatten()

# ========== 12. 生成提交文件 ==========
submission_df = pd.DataFrame({
    'id': test_ids,
    'tested_positive': test_predictions
})

# 确保没有负值（实际感染数不能为负）
submission_df['tested_positive'] = submission_df['tested_positive'].clip(lower=0)

submission_df.to_csv('submission.csv', index=False)
print("提交文件已生成: submission.csv")
print(f"预测值范围: [{submission_df['tested_positive'].min():.2f}, {submission_df['tested_positive'].max():.2f}]")