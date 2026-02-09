import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 1. 数据加载
train_df = pd.read_csv('/home/huazhao/DL-HW/ML-2025/mini-aide/input/train.csv')
test_df = pd.read_csv('/home/huazhao/DL-HW/ML-2025/mini-aide/input/test.csv')

print(f"训练集形状: {train_df.shape}")
print(f"测试集形状: {test_df.shape}")

# 2. 分离特征和目标
# 测试集没有目标列，所以要确保处理方式一致
train_features = train_df.drop(columns=['tested_positive_day3'])
train_target = train_df['tested_positive_day3']

# 确保测试集ID被正确保存
test_id = test_df['id']

# 3. 确保特征列顺序一致
# 找出共同的特征列（排除目标列）
common_features = list(set(train_features.columns) & set(test_df.columns))
common_features = sorted(common_features)  # 确保顺序一致

# 重新排列数据框
train_features = train_features[common_features]
test_df = test_df[common_features]

print(f"处理后训练特征形状: {train_features.shape}")
print(f"处理后测试特征形状: {test_df.shape}")

# 4. 时间序列划分验证集（绝对禁止shuffle！）
val_size = int(len(train_features) * 0.2)
X_train = train_features.iloc[:-val_size]
y_train = train_target.iloc[:-val_size]
X_val = train_features.iloc[-val_size:]
y_val = train_target.iloc[-val_size:]

print(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}")

# 5. 特征工程
def create_interaction_features(df):
    """创建交互特征"""
    df_copy = df.copy()
    
    # 创建一些有意义的交互特征
    # 注意：只在数值型特征上创建
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    
    # 移除ID列（如果是数值型）
    if 'id' in numeric_cols:
        numeric_cols.remove('id')
    
    # 创建一些简单的交互特征
    interaction_features = []
    
    # 与州特征交互（州特征是独热编码）
    state_cols = [col for col in df_copy.columns if col in ['AL', 'AZ', 'CA', 'CO', 'CT', 'FL', 'GA', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MO', 'NJ', 'NM', 'NY', 'NC', 'OH', 'OK', 'OR', 'PA', 'SC', 'TN', 'TX', 'VA', 'WA', 'WV', 'WI']]
    
    # 创建症状相关的交互特征
    if 'cli_day1' in numeric_cols and 'ili_day1' in numeric_cols:
        df_copy['cli_ili_day1'] = df_copy['cli_day1'] * df_copy['ili_day1']
        interaction_features.append('cli_ili_day1')
    
    if 'cli_day2' in numeric_cols and 'ili_day2' in numeric_cols:
        df_copy['cli_ili_day2'] = df_copy['cli_day2'] * df_copy['ili_day2']
        interaction_features.append('cli_ili_day2')
    
    if 'cli_day3' in numeric_cols and 'ili_day3' in numeric_cols:
        df_copy['cli_ili_day3'] = df_copy['cli_day3'] * df_copy['ili_day3']
        interaction_features.append('cli_ili_day3')
    
    # 创建口罩相关交互特征
    mask_cols = [col for col in numeric_cols if 'mask' in col.lower()]
    if 'wearing_mask_7d_day1' in numeric_cols and 'cli_day1' in numeric_cols:
        df_copy['mask_cli_day1'] = df_copy['wearing_mask_7d_day1'] * df_copy['cli_day1']
        interaction_features.append('mask_cli_day1')
    
    if 'wearing_mask_7d_day2' in numeric_cols and 'cli_day2' in numeric_cols:
        df_copy['mask_cli_day2'] = df_copy['wearing_mask_7d_day2'] * df_copy['cli_day2']
        interaction_features.append('mask_cli_day2')
    
    if 'wearing_mask_7d_day3' in numeric_cols and 'cli_day3' in numeric_cols:
        df_copy['mask_cli_day3'] = df_copy['wearing_mask_7d_day3'] * df_copy['cli_day3']
        interaction_features.append('mask_cli_day3')
    
    print(f"创建了 {len(interaction_features)} 个交互特征")
    return df_copy

# 对所有数据集应用相同的特征工程
X_train = create_interaction_features(X_train)
X_val = create_interaction_features(X_val)
test_df = create_interaction_features(test_df)

# 6. 移除ID列（不作为特征）
X_train_features = X_train.drop(columns=['id']) if 'id' in X_train.columns else X_train
X_val_features = X_val.drop(columns=['id']) if 'id' in X_val.columns else X_val
test_features = test_df.drop(columns=['id']) if 'id' in test_df.columns else test_df

# 7. 特征选择和标准化
# 7.1 先进行特征选择（减少维度）
print(f"特征选择前维度: {X_train_features.shape[1]}")

# 使用SelectKBest选择最佳特征
selector = SelectKBest(score_func=f_regression, k=min(50, X_train_features.shape[1]))
X_train_selected = selector.fit_transform(X_train_features, y_train)
X_val_selected = selector.transform(X_val_features)
test_selected = selector.transform(test_features)

# 获取选择的特征名
selected_features = X_train_features.columns[selector.get_support()].tolist()
print(f"特征选择后维度: {len(selected_features)}")

# 7.2 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_val_scaled = scaler.transform(X_val_selected)
test_scaled = scaler.transform(test_selected)

# 8. 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)
X_val_tensor = torch.FloatTensor(X_val_scaled)
y_val_tensor = torch.FloatTensor(y_val.values).view(-1, 1)

# 创建DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 训练时可以shuffle

# 9. 定义DNN模型
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
            
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.network(x)

# 10. 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = COVIDPredictor(X_train_scaled.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# 训练参数
epochs = 100
best_val_loss = float('inf')
patience = 20
patience_counter = 0

for epoch in range(epochs):
    # 训练阶段
    model.train()
    train_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # 验证阶段
    model.eval()
    with torch.no_grad():
        X_val_device = X_val_tensor.to(device)
        val_outputs = model(X_val_device)
        val_loss = criterion(val_outputs, y_val_tensor.to(device))
        val_rmse = torch.sqrt(val_loss)
    
    # 早停机制
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"早停在第 {epoch+1} 轮")
        break
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val RMSE: {val_rmse.item():.4f}")

# 11. 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))

# 12. 验证集最终评估
model.eval()
with torch.no_grad():
    X_val_device = X_val_tensor.to(device)
    val_predictions = model(X_val_device)
    val_rmse = torch.sqrt(criterion(val_predictions, y_val_tensor.to(device)))
    
    # 计算Score
    score = 1.0 / (1.0 + val_rmse.item())
    print(f"\n最终验证集RMSE: {val_rmse.item():.4f}")
    print(f"Score = (1.0 / (1.0 + RMSE)) = {score:.4f}")

# 13. 测试集预测
model.eval()
with torch.no_grad():
    test_tensor = torch.FloatTensor(test_scaled).to(device)
    test_predictions = model(test_tensor).cpu().numpy()

# 14. 生成提交文件
# 确保预测值非负（确诊人数不能为负）
test_predictions = np.maximum(test_predictions, 0)

submission_df = pd.DataFrame({
    'id': test_id,
    'tested_positive': test_predictions.flatten()
})

submission_df.to_csv('submission.csv', index=False)
print(f"\n提交文件已保存: submission.csv")
print(f"预测形状: {test_predictions.shape}")
print(f"预测值范围: [{test_predictions.min():.2f}, {test_predictions.max():.2f}]")