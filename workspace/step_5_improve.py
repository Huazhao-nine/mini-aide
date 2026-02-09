import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
import gc

# ========== 1. 数据加载与预处理 ==========
train_path = '/home/huazhao/DL-HW/ML-2025/mini-aide/input/train.csv'
test_path = '/home/huazhao/DL-HW/ML-2025/mini-aide/input/test.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# 提取目标列
target_col = 'tested_positive_day3'
y_raw = train_df[target_col].values

# ========== 2. 高级特征工程 ==========
def create_features(df, is_test=False):
    """
    创建增强的特征集，包括：
    1. 交互特征
    2. 滞后特征和移动平均
    3. 统计特征
    4. 多项式特征
    """
    df = df.copy()
    
    # 基础特征列（排除id和可能的target）
    if is_test:
        feature_cols = [col for col in df.columns if col != 'id']
    else:
        feature_cols = [col for col in df.columns if col not in ['id', target_col]]
    
    # 创建交互特征（有意义的组合）
    for day in ['1', '2', '3']:
        if f'cli_day{day}' in df.columns and f'wearing_mask_7d_day{day}' in df.columns:
            df[f'cli_mask_interaction_day{day}'] = df[f'cli_day{day}'] * df[f'wearing_mask_7d_day{day}']
        
        if f'tested_positive_day{day}' in df.columns:
            # 如果有前几天的阳性数据，创建相关特征
            for other_day in ['1', '2']:
                if other_day != day and f'tested_positive_day{other_day}' in df.columns:
                    df[f'tested_positive_ratio_day{day}_day{other_day}'] = (
                        df[f'tested_positive_day{day}'] / (df[f'tested_positive_day{other_day}'] + 1e-6)
                    )
    
    # 创建时间序列特征（移动平均）
    for col in ['cli_day', 'ili_day', 'wnohh_cmnty_cli_day']:
        for day in ['1', '2', '3']:
            col_name = f'{col}{day}'
            if col_name in df.columns:
                # 创建两天的平均
                if day in ['2', '3']:
                    prev_day = str(int(day) - 1)
                    prev_col = f'{col}{prev_day}'
                    if prev_col in df.columns:
                        df[f'{col}_avg_{prev_day}_{day}'] = (df[col_name] + df[prev_col]) / 2
    
    # 创建统计特征（州的聚合）
    state_cols = [col for col in df.columns if col in ['AL', 'AZ', 'CA', 'CO', 'CT', 'FL', 'GA', 'IL', 'IN', 'IA', 
                                                      'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MO', 'NJ', 
                                                      'NM', 'NY', 'NC', 'OH', 'OK', 'OR', 'PA', 'SC', 'TN', 'TX', 
                                                      'VA', 'WA', 'WV', 'WI']]
    
    if state_cols:
        df['states_sum'] = df[state_cols].sum(axis=1)
        df['states_mean'] = df[state_cols].mean(axis=1)
    
    # 对特定特征进行多项式变换
    poly_features = ['cli_day3', 'ili_day3', 'tested_positive_day2']
    for col in poly_features:
        if col in df.columns:
            df[f'{col}_squared'] = df[col] ** 2
            df[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
    
    # 更新特征列
    new_feature_cols = [col for col in df.columns if col not in ['id', target_col]]
    
    return df[new_feature_cols].values, new_feature_cols

# 应用特征工程
X_train_raw, feature_cols = create_features(train_df, is_test=False)
X_test_raw, _ = create_features(test_df, is_test=True)

print(f"原始训练集形状: {train_df.shape}")
print(f"特征工程后训练集形状: {X_train_raw.shape}")
print(f"特征工程后测试集形状: {X_test_raw.shape}")

# ========== 3. 目标变量变换（处理偏态） ==========
# 使用log1p变换处理偏态分布的目标变量
y_transformed = np.log1p(y_raw)

# ========== 4. 时间序列划分验证集 ==========
# 取最后20%作为验证集，不shuffle
val_size = int(0.2 * len(X_train_raw))
X_train_split = X_train_raw[:-val_size]
y_train_split = y_transformed[:-val_size]
X_val_split = X_train_raw[-val_size:]
y_val_split = y_transformed[-val_size:]
y_val_raw = y_raw[-val_size:]  # 保留原始值用于评估

print(f"训练集分割后: {X_train_split.shape}, 验证集: {X_val_split.shape}")

# ========== 5. 特征缩放和选择 ==========
# 5.1 使用RobustScaler处理异常值
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_split)
X_val_scaled = scaler.transform(X_val_split)
X_test_scaled = scaler.transform(X_test_raw)

# 5.2 特征选择 - 使用基于模型的重要性选择
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train_split)

# 选择最重要的50个特征
selector = SelectFromModel(rf, threshold='median', max_features=50)
X_train_selected = selector.fit_transform(X_train_scaled, y_train_split)
X_val_selected = selector.transform(X_val_scaled)
X_test_selected = selector.transform(X_test_scaled)

# 5.3 使用PCA进行降维（可选）
pca = PCA(n_components=0.95)  # 保留95%的方差
X_train_pca = pca.fit_transform(X_train_selected)
X_val_pca = pca.transform(X_val_selected)
X_test_pca = pca.transform(X_test_selected)

print(f"PCA后维度: {X_train_pca.shape}")

# ========== 6. 转换为PyTorch张量 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

X_train_tensor = torch.FloatTensor(X_train_pca).to(device)
y_train_tensor = torch.FloatTensor(y_train_split).to(device).view(-1, 1)
X_val_tensor = torch.FloatTensor(X_val_pca).to(device)
y_val_tensor = torch.FloatTensor(y_val_split).to(device).view(-1, 1)

# 为时间序列创建DataLoader（不shuffle！）
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

# ========== 7. 定义更强大的DNN模型 ==========
class COVIDPredictorV2(nn.Module):
    def __init__(self, input_dim):
        super(COVIDPredictorV2, self).__init__()
        
        self.batch_norm_input = nn.BatchNorm1d(input_dim)
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
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
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.batch_norm_input(x)
        return self.network(x)

model = COVIDPredictorV2(X_train_pca.shape[1]).to(device)

# ========== 8. 训练设置 ==========
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

# 学习率调度器 - 余弦退火
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2, eta_min=1e-6
)

# ========== 9. 训练循环（带早停） ==========
epochs = 300
best_val_loss = float('inf')
patience = 30
patience_counter = 0
train_losses = []
val_losses = []

for epoch in range(epochs):
    # 训练阶段
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # 验证阶段
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val_tensor)
        val_loss = criterion(val_predictions, y_val_tensor).item()
        val_losses.append(val_loss)
    
    # 学习率调度
    scheduler.step()
    
    # 早停和模型保存
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'Epoch [{epoch+1}/{epochs}] - New best model: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    else:
        patience_counter += 1
    
    # 打印进度
    if (epoch + 1) % 30 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'LR: {current_lr:.6f}')
    
    # 早停检查
    if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch+1}')
        break

# ========== 10. 在验证集上评估 ==========
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
with torch.no_grad():
    val_predictions_transformed = model(X_val_tensor)
    
    # 将预测值转换回原始尺度
    val_predictions = np.expm1(val_predictions_transformed.cpu().numpy().flatten())
    
    # 计算RMSE（在原始尺度上）
    val_rmse = np.sqrt(np.mean((val_predictions - y_val_raw) ** 2))

# 计算Score
score = 1.0 / (1.0 + val_rmse)
print(f'\n{"="*50}')
print(f'验证集RMSE: {val_rmse:.6f}')
print(f'Score = (1.0 / (1.0 + RMSE)) = {score:.6f}')
print(f'{"="*50}\n')

# ========== 11. 对测试集进行预测 ==========
X_test_tensor = torch.FloatTensor(X_test_pca).to(device)
model.eval()
with torch.no_grad():
    test_predictions_transformed = model(X_test_tensor).cpu().numpy()
    
    # 将测试集预测值转换回原始尺度
    test_predictions = np.expm1(test_predictions_transformed.flatten())

# 确保没有负值
test_predictions = np.maximum(test_predictions, 0)

# ========== 12. 生成提交文件 ==========
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'tested_positive': test_predictions
})
submission_df.to_csv('submission.csv', index=False)
print(f"提交文件已保存，形状: {submission_df.shape}")

# ========== 13. 模型集成（可选）- 训练多个模型并平均 ==========
def train_model_with_seed(seed):
    """使用不同随机种子训练模型"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model_seed = COVIDPredictorV2(X_train_pca.shape[1]).to(device)
    optimizer_seed = optim.AdamW(model_seed.parameters(), lr=0.001, weight_decay=1e-5)
    
    # 简化的训练过程
    for epoch in range(100):
        model_seed.train()
        for batch_X, batch_y in train_loader:
            optimizer_seed.zero_grad()
            predictions = model_seed(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_seed.parameters(), max_norm=1.0)
            optimizer_seed.step()
    
    return model_seed

# 如果需要可以启用集成学习
# ensemble_models = []
# seeds = [42, 123, 456, 789, 999]
# for seed in seeds:
#     model_seed = train_model_with_seed(seed)
#     ensemble_models.append(model_seed)

# 清理内存
del X_train_raw, X_test_raw, X_train_scaled, X_val_scaled, X_test_scaled
gc.collect()