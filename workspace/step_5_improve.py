import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子保证可重复性
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

# 1. 加载数据
train_df = pd.read_csv('/home/huazhao/DL-HW/ML-2025/mini-aide/input/train.csv')
test_df = pd.read_csv('/home/huazhao/DL-HW/ML-2025/mini-aide/input/test.csv')

# 2. 分离特征和目标
target_col = 'tested_positive_day3'
y_train = train_df[target_col].values.reshape(-1, 1)
X_train = train_df.drop(['id', target_col], axis=1)
X_test = test_df.drop(['id'], axis=1)

# 3. 高级特征工程
def create_advanced_features(df):
    """创建高级特征工程"""
    new_df = df.copy()
    
    # 获取州特征列
    state_cols = [col for col in df.columns if len(col) == 2 and col.isupper()]
    
    # 1. 时间序列特征（滞后和差值）
    for day in [1, 2, 3]:
        # 症状相关特征
        for symptom in ['cli', 'ili', 'nohh_cmnty_cli', 'hh_cmnty_cli']:
            col_name = f'{symptom}_day{day}'
            if col_name in df.columns:
                # 创建相对变化率
                if day > 1:
                    prev_col = f'{symptom}_day{day-1}'
                    if prev_col in df.columns:
                        new_df[f'{symptom}_diff_{day-1}_{day}'] = new_df[col_name] - new_df[prev_col]
                        new_df[f'{symptom}_ratio_{day-1}_{day}'] = new_df[col_name] / (new_df[prev_col] + 1e-8)
        
        # 行为相关特征
        for behavior in ['wearing_mask_7d', 'wshop_indoors', 'wrestaurant_indoors', 'public_transit']:
            col_name = f'{behavior}_day{day}'
            if col_name in df.columns and day > 1:
                prev_col = f'{behavior}_day{day-1}'
                if prev_col in df.columns:
                    new_df[f'{behavior}_diff_{day-1}_{day}'] = new_df[col_name] - new_df[prev_col]
    
    # 2. 基于领域知识的交互特征
    for day in [1, 2, 3]:
        # 症状与行为的交互
        new_df[f'cli_mask_ratio_day{day}'] = new_df[f'cli_day{day}'] * (100 - new_df[f'wearing_mask_7d_day{day}']) / 100
        new_df[f'worried_behavior_day{day}'] = new_df[f'wworried_catch_covid_day{day}'] * (new_df[f'wshop_indoors_day{day}'] + new_df[f'wrestaurant_indoors_day{day}']) / 2
        new_df[f'vaccine_confidence_day{day}'] = new_df[f'wcovid_vaccinated_friends_day{day}'] * new_df[f'wbelief_masking_effective_day{day}'] / 10000
        
        # 社区传播风险指标
        new_df[f'community_risk_day{day}'] = (new_df[f'nohh_cmnty_cli_day{day}'] * 0.7 + 
                                              new_df[f'hh_cmnty_cli_day{day}'] * 0.3)
        new_df[f'indoor_exposure_day{day}'] = (new_df[f'wlarge_event_indoors_day{day}'] * 0.4 + 
                                               new_df[f'wrestaurant_indoors_day{day}'] * 0.4 + 
                                               new_df[f'wshop_indoors_day{day}'] * 0.2)
    
    # 3. 跨天聚合特征
    for feature_base in ['cli', 'ili', 'tested_positive', 'wearing_mask_7d', 'wcovid_vaccinated_friends']:
        cols = [f'{feature_base}_day{i}' for i in [1, 2, 3] if f'{feature_base}_day{i}' in df.columns]
        if len(cols) >= 2:
            # 平均值
            new_df[f'{feature_base}_avg'] = df[cols].mean(axis=1)
            # 趋势（最后一天减第一天）
            if len(cols) == 3:
                new_df[f'{feature_base}_trend'] = df[cols[2]] - df[cols[0]]
            # 波动率
            new_df[f'{feature_base}_std'] = df[cols].std(axis=1)
    
    # 4. 状态级别特征（如果有多个州）
    if state_cols:
        # 为每个特征创建与州均值的比值
        for col in df.columns:
            if 'day' in col and col not in state_cols:
                # 这里简化处理，实际应用中可能需要计算每个州的统计量
                new_df[f'{col}_norm'] = new_df[col] / (new_df[col].mean() + 1e-8)
    
    # 5. 复合风险指标
    for day in [1, 2, 3]:
        new_df[f'composite_risk_day{day}'] = (
            new_df[f'cli_day{day}'] * 0.3 +
            new_df[f'nohh_cmnty_cli_day{day}'] * 0.2 +
            new_df[f'indoor_exposure_day{day}'] * 0.2 +
            (100 - new_df[f'wearing_mask_7d_day{day}']) * 0.1 +
            (100 - new_df[f'wcovid_vaccinated_friends_day{day}']) * 0.1 +
            new_df[f'public_transit_day{day}'] * 0.1
        )
    
    return new_df

print("原始特征数量:", X_train.shape[1])
X_train = create_advanced_features(X_train)
X_test = create_advanced_features(X_test)
print("特征工程后特征数量:", X_train.shape[1])

# 4. 时间序列验证集划分（最后20%）
split_idx = int(len(X_train) * 0.8)
X_val = X_train.iloc[split_idx:].copy()
y_val = y_train[split_idx:].copy()
X_train_split = X_train.iloc[:split_idx].copy()
y_train_split = y_train[:split_idx].copy()

print(f"训练集大小: {len(X_train_split)}, 验证集大小: {len(X_val)}")

# 5. 特征标准化（使用RobustScaler对异常值更鲁棒）
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_split)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 6. 特征选择（使用递归特征消除）
# 先使用PCA降维到50维，再进行特征选择
pca = PCA(n_components=min(50, X_train_scaled.shape[1]))
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"PCA后特征维度: {X_train_pca.shape[1]}")
print(f"解释方差比例: {pca.explained_variance_ratio_.sum():.4f}")

# 7. 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train_pca)
y_train_tensor = torch.FloatTensor(y_train_split)
X_val_tensor = torch.FloatTensor(X_val_pca)
y_val_tensor = torch.FloatTensor(y_val)
X_test_tensor = torch.FloatTensor(X_test_pca)

# 创建DataLoader（注意：时间序列严格禁止shuffle）
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # 时间序列禁止shuffle
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 8. 定义改进的神经网络模型（带残差连接和注意力机制）
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        
        # 如果输入输出维度不同，需要调整残差连接
        self.skip = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
    
    def forward(self, x):
        identity = self.skip(x)
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out += identity  # 残差连接
        out = self.relu(out)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class EnhancedCOVIDPredictor(nn.Module):
    def __init__(self, input_dim):
        super(EnhancedCOVIDPredictor, self).__init__()
        
        # 注意力层
        self.attention = AttentionBlock(input_dim)
        
        # 特征提取层
        self.feature_extractor = nn.Sequential(
            ResidualBlock(input_dim, 256, dropout_rate=0.3),
            ResidualBlock(256, 128, dropout_rate=0.25),
            ResidualBlock(128, 64, dropout_rate=0.2),
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
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
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 注意力机制
        x = self.attention(x)
        
        # 特征提取
        features = self.feature_extractor(x)
        
        # 输出
        output = self.output_layer(features)
        return output

# 9. 训练设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

model = EnhancedCOVIDPredictor(input_dim=X_train_pca.shape[1]).to(device)

# 使用Huber损失，对异常值更鲁棒
criterion = nn.HuberLoss(delta=1.0)

# AdamW优化器（带权重衰减）
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)

# 余弦退火学习率调度
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=50, T_mult=2, eta_min=1e-5
)

# 10. 训练循环（带早停和模型检查点）
epochs = 500
best_val_loss = float('inf')
patience = 30
patience_counter = 0

train_losses = []
val_losses = []

# 梯度裁剪
grad_clip_value = 1.0

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
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
        
        optimizer.step()
        train_loss += loss.item() * batch_X.size(0)
    
    train_loss /= len(train_loader.dataset)
    
    # 验证阶段
    model.eval()
    val_loss = 0
    val_predictions = []
    val_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_X.size(0)
            
            # 收集预测值和真实值
            val_predictions.extend(outputs.cpu().numpy())
            val_targets.extend(batch_y.cpu().numpy())
    
    val_loss /= len(val_loader.dataset)
    
    # 更新学习率
    scheduler.step()
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    # 计算验证集RMSE
    val_predictions = np.array(val_predictions).flatten()
    val_targets = np.array(val_targets).flatten()
    val_rmse = np.sqrt(np.mean((val_predictions - val_targets) ** 2))
    
    # 早停检查
    if val_rmse < best_val_loss:
        best_val_loss = val_rmse
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss,
        }, 'best_model.pth')
        best_val_predictions = val_predictions.copy()
        best_val_targets = val_targets.copy()
    else:
        patience_counter += 1
    
    if epoch % 50 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Val RMSE: {val_rmse:.4f}, LR: {current_lr:.6f}, Patience: {patience_counter}')
    
    if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch}')
        break

# 11. 加载最佳模型
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"加载最佳模型，验证集RMSE: {checkpoint['val_loss']:.4f}")

# 12. 计算最终验证分数
final_rmse = np.sqrt(np.mean((best_val_predictions - best_val_targets) ** 2))
score = 1.0 / (1.0 + final_rmse)

print(f'最终验证集RMSE: {final_rmse:.4f}')
print(f'Score = (1.0 / (1.0 + RMSE)) = {score:.4f}')

# 13. 在测试集上进行预测
test_predictions = []
model.eval()
with torch.no_grad():
    # 分批预测
    batch_size_test = 64
    for i in range(0, len(X_test_tensor), batch_size_test):
        batch = X_test_tensor[i:i+batch_size_test].to(device)
        preds = model(batch)
        test_predictions.extend(preds.cpu().numpy())

test_predictions = np.array(test_predictions).flatten()

# 14. 后处理
# 确保没有负值
test_predictions = np.maximum(test_predictions, 0)

# 轻微的平滑处理（移动平均）
window_size = 3
if len(test_predictions) > window_size:
    # 使用简单移动平均平滑预测
    smoothed = np.convolve(test_predictions, np.ones(window_size)/window_size, mode='valid')
    # 保持两端不变
    test_predictions[window_size-1:] = smoothed

# 15. 生成提交文件
submission = pd.DataFrame({
    'id': test_df['id'],
    'tested_positive': test_predictions
})

submission.to_csv('submission.csv', index=False)
print(f"提交文件已保存，包含 {len(submission)} 条预测结果")

# 16. 打印最终分数
print(f"Score= (1.0 / (1.0 + RMSE)) = {score:.4f}")

# 17. 绘制训练曲线（可选）
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training History')
plt.legend()
plt.grid(True)
plt.savefig('training_history.png')
print("训练历史图已保存为 training_history.png")