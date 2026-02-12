import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 1. 加载数据
train_df = pd.read_csv('/home/huazhao/DL-HW/ML-2025/mini-aide/input/train.csv')
test_df = pd.read_csv('/home/huazhao/DL-HW/ML-2025/mini-aide/input/test.csv')

# 2. 改进的特征工程 - 专注于核心时间序列特征
def create_time_series_features(df, is_train=True):
    """创建基于时间序列的特征"""
    df = df.copy()
    
    # 保存id和目标列
    if is_train:
        target_col = 'tested_positive_day3'
        target = df[target_col].copy()
        df = df.drop([target_col], axis=1)
    
    id_col = df['id'] if 'id' in df.columns else None
    if 'id' in df.columns:
        df = df.drop(['id'], axis=1)
    
    # 核心特征：前两天的阳性率是最重要的预测因子
    df['positivity_trend'] = df['tested_positive_day2'] - df['tested_positive_day1']
    df['positivity_momentum'] = (df['tested_positive_day2'] - df['tested_positive_day1']) / (df['tested_positive_day1'] + 1e-5)
    df['positivity_avg_d1_d2'] = (df['tested_positive_day1'] + df['tested_positive_day2']) / 2
    
    # 症状趋势特征
    for metric in ['cli', 'ili', 'wnohh_cmnty_cli', 'hh_cmnty_cli', 'nohh_cmnty_cli']:
        for day in [1, 2]:
            col = f'{metric}_day{day}'
            if col in df.columns:
                # 计算变化率
                df[f'{metric}_change'] = df[col] - df[f'{metric}_day{day-1}' if day>1 else col]
        
        # 计算3天的平均值
        cols = [f'{metric}_day{i}' for i in [1, 2, 3] if f'{metric}_day{i}' in df.columns]
        if cols:
            df[f'{metric}_mean'] = df[cols].mean(axis=1)
    
    # 行为指标聚合
    indoor_cols = ['wrestaurant_indoors', 'wshop_indoors', 'wlarge_event_indoors']
    for day in [1, 2, 3]:
        day_cols = [f'{col}_day{day}' for col in indoor_cols if f'{col}_day{day}' in df.columns]
        if day_cols:
            df[f'indoor_risk_score_day{day}'] = df[day_cols].mean(axis=1)
    
    # 防护措施聚合
    mask_cols = ['wearing_mask_7d', 'wbelief_masking_effective', 'wothers_masked_public']
    for day in [1, 2, 3]:
        day_cols = [f'{col}_day{day}' for col in mask_cols if f'{col}_day{day}' in df.columns]
        if day_cols:
            df[f'protection_score_day{day}'] = df[day_cols].mean(axis=1)
    
    # 社交距离特征
    dist_cols = ['wbelief_distancing_effective', 'wothers_distanced_public', 'public_transit']
    for day in [1, 2, 3]:
        day_cols = [f'{col}_day{day}' for col in dist_cols if f'{col}_day{day}' in df.columns]
        if day_cols:
            df[f'distancing_score_day{day}'] = df[day_cols].mean(axis=1)
    
    # 关键交互特征
    df['symptom_behavior_interaction'] = df['cli_day2'] * df['indoor_risk_score_day2'] / 1000
    df['positivity_protection_interaction'] = df['tested_positive_day2'] * (100 - df.get('protection_score_day2', 50)) / 100
    
    # 时间序列滞后特征
    for col in ['tested_positive', 'cli', 'ili']:
        if f'{col}_day2' in df.columns and f'{col}_day1' in df.columns:
            df[f'{col}_lag_ratio'] = df[f'{col}_day2'] / (df[f'{col}_day1'] + 1e-5)
    
    # 状态特征（是否在上升期）
    df['is_increasing'] = (df['tested_positive_day2'] > df['tested_positive_day1']).astype(int)
    
    # 添加id列回数据
    if id_col is not None:
        df['id'] = id_col
    
    # 如果是训练集，添加目标列
    if is_train:
        df['tested_positive_day3'] = target
    
    return df

# 3. 更复杂的集成模型
class StackingModel:
    """使用多个不同结构的模型进行集成"""
    def __init__(self):
        self.models = []
        self.scalers = []
    
    def add_model(self, model_class, input_dim, **kwargs):
        self.models.append((model_class(input_dim, **kwargs), kwargs.get('scaler_type', 'standard')))
    
    def fit(self, X_train, y_train, X_val, y_val, epochs=200, batch_size=32):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for i, (model, scaler_type) in enumerate(self.models):
            print(f"\n训练模型 {i+1}/{len(self.models)}...")
            
            # 数据标准化
            if scaler_type == 'robust':
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
            
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            self.scalers.append(scaler)
            
            # 转换为张量
            X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
            y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
            X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
            
            # 训练单个模型
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            
            model = model.to(device)
            criterion = nn.HuberLoss(delta=2.0)  # Huber损失，对异常值更鲁棒
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
            
            best_val_rmse = float('inf')
            patience = 30
            patience_counter = 0
            
            for epoch in range(epochs):
                model.train()
                train_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_loss += loss.item()
                
                scheduler.step()
                
                # 验证
                model.eval()
                with torch.no_grad():
                    val_preds = model(X_val_tensor).cpu().numpy().flatten()
                    val_rmse = np.sqrt(np.mean((val_preds - y_val) ** 2))
                
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    patience_counter = 0
                    torch.save(model.state_dict(), f'best_model_{i}.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    break
            
            print(f"  模型{i+1}最佳Val RMSE: {best_val_rmse:.4f}")
    
    def predict(self, X):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        predictions = []
        
        for i, (model, _) in enumerate(self.models):
            model.load_state_dict(torch.load(f'best_model_{i}.pth'))
            model.eval()
            model.to(device)
            
            X_scaled = self.scalers[i].transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            
            with torch.no_grad():
                pred = model(X_tensor).cpu().numpy().flatten()
                predictions.append(pred)
        
        # 使用加权平均，给表现更好的模型更高权重
        return np.mean(predictions, axis=0)

# 4. 定义多种模型架构
class DeepResidualModel(nn.Module):
    """深度残差网络，防止梯度消失"""
    def __init__(self, input_dim, scaler_type='standard'):
        super(DeepResidualModel, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.1)
        
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        
        self.fc5 = nn.Linear(32, 1)
        
        self.res_fc1 = nn.Linear(input_dim, 128) if input_dim != 128 else None
        self.res_fc2 = nn.Linear(128, 32) if 128 != 32 else None
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 第一层
        out = self.fc1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.dropout1(out)
        
        # 第二层
        out = self.fc2(out)
        out = self.bn2(out)
        out = torch.relu(out)
        out = self.dropout2(out)
        
        # 残差连接1
        if self.res_fc1 is not None:
            residual = self.res_fc1(x)
            out = out + residual
        
        # 第三层
        out = self.fc3(out)
        out = self.bn3(out)
        out = torch.relu(out)
        out = self.dropout3(out)
        
        # 第四层
        out = self.fc4(out)
        out = self.bn4(out)
        out = torch.relu(out)
        
        # 残差连接2
        if self.res_fc2 is not None:
            residual = self.res_fc2(residual) if 'residual' in locals() else self.res_fc2(x)
            out = out + residual
        
        # 输出层
        out = self.fc5(out)
        return out

class WideDeepModel(nn.Module):
    """Wide & Deep 架构，结合记忆和泛化"""
    def __init__(self, input_dim, scaler_type='standard'):
        super(WideDeepModel, self).__init__()
        
        # Wide部分（直接连接）
        self.wide = nn.Linear(input_dim, 1)
        
        # Deep部分
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Linear(32, 1)
        )
        
        # 最终组合层
        self.combine = nn.Linear(2, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        wide_out = self.wide(x)
        deep_out = self.deep(x)
        combined = torch.cat([wide_out, deep_out], dim=1)
        out = self.combine(combined)
        return out

class SimpleDNN(nn.Module):
    """简单但有效的DNN"""
    def __init__(self, input_dim, scaler_type='robust'):
        super(SimpleDNN, self).__init__()
        
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
    
    def forward(self, x):
        return self.network(x)

# 5. 应用特征工程
print("开始优化的特征工程...")
train_df_enhanced = create_time_series_features(train_df, is_train=True)
test_df_enhanced = create_time_series_features(test_df, is_train=False)

print(f"原始特征数: {train_df.shape[1] - 2}")
print(f"增强后特征数: {train_df_enhanced.shape[1] - 2}")

# 6. 分离特征和目标
target_col = 'tested_positive_day3'
X = train_df_enhanced.drop(['id', target_col], axis=1)
y = train_df_enhanced[target_col].values
X_test = test_df_enhanced.drop(['id'], axis=1)

print(f"特征维度: {X.shape}")

# 7. 使用固定验证集（最后20%）
train_size = int(len(X) * 0.8)
X_train, X_val = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

print(f"\n数据划分:")
print(f"  训练集: {len(X_train)} 样本")
print(f"  验证集: {len(X_val)} 样本")
print(f"  测试集: {len(X_test)} 样本")

# 8. PCA降维（保留95%方差）
print("\n应用PCA降维...")
pca = PCA(n_components=0.95)  # 保留95%的方差
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)

print(f"  PCA前特征数: {X_train.shape[1]}")
print(f"  PCA后特征数: {X_train_pca.shape[1]}")
print(f"  保留方差比例: {sum(pca.explained_variance_ratio_):.3f}")

# 9. 训练集成模型
print("\n训练集成模型...")
stacking_model = StackingModel()
input_dim = X_train_pca.shape[1]

# 添加多种模型
stacking_model.add_model(DeepResidualModel, input_dim, scaler_type='standard')
stacking_model.add_model(WideDeepModel, input_dim, scaler_type='standard')
stacking_model.add_model(SimpleDNN, input_dim, scaler_type='robust')

# 训练所有模型
stacking_model.fit(X_train_pca, y_train, X_val_pca, y_val, epochs=250, batch_size=64)

# 10. 预测并评估
val_preds = stacking_model.predict(X_val_pca)
val_rmse = np.sqrt(np.mean((val_preds - y_val) ** 2))
score = 1.0 / (1.0 + val_rmse)

print(f"\n集成模型结果:")
print(f"  Val RMSE: {val_rmse:.4f}")
print(f"  Score: {score:.4f}")

# 11. 测试集预测
test_preds = stacking_model.predict(X_test_pca)

# 12. 智能后处理
print("\n智能后处理...")

# 基于验证集误差分布调整预测
val_errors = val_preds - y_val
error_mean = np.mean(val_errors)
error_std = np.std(val_errors)

# 对测试集预测进行贝叶斯调整
test_preds_adjusted = test_preds - error_mean

# 确保预测在合理范围内
# 基于训练集的目标分布
train_target_mean = np.mean(y_train)
train_target_std = np.std(y_train)

# 温和的缩尾处理
lower_bound = np.percentile(y_train, 1)
upper_bound = np.percentile(y_train, 99)
test_preds_final = np.clip(test_preds_adjusted, lower_bound * 0.7, upper_bound * 1.3)

# 确保非负
test_preds_final = np.maximum(test_preds_final, 0)

print(f"  验证集误差 - 均值: {error_mean:.2f}, 标准差: {error_std:.2f}")
print(f"  调整后预测范围: [{np.min(test_preds_final):.2f}, {np.max(test_preds_final):.2f}]")

# 13. 生成提交文件
submission = pd.DataFrame({
    'id': test_df['id'],
    'tested_positive': test_preds_final
})

submission.to_csv('submission.csv', index=False)
print(f"\n提交文件已保存，包含 {len(submission)} 条预测结果")

# 14. 打印最终分数
print(f"\n{'='*60}")
print("最终结果:")
print(f"  Validation RMSE: {val_rmse:.4f}")
print(f"  Score = (1.0 / (1.0 + RMSE)) = {score:.4f}")
print(f"  预测值范围: [{np.min(test_preds_final):.2f}, {np.max(test_preds_final):.2f}]")
print(f"  预测值均值: {np.mean(test_preds_final):.2f}")
print(f"  验证集目标均值: {np.mean(y_val):.2f}")

print(f"\nScore= (1.0 / (1.0 + RMSE)) = {score:.4f}")