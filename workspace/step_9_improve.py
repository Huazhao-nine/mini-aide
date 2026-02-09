import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')
import joblib
from scipy import stats

# ========== 1. 数据加载与预处理 ==========
train_path = '/home/huazhao/DL-HW/ML-2025/mini-aide/input/train.csv'
test_path = '/home/huazhao/DL-HW/ML-2025/mini-aide/input/test.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# 提取目标列
target_col = 'tested_positive_day3'
y = train_df[target_col].values

# 获取特征列
feature_cols = [col for col in train_df.columns if col not in ['id', target_col]]
X_train_raw = train_df[feature_cols].copy()
X_test_raw = test_df[feature_cols].copy()

print(f"训练集形状: {X_train_raw.shape}, 测试集形状: {X_test_raw.shape}")

# ========== 2. 高级特征工程 ==========
def create_features(df):
    """创建时间序列特征和交互特征"""
    df = df.copy()
    
    # 为每个州创建特征（因为有35个州的独热编码）
    state_cols = ['AL', 'AZ', 'CA', 'CO', 'CT', 'FL', 'GA', 'IL', 'IN', 'IA', 'KS', 'KY', 
                  'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MO', 'NJ', 'NM', 'NY', 'NC', 'OH', 
                  'OK', 'OR', 'PA', 'SC', 'TN', 'TX', 'VA', 'WA', 'WV', 'WI']
    
    # 计算每个州的时间序列特征
    for i in range(1, 4):
        # 1. 症状相关特征的加权和
        df[f'symptoms_weighted_day{i}'] = (
            df[f'cli_day{i}'] * 0.5 + 
            df[f'ili_day{i}'] * 0.3 + 
            df[f'tested_positive_day{i}'] * 0.2
        )
        
        # 2. 防护行为的加权和
        df[f'protection_score_day{i}'] = (
            df[f'wearing_mask_7d_day{i}'] * 0.4 +
            df[f'wbelief_masking_effective_day{i}'] * 0.3 +
            df[f'wbelief_distancing_effective_day{i}'] * 0.3
        )
        
        # 3. 风险行为的加权和
        df[f'risk_score_day{i}'] = (
            df[f'wlarge_event_indoors_day{i}'] * 0.25 +
            df[f'wshop_indoors_day{i}'] * 0.25 +
            df[f'wrestaurant_indoors_day{i}'] * 0.25 +
            df[f'public_transit_day{i}'] * 0.25
        )
    
    # 跨时间的变化特征（关键！）
    for metric in ['cli', 'ili', 'wnohh_cmnty_cli', 'tested_positive', 
                   'symptoms_weighted', 'protection_score', 'risk_score']:
        if f'{metric}_day2' in df.columns and f'{metric}_day1' in df.columns:
            df[f'{metric}_change_1_to_2'] = df[f'{metric}_day2'] - df[f'{metric}_day1']
        if f'{metric}_day3' in df.columns and f'{metric}_day2' in df.columns:
            df[f'{metric}_change_2_to_3'] = df[f'{metric}_day3'] - df[f'{metric}_day2']
    
    # 交互特征
    for i in range(1, 4):
        df[f'cli_mask_interaction_day{i}'] = df[f'cli_day{i}'] * df[f'wearing_mask_7d_day{i}']
        df[f'tested_risk_interaction_day{i}'] = df[f'tested_positive_day{i}'] * df[f'risk_score_day{i}']
        df[f'symptoms_protection_interaction_day{i}'] = df[f'symptoms_weighted_day{i}'] * df[f'protection_score_day{i}']
    
    # 州级别的聚合统计（对数值特征）
    numeric_cols = [col for col in df.columns if any(x in col for x in ['day1', 'day2', 'day3', 'change', 'score', 'interaction'])]
    
    for col in numeric_cols:
        if col in df.columns:
            for state in state_cols:
                if state in df.columns:
                    df[f'{col}_state_{state}'] = df[col] * df[state]
    
    return df

# 应用特征工程
print("创建高级特征...")
X_train_enhanced = create_features(X_train_raw)
X_test_enhanced = create_features(X_test_raw)

print(f"特征工程后训练集形状: {X_train_enhanced.shape}")

# ========== 3. 分层时间序列划分 ==========
# 按州分组进行时间序列划分，确保每个州的数据完整性
state_cols = [col for col in X_train_enhanced.columns if col in 
              ['AL', 'AZ', 'CA', 'CO', 'CT', 'FL', 'GA', 'IL', 'IN', 'IA', 'KS', 'KY', 
               'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MO', 'NJ', 'NM', 'NY', 'NC', 'OH', 
               'OK', 'OR', 'PA', 'SC', 'TN', 'TX', 'VA', 'WA', 'WV', 'WI']]

# 根据州信息创建分组
def create_state_groups(X, y):
    """创建州级别的分组，用于分层采样"""
    groups = []
    for idx in range(len(X)):
        active_states = [state for state in state_cols if X[state].iloc[idx] == 1]
        if active_states:
            groups.append(active_states[0])
        else:
            groups.append('unknown')
    return groups

train_groups = create_state_groups(X_train_enhanced, y)

# 分层时间序列划分：每个州单独划分
train_indices, val_indices = [], []
val_ratio = 0.2

for state in set(train_groups):
    state_indices = [i for i, g in enumerate(train_groups) if g == state]
    if len(state_indices) > 10:  # 只对有足够数据的州进行划分
        split_idx = int(len(state_indices) * (1 - val_ratio))
        train_indices.extend(state_indices[:split_idx])
        val_indices.extend(state_indices[split_idx:])
    else:
        train_indices.extend(state_indices)

X_train_split = X_train_enhanced.iloc[train_indices].values
y_train_split = y[train_indices]
X_val_split = X_train_enhanced.iloc[val_indices].values
y_val_split = y[val_indices]

print(f"分层划分后 - 训练集: {X_train_split.shape}, 验证集: {X_val_split.shape}")

# ========== 4. 高级特征预处理 ==========
# 4.1 使用RobustScaler对异常值更鲁棒
scaler = RobustScaler(quantile_range=(10, 90))
X_train_scaled = scaler.fit_transform(X_train_split)
X_val_scaled = scaler.transform(X_val_split)
X_test_scaled = scaler.transform(X_test_enhanced.values)

# 4.2 基于随机森林的特征选择
print("进行基于模型的特征选择...")
rf_selector = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_selector.fit(X_train_scaled, y_train_split)

# 选择最重要的特征
importance_threshold = np.percentile(rf_selector.feature_importances_, 50)
selector_mask = rf_selector.feature_importances_ > importance_threshold

X_train_selected = X_train_scaled[:, selector_mask]
X_val_selected = X_val_scaled[:, selector_mask]
X_test_selected = X_test_scaled[:, selector_mask]

print(f"特征选择后维度: {X_train_selected.shape}")

# 4.3 保存特征名称用于分析
selected_feature_names = X_train_enhanced.columns[selector_mask]
print(f"\n前20个重要特征:")
for i, (name, imp) in enumerate(sorted(zip(selected_feature_names, rf_selector.feature_importances_[selector_mask]), 
                                       key=lambda x: x[1], reverse=True)[:20]):
    print(f"{i+1:2d}. {name:40s} - {imp:.6f}")

# ========== 5. 定义高级DNN模型 ==========
class AdvancedCOVIDPredictor(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.2):
        super(AdvancedCOVIDPredictor, self).__init__()
        
        # 更深的网络架构，带残差连接
        self.layer1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        
        self.layer2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.layer3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.layer4 = nn.Linear(64, 64)
        self.bn4 = nn.BatchNorm1d(64)
        
        self.layer5 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32)
        
        self.output = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 第一层
        out = self.layer1(x)
        out = self.bn1(out)
        out = self.elu(out)
        out = self.dropout(out)
        
        # 第二层带残差
        identity = out
        out = self.layer2(out)
        out = self.bn2(out)
        out = self.elu(out)
        out = self.dropout(out)
        out = out + identity  # 残差连接
        
        # 第三层
        out = self.layer3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # 第四层
        out = self.layer4(out)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # 第五层
        out = self.layer5(out)
        out = self.bn5(out)
        out = self.relu(out)
        
        # 输出层
        out = self.output(out)
        return out

# ========== 6. 训练设置与集成 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train_selected).to(device)
y_train_tensor = torch.FloatTensor(y_train_split).to(device).view(-1, 1)
X_val_tensor = torch.FloatTensor(X_val_selected).to(device)
y_val_tensor = torch.FloatTensor(y_val_split).to(device).view(-1, 1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

# 使用多个模型进行集成
def train_model(model_idx, X_train, y_train, X_val, y_val, epochs=150):
    """训练单个模型"""
    model = AdvancedCOVIDPredictor(X_train.shape[1], dropout_rate=0.2 + 0.05 * model_idx).to(device)
    
    # 不同学习率的优化器
    optimizer = optim.AdamW(model.parameters(), 
                          lr=0.001 * (0.9 ** model_idx),  # 递减学习率
                          weight_decay=0.01)
    
    # 带热重启的余弦退火调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-5
    )
    
    criterion = nn.HuberLoss(delta=2.0)  # 对异常值更鲁棒的损失函数
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 30
    
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val)
            val_loss = criterion(val_predictions, y_val).item()
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss)
        
        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'best_model_{model_idx}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"模型{model_idx}: 早停在epoch {epoch+1}")
                break
        
        if (epoch + 1) % 30 == 0:
            print(f'模型{model_idx} - Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {train_loss/len(train_loader):.4f}, '
                  f'Val Loss: {val_loss:.4f}, '
                  f'LR: {scheduler.get_last_lr()[0]:.6f}')
    
    # 加载最佳模型
    model.load_state_dict(torch.load(f'best_model_{model_idx}.pth'))
    return model, best_val_loss

# 训练多个模型集成
n_models = 5
models = []
val_scores = []

print(f"\n训练{n_models}个模型进行集成...")
for i in range(n_models):
    print(f"\n=== 训练模型 {i+1}/{n_models} ===")
    model, val_loss = train_model(i, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, epochs=200)
    models.append(model)
    val_scores.append(val_loss)

# 计算模型权重（基于验证损失）
val_scores = np.array(val_scores)
model_weights = 1.0 / (val_scores + 1e-10)
model_weights = model_weights / model_weights.sum()

print(f"\n模型权重: {model_weights}")

# ========== 7. 在验证集上评估集成模型 ==========
def ensemble_predict(models, X, weights=None):
    """集成模型的预测"""
    if weights is None:
        weights = np.ones(len(models)) / len(models)
    
    predictions = []
    for model, weight in zip(models, weights):
        model.eval()
        with torch.no_grad():
            pred = model(X).cpu().numpy() * weight
            predictions.append(pred)
    
    return np.sum(predictions, axis=0)

# 验证集评估
val_predictions_ensemble = ensemble_predict(models, X_val_tensor, model_weights)
val_rmse = np.sqrt(np.mean((val_predictions_ensemble - y_val_tensor.cpu().numpy()) ** 2))

# 计算Score
score = 1.0 / (1.0 + val_rmse)
print(f"\n{'='*50}')
print(f"集成模型性能:")
print(f"验证集RMSE: {val_rmse:.6f}")
print(f"Score = (1.0 / (1.0 + RMSE)) = {score:.6f}")
print(f"{'='*50}")

# 单个模型评估
print(f"\n单个模型性能:")
for i, model in enumerate(models):
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_tensor)
        val_rmse_single = np.sqrt(np.mean((val_pred.cpu().numpy() - y_val_tensor.cpu().numpy()) ** 2))
        print(f"模型{i+1}: RMSE = {val_rmse_single:.6f}, Score = {1.0/(1.0+val_rmse_single):.6f}")

# ========== 8. 对测试集进行预测 ==========
X_test_tensor = torch.FloatTensor(X_test_selected).to(device)

# 集成预测
test_predictions_ensemble = ensemble_predict(models, X_test_tensor, model_weights)

# 后处理：确保预测值非负
test_predictions_ensemble = np.maximum(test_predictions_ensemble, 0)

# ========== 9. 生成提交文件 ==========
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'tested_positive': test_predictions_ensemble.flatten()
})

# 添加简单的后处理平滑
def smooth_predictions(predictions, window=3):
    """对预测结果进行平滑处理"""
    smoothed = np.copy(predictions)
    for i in range(len(predictions)):
        start = max(0, i - window // 2)
        end = min(len(predictions), i + window // 2 + 1)
        smoothed[i] = np.mean(predictions[start:end])
    return smoothed

submission_df['tested_positive'] = smooth_predictions(submission_df['tested_positive'].values)

submission_df.to_csv('submission_ensemble.csv', index=False)
print(f"\n提交文件已保存: submission_ensemble.csv")
print(f"提交文件形状: {submission_df.shape}")
print(f"预测值统计:")
print(f"  最小值: {submission_df['tested_positive'].min():.4f}")
print(f"  最大值: {submission_df['tested_positive'].max():.4f}")
print(f"  均值: {submission_df['tested_positive'].mean():.4f}")
print(f"  中位数: {submission_df['tested_positive'].median():.4f}")

# 保存特征重要性
feature_importance_df = pd.DataFrame({
    'feature': selected_feature_names,
    'importance': rf_selector.feature_importances_[selector_mask]
})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
feature_importance_df.to_csv('feature_importance.csv', index=False)
print(f"\n特征重要性已保存到: feature_importance.csv")