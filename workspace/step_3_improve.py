import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
import os

# 设置随机种子确保可重复性
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ========== 1. 数据加载与预处理 ==========
train_path = '/home/huazhao/DL-HW/ML-2025/mini-aide/input/train.csv'
test_path = '/home/huazhao/DL-HW/ML-2025/mini-aide/input/test.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print(f"原始训练集形状: {train_df.shape}, 测试集形状: {test_df.shape}")

# 提取目标列
target_col = 'tested_positive_day3'
y = train_df[target_col].values

# ========== 2. 高级特征工程 ==========
def create_features(df):
    """创建增强的特征集"""
    df = df.copy()
    
    # 保留原始特征列（排除id）
    if 'id' in df.columns:
        features_df = df.drop(['id'], axis=1)
    else:
        features_df = df.copy()
    
    # 如果是训练数据，排除目标列
    if target_col in features_df.columns:
        features_df = features_df.drop([target_col], axis=1)
    
    # 创建新的特征列
    new_features = pd.DataFrame(index=df.index)
    
    # 1. 时序特征：创建day1-day3的变化率
    for prefix in ['cli', 'ili', 'wnohh_cmnty_cli', 'tested_positive', 
                   'wearing_mask_7d', 'public_transit', 'worried_finances']:
        for day in ['1', '2', '3']:
            col_name = f'{prefix}_day{day}'
            if col_name in df.columns:
                new_features[f'{col_name}_orig'] = df[col_name]
    
    # 2. 创建变化率特征 (day2/day1, day3/day2)
    for prefix in ['cli', 'ili', 'wnohh_cmnty_cli', 'tested_positive']:
        for day1, day2 in [('1', '2'), ('2', '3')]:
            col1 = f'{prefix}_day{day1}'
            col2 = f'{prefix}_day{day2}'
            if col1 in df.columns and col2 in df.columns:
                # 变化率（加1避免除零）
                new_features[f'{prefix}_change_{day1}_to_{day2}'] = (
                    df[col2] - df[col1]) / (df[col1] + 1)
    
    # 3. 创建交互特征
    for day in ['1', '2', '3']:
        # cli与mask的交互
        cli_col = f'cli_day{day}'
        mask_col = f'wearing_mask_7d_day{day}'
        if cli_col in df.columns and mask_col in df.columns:
            new_features[f'cli_mask_interaction_day{day}'] = df[cli_col] * df[mask_col]
        
        # tested_positive与worried_finances的交互
        tp_col = f'tested_positive_day{day}'
        worried_col = f'worried_finances_day{day}'
        if tp_col in df.columns and worried_col in df.columns:
            new_features[f'tp_worried_interaction_day{day}'] = df[tp_col] * df[worried_col]
    
    # 4. 州特征的统计信息（保留原始州特征）
    state_columns = ['AL', 'AZ', 'CA', 'CO', 'CT', 'FL', 'GA', 'IL', 'IN', 'IA', 
                     'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MO', 'NJ', 
                     'NM', 'NY', 'NC', 'OH', 'OK', 'OR', 'PA', 'SC', 'TN', 'TX', 
                     'VA', 'WA', 'WV', 'WI']
    
    existing_states = [col for col in state_columns if col in df.columns]
    if existing_states:
        state_data = df[existing_states]
        new_features['state_count'] = state_data.sum(axis=1)  # 活跃州数量
        new_features['state_std'] = state_data.std(axis=1)   # 州特征的方差
    
    # 5. 信念特征的聚合
    belief_prefixes = ['wbelief_masking_effective', 'wbelief_distancing_effective',
                       'wcovid_vaccinated_friends', 'wothers_masked_public',
                       'wothers_distanced_public', 'wworried_catch_covid']
    
    for day in ['1', '2', '3']:
        belief_cols = [f'{prefix}_day{day}' for prefix in belief_prefixes 
                      if f'{prefix}_day{day}' in df.columns]
        if belief_cols:
            belief_data = df[belief_cols]
            new_features[f'belief_mean_day{day}'] = belief_data.mean(axis=1)
            new_features[f'belief_std_day{day}'] = belief_data.std(axis=1)
    
    # 6. 活动参与度特征
    activity_prefixes = ['wshop_indoors', 'wrestaurant_indoors', 'wlarge_event_indoors']
    for day in ['1', '2', '3']:
        activity_cols = [f'{prefix}_day{day}' for prefix in activity_prefixes 
                        if f'{prefix}_day{day}' in df.columns]
        if activity_cols:
            activity_data = df[activity_cols]
            new_features[f'activity_sum_day{day}'] = activity_data.sum(axis=1)
            new_features[f'activity_mean_day{day}'] = activity_data.mean(axis=1)
    
    # 合并所有特征
    # 首先添加原始特征
    final_features = pd.concat([features_df, new_features], axis=1)
    
    # 填充可能的NaN值
    final_features = final_features.fillna(final_features.mean())
    
    return final_features

# 创建增强特征
X_train_raw = create_features(train_df)
X_test_raw = create_features(test_df)

print(f"特征工程后 - 训练特征形状: {X_train_raw.shape}, 测试特征形状: {X_test_raw.shape}")

# ========== 3. 时间序列划分验证集 ==========
# 取最后20%作为验证集，不shuffle
val_size = int(0.2 * len(X_train_raw))
X_train_split = X_train_raw.values[:-val_size]
y_train_split = y[:-val_size]
X_val_split = X_train_raw.values[-val_size:]
y_val_split = y[-val_size:]

print(f"训练集分割后: {X_train_split.shape}, 验证集: {X_val_split.shape}")

# ========== 4. 数据预处理 ==========
# 4.1 使用RobustScaler处理异常值
scaler = RobustScaler(quantile_range=(5, 95))  # 对异常值更鲁棒
X_train_scaled = scaler.fit_transform(X_train_split)
X_val_scaled = scaler.transform(X_val_split)
X_test_scaled = scaler.transform(X_test_raw.values)

# 4.2 对目标值进行log1p变换，使分布更接近正态分布
y_train_log = np.log1p(y_train_split)
y_val_log = np.log1p(y_val_split)

# 4.3 特征选择 - 使用互信息方法，选择更多特征
selector = SelectKBest(score_func=mutual_info_regression, k=min(50, X_train_scaled.shape[1]))
X_train_selected = selector.fit_transform(X_train_scaled, y_train_log)
X_val_selected = selector.transform(X_val_scaled)
X_test_selected = selector.transform(X_test_scaled)

print(f"特征选择后维度: {X_train_selected.shape}")

# 4.4 PCA降维去除冗余
if X_train_selected.shape[1] > 20:
    pca = PCA(n_components=0.95)  # 保留95%的方差
    X_train_final = pca.fit_transform(X_train_selected)
    X_val_final = pca.transform(X_val_selected)
    X_test_final = pca.transform(X_test_selected)
    print(f"PCA降维后维度: {X_train_final.shape}")
else:
    X_train_final = X_train_selected
    X_val_final = X_val_selected
    X_test_final = X_test_selected

# ========== 5. 转换为PyTorch张量 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

X_train_tensor = torch.FloatTensor(X_train_final).to(device)
y_train_tensor = torch.FloatTensor(y_train_log).to(device).view(-1, 1)
X_val_tensor = torch.FloatTensor(X_val_final).to(device)
y_val_tensor = torch.FloatTensor(y_val_log).to(device).view(-1, 1)

# 创建数据加载器 - 注意：训练时不shuffle以保持时序特性
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)  # 时序任务不shuffle

# ========== 6. 定义增强的DNN模型 ==========
class EnhancedCOVIDPredictor(nn.Module):
    def __init__(self, input_dim):
        super(EnhancedCOVIDPredictor, self).__init__()
        
        # 更深的网络结构
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.25),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            
            nn.Linear(16, 1)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

model = EnhancedCOVIDPredictor(X_train_final.shape[1]).to(device)
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# ========== 7. 训练设置 ==========
criterion = nn.MSELoss()
# 使用AdamW优化器，带权重衰减
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# 更复杂的学习率调度器
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=0.01,
    steps_per_epoch=len(train_loader),
    epochs=200,
    pct_start=0.1
)

# ========== 8. 训练循环（带早停） ==========
epochs = 200
best_val_loss = float('inf')
patience = 20
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
        scheduler.step()
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # 验证阶段
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val_tensor)
        val_loss = criterion(val_predictions, y_val_tensor).item()
        val_losses.append(val_loss)
    
    # 早停机制
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'✅ Epoch [{epoch+1}/{epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f} (最佳)')
    else:
        patience_counter += 1
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}')
    
    if patience_counter >= patience:
        print(f'早停在epoch {epoch+1}, 最佳验证损失: {best_val_loss:.4f}')
        break

# ========== 9. 模型集成：创建多个模型提升稳定性 ==========
def create_ensemble_predictions(X, num_models=5):
    """创建模型集成预测"""
    predictions_list = []
    
    for i in range(num_models):
        # 为每个模型设置不同的随机种子
        set_seed(42 + i)
        
        # 创建并训练新模型
        model_i = EnhancedCOVIDPredictor(X_train_final.shape[1]).to(device)
        optimizer_i = optim.AdamW(model_i.parameters(), lr=0.001)
        
        # 简化的训练循环
        for epoch in range(50):
            model_i.train()
            for batch_X, batch_y in train_loader:
                optimizer_i.zero_grad()
                predictions = model_i(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer_i.step()
        
        # 预测
        model_i.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            pred = model_i(X_tensor).cpu().numpy()
            predictions_list.append(pred)
    
    # 返回集成预测（平均）
    return np.mean(predictions_list, axis=0)

# ========== 10. 在验证集上评估 ==========
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

with torch.no_grad():
    # 主模型预测
    val_predictions_log = model(X_val_tensor).cpu().numpy()
    
    # 集成模型预测
    val_predictions_ensemble_log = create_ensemble_predictions(X_val_final, num_models=3)
    
    # 加权集成 (主模型权重更高)
    val_predictions_final_log = 0.7 * val_predictions_log + 0.3 * val_predictions_ensemble_log
    
    # 将log预测转换回原始尺度
    val_predictions = np.expm1(val_predictions_final_log)
    
    # 计算指标
    val_rmse = np.sqrt(np.mean((val_predictions - y_val_split) ** 2))
    val_mae = np.mean(np.abs(val_predictions - y_val_split))

# 计算Score
score = 1.0 / (1.0 + val_rmse)
print(f"\n{'='*50}")
print(f"验证集评估结果:")
print(f"RMSE: {val_rmse:.6f}")
print(f"MAE: {val_mae:.6f}")
print(f"Score: {score:.6f}")
print(f"Score = (1.0 / (1.0 + RMSE)) = {score:.6f}")
print(f"{'='*50}")

# ========== 11. 对测试集进行预测 ==========
# 主模型预测
X_test_tensor = torch.FloatTensor(X_test_final).to(device)
model.eval()
with torch.no_grad():
    test_predictions_log = model(X_test_tensor).cpu().numpy()

# 集成模型预测
test_predictions_ensemble_log = create_ensemble_predictions(X_test_final, num_models=3)

# 加权集成
test_predictions_final_log = 0.7 * test_predictions_log + 0.3 * test_predictions_ensemble_log

# 转换回原始尺度
test_predictions = np.expm1(test_predictions_final_log)

# 后处理：确保非负预测值
test_predictions = np.clip(test_predictions, 0, None)

# ========== 12. 生成提交文件 ==========
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'tested_positive': test_predictions.flatten()
})
submission_df.to_csv('submission.csv', index=False)
print(f"\n提交文件已保存，形状: {submission_df.shape}")

# ========== 13. 保存重要特征信息 ==========
# 获取特征重要性
feature_importance = pd.DataFrame({
    'feature': range(X_train_final.shape[1]),
    'importance': selector.scores_[:X_train_final.shape[1]]
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print(f"\nTop 10重要特征索引: {feature_importance['feature'].head(10).tolist()}")

# 保存训练曲线
if len(train_losses) > 0 and len(val_losses) > 0:
    train_curve = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    train_curve.to_csv('training_curve.csv', index=False)
    print("训练曲线已保存到 training_curve.csv")