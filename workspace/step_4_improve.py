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

# 设置随机种子以保证可重复性
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42) if torch.cuda.is_available() else None

# 1. 数据加载与清洗
print("Loading data...")
train_df = pd.read_csv('/home/huazhao/DL-HW/ML-2025/mini-aide/input/train.csv')
test_df = pd.read_csv('/home/huazhao/DL-HW/ML-2025/mini-aide/input/test.csv')

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# 分离特征和目标
target_col = 'tested_positive_day3'
y_train_full = train_df[target_col].values.astype(np.float32)
X_train_full = train_df.drop(columns=[target_col, 'id']).copy()
X_test = test_df.drop(columns=['id']).copy()

print(f"Original features: {X_train_full.shape[1]}")

# 2. 增强的特征工程
def enhanced_feature_engineering(df):
    df = df.copy()
    
    # 基础交互特征
    for day in ['1', '2', '3']:
        if f'cli_day{day}' in df.columns:
            # 症状与防护交互
            if f'wearing_mask_7d_day{day}' in df.columns:
                df[f'cli_mask_ratio_day{day}'] = df[f'cli_day{day}'] / (df[f'wearing_mask_7d_day{day}'] + 1e-5)
                df[f'cli_mask_interaction_day{day}'] = df[f'cli_day{day}'] * df[f'wearing_mask_7d_day{day}']
            
            # 症状与担忧交互
            if f'wworried_catch_covid_day{day}' in df.columns:
                df[f'cli_worry_day{day}'] = df[f'cli_day{day}'] * df[f'wworried_catch_covid_day{day}']
    
    # 时间序列特征 - 差值特征
    for feature_prefix in ['cli', 'ili', 'tested_positive', 'wearing_mask_7d']:
        day1_col = f'{feature_prefix}_day1'
        day2_col = f'{feature_prefix}_day2'
        day3_col = f'{feature_prefix}_day3' if f'{feature_prefix}_day3' in df.columns else None
        
        if day1_col in df.columns and day2_col in df.columns:
            df[f'{feature_prefix}_diff_2_1'] = df[day2_col] - df[day1_col]
            
            # 增长率
            df[f'{feature_prefix}_growth_2_1'] = (df[day2_col] - df[day1_col]) / (df[day1_col].abs() + 1e-5)
            
            if day3_col and day3_col in df.columns:
                df[f'{feature_prefix}_diff_3_2'] = df[day3_col] - df[day2_col]
                df[f'{feature_prefix}_avg_growth'] = ((df[day2_col] - df[day1_col]) + (df[day3_col] - df[day2_col])) / 2
    
    # 创建聚合特征 - 各天的平均值
    for feature_prefix in ['cli', 'ili', 'wnohh_cmnty_cli', 'wearing_mask_7d']:
        day_cols = [f'{feature_prefix}_day{i}' for i in [1, 2, 3] if f'{feature_prefix}_day{i}' in df.columns]
        if len(day_cols) >= 2:
            df[f'{feature_prefix}_mean'] = df[day_cols].mean(axis=1)
            df[f'{feature_prefix}_std'] = df[day_cols].std(axis=1)
            if len(day_cols) == 3:
                df[f'{feature_prefix}_trend'] = df[f'{feature_prefix}_day3'] - df[f'{feature_prefix}_day1']
    
    # 州相关特征 - 将州one-hot转换为更有意义的特征
    state_cols = ['AL', 'AZ', 'CA', 'CO', 'CT', 'FL', 'GA', 'IL', 'IN', 'IA', 'KS', 'KY', 
                  'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MO', 'NJ', 'NM', 'NY', 'NC', 'OH', 
                  'OK', 'OR', 'PA', 'SC', 'TN', 'TX', 'VA', 'WA', 'WV', 'WI']
    
    existing_state_cols = [col for col in state_cols if col in df.columns]
    if existing_state_cols:
        df['state_count'] = df[existing_state_cols].sum(axis=1)
    
    # 创建多项式特征（简单的平方特征）
    important_features = ['cli_day1', 'ili_day1', 'tested_positive_day1', 'wearing_mask_7d_day1']
    for feat in important_features:
        if feat in df.columns:
            df[f'{feat}_squared'] = df[feat] ** 2
            df[f'{feat}_sqrt'] = np.sqrt(np.abs(df[feat]) + 1e-5)
    
    # 创建综合风险评分
    if all(col in df.columns for col in ['cli_day1', 'wearing_mask_7d_day1', 'wworried_catch_covid_day1']):
        df['risk_score'] = (df['cli_day1'] * df['wworried_catch_covid_day1']) / (df['wearing_mask_7d_day1'] + 1)
    
    # 处理可能的缺失值（用中位数填充）
    for col in df.columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    return df

print("Performing enhanced feature engineering...")
X_train_full = enhanced_feature_engineering(X_train_full)
X_test = enhanced_feature_engineering(X_test)

print(f"Features after enhanced engineering: {X_train_full.shape[1]}")

# 3. 时间序列分割（最后20%作为验证集，不打乱）
val_size = int(len(X_train_full) * 0.2)
X_train = X_train_full.iloc[:-val_size].copy()
X_val = X_train_full.iloc[-val_size:].copy()
y_train = y_train_full[:-val_size]
y_val = y_train_full[-val_size:]

print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

# 4. 数据标准化（使用RobustScaler处理异常值）
scaler = RobustScaler(quantile_range=(25, 75))
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 5. 特征选择（使用多种方法组合）
print("Performing feature selection...")

# 方法1：基于相关性的特征选择
correlation_threshold = 0.05
feature_names = X_train.columns.tolist()

# 计算特征与目标的相关性（仅对训练集）
train_features_for_corr = pd.DataFrame(X_train_scaled, columns=feature_names)
train_features_for_corr['target'] = y_train

correlations = train_features_for_corr.corr()['target'].abs().sort_values(ascending=False)
high_corr_features = correlations[correlations > correlation_threshold].index.tolist()
high_corr_features.remove('target')  # 移除目标列

print(f"High correlation features ({correlation_threshold} threshold): {len(high_corr_features)}")

# 方法2：使用SelectKBest选择最佳特征
if len(high_corr_features) > 20:
    selector = SelectKBest(score_func=f_regression, k=min(50, len(high_corr_features)))
    
    # 只对高相关性特征进行选择
    high_corr_indices = [feature_names.index(f) for f in high_corr_features]
    X_train_high_corr = X_train_scaled[:, high_corr_indices]
    
    selector.fit(X_train_high_corr, y_train)
    selected_mask = selector.get_support()
    selected_features = [high_corr_features[i] for i in range(len(high_corr_features)) if selected_mask[i]]
else:
    selected_features = high_corr_features

print(f"Final selected features: {len(selected_features)}")

# 获取选定特征的索引
selected_indices = [feature_names.index(f) for f in selected_features if f in feature_names]

X_train_selected = X_train_scaled[:, selected_indices]
X_val_selected = X_val_scaled[:, selected_indices]
X_test_selected = X_test_scaled[:, selected_indices]

# 6. 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train_selected)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
X_val_tensor = torch.FloatTensor(X_val_selected)
y_val_tensor = torch.FloatTensor(y_val).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_selected)

# 7. 创建数据加载器（训练时打乱，验证时不打乱）
batch_size = 128  # 增加批大小以稳定训练
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 8. 定义改进的DNN模型（带残差连接）
class EnhancedCOVIDPredictor(nn.Module):
    def __init__(self, input_dim):
        super(EnhancedCOVIDPredictor, self).__init__()
        
        # 第一层
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 第二层
        self.layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        
        # 第三层（残差块）
        self.residual_block = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128)
        )
        self.residual_activation = nn.ReLU()
        
        # 第四层
        self.layer3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15)
        )
        
        # 第五层
        self.layer4 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 输出层
        self.output_layer = nn.Linear(32, 1)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        
        # 残差连接
        residual = x
        x = self.residual_block(x)
        x = self.residual_activation(x + residual)
        
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.output_layer(x)
        return x

# 9. 初始化模型、损失函数和优化器
input_dim = X_train_selected.shape[1]
model = EnhancedCOVIDPredictor(input_dim)
print(f"Model architecture:\n{model}")
print(f"Input dimension: {input_dim}")

# 使用更好的损失函数（Smooth L1 Loss对异常值更鲁棒）
criterion = nn.SmoothL1Loss(beta=1.0)  # 当误差小于beta时使用L2，大于时使用L1

# 使用AdamW优化器，带权重衰减
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)

# 学习率调度器：预热 + 余弦退火
def warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, base_lr):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # 线性预热
            return (epoch + 1) / warmup_epochs
        else:
            # 余弦退火
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scheduler = warmup_cosine_scheduler(optimizer, warmup_epochs=10, total_epochs=200, base_lr=5e-4)

# 10. 训练模型
num_epochs = 200
best_val_loss = float('inf')
best_val_rmse = float('inf')
patience = 20
patience_counter = 0

# 添加梯度裁剪
grad_clip_value = 1.0

print("\nStarting enhanced training...")
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
        
        optimizer.step()
        train_loss += loss.item()
    
    # 更新学习率
    scheduler.step()
    
    # 验证阶段
    model.eval()
    val_loss = 0
    val_predictions_all = []
    val_targets_all = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            val_loss += loss.item()
            
            val_predictions_all.append(predictions)
            val_targets_all.append(batch_y)
    
    # 计算平均损失
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    
    # 计算RMSE
    val_predictions_all = torch.cat(val_predictions_all, dim=0)
    val_targets_all = torch.cat(val_targets_all, dim=0)
    val_rmse = torch.sqrt(torch.mean((val_predictions_all - val_targets_all) ** 2)).item()
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    # 早停机制（基于RMSE）
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_val_loss = val_loss
        patience_counter = 0
        # 保存最佳模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_rmse': best_val_rmse,
            'val_loss': best_val_loss,
        }, 'best_model_enhanced.pth')
        print(f'[Best] Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, Val RMSE = {val_rmse:.6f}')
    else:
        patience_counter += 1
    
    if epoch % 20 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, Val RMSE = {val_rmse:.6f}, LR = {current_lr:.6f}')
    
    if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch}')
        break

# 加载最佳模型
checkpoint = torch.load('best_model_enhanced.pth')
model.load_state_dict(checkpoint['model_state_dict'])
best_val_rmse = checkpoint['val_rmse']

# 11. 在验证集上最终评估
model.eval()
with torch.no_grad():
    val_predictions = model(X_val_tensor)
    val_mse = torch.mean((val_predictions - y_val_tensor) ** 2).item()
    val_rmse = np.sqrt(val_mse)

# 计算分数
score = 1.0 / (1.0 + val_rmse)
print(f'\n{"="*50}')
print(f'Final Validation Results:')
print(f'Best Validation RMSE: {best_val_rmse:.6f}')
print(f'Final Validation RMSE: {val_rmse:.6f}')
print(f'Score = 1.0 / (1.0 + {val_rmse:.6f}) = {score:.6f}')
print(f'{"="*50}')

# 12. 对测试集进行预测
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor).numpy().flatten()

# 确保预测值非负（使用ReLU函数确保）
test_predictions = np.maximum(test_predictions, 0)

# 13. 生成提交文件
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'tested_positive': test_predictions
})

submission_df.to_csv('submission_enhanced.csv', index=False)
print(f"\nEnhanced submission file saved.")
print(f"Predictions range: [{test_predictions.min():.2f}, {test_predictions.max():.2f}]")
print(f"Mean prediction: {test_predictions.mean():.2f}")
print(f"Std prediction: {test_predictions.std():.2f}")

# 14. 额外：创建多个模型的集成（可选，进一步提高分数）
print("\nCreating model ensemble for better performance...")

# 训练多个不同架构的模型（简化示例）
def create_different_model(input_dim, model_type='default'):
    if model_type == 'wide':
        return nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    elif model_type == 'deep':
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
    else:  # simple
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1)
        )

# 训练一个额外的简单模型作为对比
simple_model = create_different_model(input_dim, 'simple')
simple_optimizer = optim.Adam(simple_model.parameters(), lr=1e-3)
simple_criterion = nn.MSELoss()

print("Training additional simple model for ensemble...")
for epoch in range(100):
    simple_model.train()
    for batch_x, batch_y in train_loader:
        simple_optimizer.zero_grad()
        predictions = simple_model(batch_x)
        loss = simple_criterion(predictions, batch_y)
        loss.backward()
        simple_optimizer.step()

# 使用两个模型的预测进行集成
model.eval()
simple_model.eval()

with torch.no_grad():
    # 主模型预测
    test_pred_main = model(X_test_tensor).numpy().flatten()
    # 简单模型预测
    test_pred_simple = simple_model(X_test_tensor).numpy().flatten()
    
    # 加权集成（给主模型更高权重）
    ensemble_weight = 0.7
    test_predictions_ensemble = (ensemble_weight * test_pred_main + 
                                 (1 - ensemble_weight) * test_pred_simple)
    
    # 确保非负
    test_predictions_ensemble = np.maximum(test_predictions_ensemble, 0)

# 生成集成预测的提交文件
ensemble_submission_df = pd.DataFrame({
    'id': test_df['id'],
    'tested_positive': test_predictions_ensemble
})

ensemble_submission_df.to_csv('submission_ensemble.csv', index=False)
print(f"Ensemble submission file saved.")
print(f"Ensemble predictions range: [{test_predictions_ensemble.min():.2f}, {test_predictions_ensemble.max():.2f}]")

print("\nOptimization complete! Try submitting both submission_enhanced.csv and submission_ensemble.csv.")
print("The ensemble submission may provide better performance.")