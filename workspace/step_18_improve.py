import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 1. 加载数据
train_df = pd.read_csv('/home/huazhao/DL-HW/ML-2025/mini-aide/input/train.csv')
test_df = pd.read_csv('/home/huazhao/DL-HW/ML-2025/mini-aide/input/test.csv')

print(f"训练集形状: {train_df.shape}")
print(f"测试集形状: {test_df.shape}")

# 2. 系统化的特征工程
def create_advanced_features(df, is_train=True):
    """创建高级时间序列特征"""
    df = df.copy()
    
    # 保存id和目标列
    if is_train:
        target_col = 'tested_positive_day3'
        target = df[target_col].copy()
        df = df.drop([target_col], axis=1)
    
    id_col = df['id'] if 'id' in df.columns else None
    if 'id' in df.columns:
        df = df.drop(['id'], axis=1)
    
    # 1. 基础滞后特征（最重要的预测因子）
    if 'tested_positive_day2' in df.columns:
        # 最近的变化趋势
        df['positivity_change_1d'] = df['tested_positive_day2'] - df['tested_positive_day1']
        df['positivity_change_2d'] = df['tested_positive_day2'] - df.get('tested_positive_day0', df['tested_positive_day1'])
        df['positivity_momentum'] = df['positivity_change_1d'] / (df['tested_positive_day1'] + 1e-5)
        
        # 加权平均（近期的权重更高）
        df['positivity_weighted_avg'] = (
            0.2 * df['tested_positive_day1'] + 
            0.5 * df['tested_positive_day2'] + 
            0.3 * df.get('tested_positive_day3', df['tested_positive_day2'])
        )
    
    # 2. 症状相关特征
    symptom_cols = ['cli', 'ili', 'wnohh_cmnty_cli', 'hh_cmnty_cli', 'nohh_cmnty_cli']
    for col in symptom_cols:
        for day in [1, 2, 3]:
            day_col = f'{col}_day{day}'
            if day_col in df.columns:
                # 症状变化率
                if day > 1:
                    prev_col = f'{col}_day{day-1}'
                    if prev_col in df.columns:
                        df[f'{col}_change_day{day}'] = df[day_col] - df[prev_col]
                        df[f'{col}_change_rate_day{day}'] = df[f'{col}_change_day{day}'] / (df[prev_col] + 1e-5)
                
                # 症状相对水平（标准化）
                if day == 2:  # 使用day2作为基准
                    df[f'{col}_relative_day{day}'] = (df[day_col] - df[day_col].mean()) / (df[day_col].std() + 1e-5)
    
    # 症状聚合特征
    for day in [1, 2, 3]:
        day_symptom_cols = [f'{col}_day{day}' for col in symptom_cols if f'{col}_day{day}' in df.columns]
        if day_symptom_cols:
            df[f'symptom_score_day{day}'] = df[day_symptom_cols].mean(axis=1)
            df[f'symptom_variation_day{day}'] = df[day_symptom_cols].std(axis=1)
    
    # 3. 行为与信念特征
    # 室内活动风险评分
    indoor_cols = ['wrestaurant_indoors', 'wshop_indoors', 'wlarge_event_indoors']
    for day in [1, 2, 3]:
        day_indoor_cols = [f'{col}_day{day}' for col in indoor_cols if f'{col}_day{day}' in df.columns]
        if day_indoor_cols:
            df[f'indoor_risk_day{day}'] = df[day_indoor_cols].mean(axis=1)
            df[f'indoor_risk_trend_day{day}'] = df[f'indoor_risk_day{day}'] - df.get(f'indoor_risk_day{day-1}', df[f'indoor_risk_day{day}'])
    
    # 防护措施评分
    protection_cols = ['wearing_mask_7d', 'wbelief_masking_effective', 'wothers_masked_public']
    for day in [1, 2, 3]:
        day_protection_cols = [f'{col}_day{day}' for col in protection_cols if f'{col}_day{day}' in df.columns]
        if day_protection_cols:
            df[f'protection_score_day{day}'] = df[day_protection_cols].mean(axis=1)
    
    # 社交距离评分
    distancing_cols = ['wbelief_distancing_effective', 'wothers_distanced_public', 'public_transit']
    for day in [1, 2, 3]:
        day_distancing_cols = [f'{col}_day{day}' for col in distancing_cols if f'{col}_day{day}' in df.columns]
        if day_distancing_cols:
            df[f'distancing_score_day{day}'] = df[day_distancing_cols].mean(axis=1)
    
    # 4. 心理与社会因素
    worry_cols = ['wworried_catch_covid', 'wworried_finance']
    for day in [1, 2, 3]:
        day_worry_cols = [f'{col}_day{day}' for col in worry_cols if f'{col}_day{day}' in df.columns]
        if day_worry_cols:
            df[f'worry_score_day{day}'] = df[day_worry_cols].mean(axis=1)
    
    # 5. 关键交互特征（基于领域知识）
    if 'tested_positive_day2' in df.columns:
        # 阳性率与防护措施的交互
        df['positivity_protection_interaction'] = df['tested_positive_day2'] * (100 - df.get('protection_score_day2', 50)) / 100
        
        # 阳性率与室内活动的交互
        df['positivity_indoor_interaction'] = df['tested_positive_day2'] * df.get('indoor_risk_day2', 25) / 100
        
        # 症状与防护的交互
        df['symptom_protection_interaction'] = df.get('symptom_score_day2', 10) * (100 - df.get('protection_score_day2', 50)) / 1000
        
        # 疫苗朋友比例与阳性率的交互
        df['vaccine_positivity_interaction'] = df['wcovid_vaccinated_friends_day2'] * df['tested_positive_day2'] / 100
    
    # 6. 状态特征
    if 'tested_positive_day2' in df.columns and 'tested_positive_day1' in df.columns:
        df['is_increasing'] = (df['tested_positive_day2'] > df['tested_positive_day1']).astype(float)
        df['increase_magnitude'] = (df['tested_positive_day2'] - df['tested_positive_day1']).abs()
        
        # 变化加速
        if 'tested_positive_day0' in df.columns:
            df['acceleration'] = (df['tested_positive_day2'] - 2*df['tested_positive_day1'] + df['tested_positive_day0'])
        else:
            df['acceleration'] = (df['tested_positive_day2'] - df['tested_positive_day1'])
    
    # 7. 聚合统计特征
    # 3天平均值和标准差
    for metric in ['cli', 'ili', 'wnohh_cmnty_cli', 'tested_positive']:
        day_cols = [f'{metric}_day{i}' for i in [1, 2, 3] if f'{metric}_day{i}' in df.columns]
        if len(day_cols) >= 2:
            df[f'{metric}_3day_mean'] = df[day_cols].mean(axis=1)
            df[f'{metric}_3day_std'] = df[day_cols].std(axis=1)
            df[f'{metric}_3day_range'] = df[day_cols].max(axis=1) - df[day_cols].min(axis=1)
    
    # 8. 相对变化特征
    for day in [2, 3]:
        if f'tested_positive_day{day}' in df.columns and f'tested_positive_day{day-1}' in df.columns:
            df[f'positivity_ratio_day{day}'] = df[f'tested_positive_day{day}'] / (df[f'tested_positive_day{day-1}'] + 1e-5)
    
    # 9. 添加id列回数据
    if id_col is not None:
        df['id'] = id_col
    
    # 如果是训练集，添加目标列
    if is_train:
        df['tested_positive_day3'] = target
    
    return df

# 3. 应用特征工程
print("开始高级特征工程...")
train_df_enhanced = create_advanced_features(train_df, is_train=True)
test_df_enhanced = create_advanced_features(test_df, is_train=False)

print(f"原始特征数: {train_df.shape[1] - 2}")
print(f"增强后特征数: {train_df_enhanced.shape[1] - 2}")
print(f"训练集形状: {train_df_enhanced.shape}")

# 4. 分离特征和目标
target_col = 'tested_positive_day3'
X = train_df_enhanced.drop(['id', target_col], axis=1).fillna(0)
y = train_df_enhanced[target_col].values
X_test = test_df_enhanced.drop(['id'], axis=1).fillna(0)

print(f"特征维度: {X.shape}")
print(f"测试集特征维度: {X_test.shape}")

# 5. 使用固定验证集（最后20%）
train_size = int(len(X) * 0.8)
X_train, X_val = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

print(f"\n数据划分:")
print(f"  训练集: {len(X_train)} 样本")
print(f"  验证集: {len(X_val)} 样本")
print(f"  测试集: {len(X_test)} 样本")

# 6. 特征选择（选择最重要的30个特征）
print("\n进行特征选择...")
selector = SelectKBest(score_func=f_regression, k=min(30, X_train.shape[1]))
X_train_selected = selector.fit_transform(X_train, y_train)
X_val_selected = selector.transform(X_val)
X_test_selected = selector.transform(X_test)

selected_features = X_train.columns[selector.get_support()]
print(f"  原始特征数: {X_train.shape[1]}")
print(f"  选择后特征数: {X_train_selected.shape[1]}")
print(f"  选择特征: {list(selected_features[:10])}...")

# 7. 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_val_scaled = scaler.transform(X_val_selected)
X_test_scaled = scaler.transform(X_test_selected)

# 8. 定义优化的DNN模型
class OptimizedDNN(nn.Module):
    """优化的深度神经网络"""
    def __init__(self, input_dim):
        super(OptimizedDNN, self).__init__()
        
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
            
            nn.Linear(16, 8),
            nn.ReLU(),
            
            nn.Linear(8, 1)
        )
        
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
        return self.network(x)

# 9. 训练设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n使用设备: {device}")

# 转换为张量
X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

# 初始化模型
input_dim = X_train_scaled.shape[1]
model = OptimizedDNN(input_dim).to(device)

# 损失函数和优化器
criterion = nn.HuberLoss(delta=1.0)  # Huber损失对异常值更鲁棒
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, verbose=True
)

# 10. 训练循环
print("\n开始训练...")
epochs = 300
best_val_rmse = float('inf')
best_model_state = None
patience = 25
patience_counter = 0

train_losses = []
val_rmses = []

for epoch in range(epochs):
    # 训练阶段
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # 验证阶段
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_tensor).cpu().numpy().flatten()
        val_rmse = np.sqrt(np.mean((val_preds - y_val) ** 2))
        val_rmses.append(val_rmse)
    
    # 学习率调度
    scheduler.step(val_rmse)
    
    # 早停和模型保存
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_model_state = model.state_dict().copy()
        patience_counter = 0
        
        # 保存最佳模型
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
    
    # 打印进度
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val RMSE: {val_rmse:.4f} | "
              f"Best RMSE: {best_val_rmse:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # 早停检查
    if patience_counter >= patience:
        print(f"\n早停在 epoch {epoch+1}")
        break

print(f"\n最佳验证集RMSE: {best_val_rmse:.4f}")

# 11. 加载最佳模型进行预测
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# 验证集预测
with torch.no_grad():
    val_preds = model(X_val_tensor).cpu().numpy().flatten()

# 计算最终验证分数
val_rmse_final = np.sqrt(np.mean((val_preds - y_val) ** 2))
score = 1.0 / (1.0 + val_rmse_final)

print(f"\n最终验证结果:")
print(f"  Val RMSE: {val_rmse_final:.4f}")
print(f"  Score: {score:.4f}")

# 12. 测试集预测
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
with torch.no_grad():
    test_preds = model(X_test_tensor).cpu().numpy().flatten()

# 13. 智能后处理
print("\n进行智能后处理...")

# 分析验证集误差分布
val_errors = val_preds - y_val
error_mean = np.mean(val_errors)
error_std = np.std(val_errors)

print(f"  验证集误差统计:")
print(f"    均值: {error_mean:.4f}")
print(f"    标准差: {error_std:.4f}")
print(f"    中位数: {np.median(val_errors):.4f}")

# 基于误差分布调整测试集预测
# 使用贝叶斯调整：考虑验证集的系统偏差
test_preds_adjusted = test_preds - error_mean

# 进一步校准：考虑预测值本身的分布
# 计算验证集预测值的分位数映射
val_pred_quantiles = np.percentile(val_preds, [10, 25, 50, 75, 90])
val_target_quantiles = np.percentile(y_val, [10, 25, 50, 75, 90])

# 简单的分位数校准
for i in range(len(val_pred_quantiles)-1):
    mask = (test_preds_adjusted >= val_pred_quantiles[i]) & (test_preds_adjusted < val_pred_quantiles[i+1])
    if np.sum(mask) > 0:
        # 调整到对应的目标分位数
        adjustment_factor = val_target_quantiles[i] / (val_pred_quantiles[i] + 1e-5)
        test_preds_adjusted[mask] = test_preds_adjusted[mask] * adjustment_factor

# 确保预测在合理范围内
# 基于训练集目标分布
train_target_mean = np.mean(y_train)
train_target_std = np.std(y_train)

# 温和的缩尾处理（保留98%的数据范围）
lower_bound = np.percentile(y_train, 1)
upper_bound = np.percentile(y_train, 99)
test_preds_final = np.clip(test_preds_adjusted, lower_bound * 0.8, upper_bound * 1.2)

# 确保非负
test_preds_final = np.maximum(test_preds_final, 0)

print(f"\n后处理结果:")
print(f"  原始测试集预测范围: [{np.min(test_preds):.2f}, {np.max(test_preds):.2f}]")
print(f"  调整后预测范围: [{np.min(test_preds_final):.2f}, {np.max(test_preds_final):.2f}]")
print(f"  调整后预测均值: {np.mean(test_preds_final):.2f}")
print(f"  训练集目标均值: {train_target_mean:.2f}")
print(f"  验证集目标均值: {np.mean(y_val):.2f}")

# 14. 生成提交文件
submission = pd.DataFrame({
    'id': test_df['id'],
    'tested_positive': test_preds_final
})

submission.to_csv('submission.csv', index=False)
print(f"\n提交文件已保存: submission.csv")
print(f"包含 {len(submission)} 条预测结果")

# 15. 打印最终分数
print(f"\n{'='*60}")
print("最终结果:")
print(f"  Validation RMSE: {val_rmse_final:.4f}")
print(f"  Score = (1.0 / (1.0 + RMSE)) = {score:.4f}")
print(f"  预测值范围: [{np.min(test_preds_final):.2f}, {np.max(test_preds_final):.2f}]")
print(f"  预测值均值: {np.mean(test_preds_final):.2f} ± {np.std(test_preds_final):.2f}")
print(f"{'='*60}")

print(f"\nScore= (1.0 / (1.0 + RMSE)) = {score:.4f}")