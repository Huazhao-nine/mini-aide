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

# 2. 重新设计特征工程 - 专注于时间序列模式和关键特征
def create_features(df, is_train=True):
    """创建优化的时间序列特征"""
    df = df.copy()
    
    # 保存id和目标列
    if is_train:
        target = df['tested_positive_day3'].copy()
        df = df.drop(['tested_positive_day3'], axis=1)
    
    # 移除id列
    ids = df['id'].copy() if 'id' in df.columns else None
    if 'id' in df.columns:
        df = df.drop(['id'], axis=1)
    
    # 核心时间序列特征
    # 1. 阳性率趋势特征（最重要的预测因子）
    df['positivity_trend_1_2'] = df['tested_positive_day2'] - df['tested_positive_day1']
    df['positivity_ratio_2_1'] = df['tested_positive_day2'] / (df['tested_positive_day1'] + 1e-5)
    df['positivity_momentum'] = df['positivity_trend_1_2'] / (df['tested_positive_day1'] + 1e-5)
    df['positivity_avg_d1_d2'] = (df['tested_positive_day1'] + df['tested_positive_day2']) / 2
    
    # 2. 症状指标的趋势和聚合
    symptom_metrics = ['cli', 'ili', 'hh_cmnty_cli', 'nohh_cmnty_cli']
    for metric in symptom_metrics:
        # 计算变化率
        df[f'{metric}_change_1_2'] = df[f'{metric}_day2'] - df[f'{metric}_day1']
        df[f'{metric}_ratio_2_1'] = df[f'{metric}_day2'] / (df[f'{metric}_day1'] + 1e-5)
        
        # 计算第1、2天的平均值
        df[f'{metric}_avg_d1_d2'] = (df[f'{metric}_day1'] + df[f'{metric}_day2']) / 2
    
    # 3. 行为指标聚合
    # 室内活动风险评分
    indoor_cols = ['wrestaurant_indoors', 'wshop_indoors', 'wlarge_event_indoors']
    for day in [1, 2, 3]:
        day_cols = [f'{col}_day{day}' for col in indoor_cols]
        df[f'indoor_risk_day{day}'] = df[day_cols].mean(axis=1)
    
    # 防护措施评分
    mask_cols = ['wearing_mask_7d', 'wbelief_masking_effective', 'wothers_masked_public']
    for day in [1, 2, 3]:
        day_cols = [f'{col}_day{day}' for col in mask_cols]
        df[f'protection_day{day}'] = df[day_cols].mean(axis=1)
    
    # 社交距离评分
    dist_cols = ['wbelief_distancing_effective', 'wothers_distanced_public']
    for day in [1, 2, 3]:
        day_cols = [f'{col}_day{day}' for col in dist_cols]
        df[f'distancing_day{day}'] = df[day_cols].mean(axis=1)
    
    # 4. 关键交互特征（基于领域知识）
    df['symptom_exposure_interaction'] = df['cli_day2'] * df['indoor_risk_day2'] / 1000
    df['positivity_protection_interaction'] = df['tested_positive_day2'] * (100 - df['protection_day2']) / 100
    df['vaccine_symptom_interaction'] = df['wcovid_vaccinated_friends_day2'] * df['cli_day2'] / 100
    
    # 5. 趋势一致性特征
    df['symptom_positivity_alignment'] = (
        (df['cli_change_1_2'] > 0) & (df['positivity_trend_1_2'] > 0)
    ).astype(float)
    
    # 6. 波动性特征
    df['positivity_volatility'] = abs(df['positivity_trend_1_2']) / (df['positivity_avg_d1_d2'] + 1e-5)
    
    # 7. 滞后特征（如果可用）
    if 'tested_positive_day1' in df.columns and 'tested_positive_day2' in df.columns:
        # 计算二阶差分（加速度）
        if 'tested_positive_day0' not in df.columns:  # 假设没有day0
            # 使用一阶差分的变化率作为近似加速度
            df['positivity_acceleration'] = df['positivity_trend_1_2'] - df.get(
                'positivity_trend_0_1', df['positivity_trend_1_2']
            )
    
    # 8. 状态特征
    df['is_increasing'] = (df['tested_positive_day2'] > df['tested_positive_day1']).astype(float)
    df['is_high_symptom'] = (df['cli_day2'] > df['cli_day2'].median()).astype(float)
    
    # 9. 基于州的特征（如果可用）
    state_cols = [col for col in df.columns if len(col) == 2 and col.isupper()]
    if state_cols:
        # 为每个州计算特定特征（这里简化处理）
        df['state_specific_risk'] = 0
        for state in state_cols:
            state_mask = df[state] == 1
            if state_mask.any():
                # 可以在这里添加州特定的计算，但注意避免数据泄露
                pass
    
    # 10. 关键指标的加权组合
    df['composite_risk_score'] = (
        0.4 * df['tested_positive_day2'] +
        0.3 * df['cli_day2'] +
        0.2 * df['indoor_risk_day2'] +
        0.1 * (100 - df['protection_day2'])
    ) / 100
    
    # 添加id列回数据
    if ids is not None:
        df['id'] = ids
    
    # 如果是训练集，添加目标列
    if is_train:
        df['tested_positive_day3'] = target
    
    return df

# 3. 应用特征工程
print("应用特征工程...")
train_df_enhanced = create_features(train_df, is_train=True)
test_df_enhanced = create_features(test_df, is_train=False)

print(f"原始训练集特征数: {train_df.shape[1] - 2}")
print(f"增强后训练集特征数: {train_df_enhanced.shape[1] - 2}")

# 4. 分离特征和目标
target_col = 'tested_positive_day3'
X = train_df_enhanced.drop(['id', target_col], axis=1)
y = train_df_enhanced[target_col].values
X_test = test_df_enhanced.drop(['id'], axis=1)

print(f"特征维度: {X.shape}")

# 5. 按时间顺序划分验证集（最后20%）
train_size = int(len(X) * 0.8)
X_train, X_val = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

print(f"\n数据划分:")
print(f"  训练集: {len(X_train)} 样本")
print(f"  验证集: {len(X_val)} 样本")
print(f"  测试集: {len(X_test)} 样本")

# 6. 特征选择（选择最重要的特征）
print("\n应用特征选择...")
k_best = 30  # 选择30个最重要的特征
selector = SelectKBest(score_func=f_regression, k=min(k_best, X_train.shape[1]))
X_train_selected = selector.fit_transform(X_train, y_train)
X_val_selected = selector.transform(X_val)
X_test_selected = selector.transform(X_test)

selected_features = X.columns[selector.get_support()]
print(f"  选择 {len(selected_features)} 个最重要特征")
print(f"  前10个重要特征: {list(selected_features[:10])}")

# 7. 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_val_scaled = scaler.transform(X_val_selected)
X_test_scaled = scaler.transform(X_test_selected)

print(f"\n标准化后特征范围:")
print(f"  训练集: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")
print(f"  验证集: [{X_val_scaled.min():.2f}, {X_val_scaled.max():.2f}]")

# 8. 定义优化的PyTorch模型
class OptimizedDNN(nn.Module):
    """优化的深度神经网络，专为时间序列回归设计"""
    def __init__(self, input_dim):
        super(OptimizedDNN, self).__init__()
        
        self.network = nn.Sequential(
            # 第一层
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # 第二层
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # 第三层
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # 输出层
            nn.Linear(32, 1)
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
        return self.network(x)

# 9. 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n使用设备: {device}")

# 转换数据为张量
X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(device)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)  # 重要：不shuffle

# 初始化模型
input_dim = X_train_scaled.shape[1]
model = OptimizedDNN(input_dim).to(device)

# 损失函数和优化器
criterion = nn.L1Loss()  # MAE损失，对异常值更鲁棒
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, verbose=True
)

# 训练参数
epochs = 300
best_val_rmse = float('inf')
best_model_state = None
patience = 30
patience_counter = 0

print("\n开始训练模型...")
for epoch in range(epochs):
    # 训练阶段
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        train_loss += loss.item()
    
    # 验证阶段
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_tensor).cpu().numpy().flatten()
        val_rmse = np.sqrt(np.mean((val_preds - y_val) ** 2))
    
    # 更新学习率
    scheduler.step(val_rmse)
    
    # 早停检查
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_model_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
    
    if (epoch + 1) % 50 == 0:
        print(f"  Epoch {epoch+1}/{epochs}: Train Loss={train_loss/len(train_loader):.4f}, Val RMSE={val_rmse:.4f}")
    
    if patience_counter >= patience:
        print(f"  早停在 epoch {epoch+1}")
        break

# 加载最佳模型
model.load_state_dict(best_model_state)

# 10. 评估模型
model.eval()
with torch.no_grad():
    val_preds = model(X_val_tensor).cpu().numpy().flatten()

val_rmse = np.sqrt(np.mean((val_preds - y_val) ** 2))
score = 1.0 / (1.0 + val_rmse)

print(f"\n模型评估:")
print(f"  最佳验证集 RMSE: {best_val_rmse:.4f}")
print(f"  最终验证集 RMSE: {val_rmse:.4f}")
print(f"  分数: {score:.4f}")

# 11. 测试集预测
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
model.eval()
with torch.no_grad():
    test_preds = model(X_test_tensor).cpu().numpy().flatten()

# 12. 智能后处理
print("\n智能后处理...")

# 基于验证集的误差分布进行调整
val_errors = val_preds - y_val
error_mean = np.mean(val_errors)
error_std = np.std(val_errors)

# 1. 偏差校正
test_preds_corrected = test_preds - error_mean

# 2. 确保预测在合理范围内（基于训练数据分布）
train_target_stats = {
    'min': np.min(y_train),
    'max': np.max(y_train),
    'mean': np.mean(y_train),
    'std': np.std(y_train),
    'q1': np.percentile(y_train, 25),
    'q3': np.percentile(y_train, 75)
}

# 温和的缩尾处理（保留极端值但避免不合理值）
lower_bound = max(0, train_target_stats['q1'] - 1.5 * (train_target_stats['q3'] - train_target_stats['q1']))
upper_bound = train_target_stats['q3'] + 1.5 * (train_target_stats['q3'] - train_target_stats['q1'])

test_preds_clipped = np.clip(test_preds_corrected, lower_bound, upper_bound)

# 3. 应用平滑（基于时间序列特性）
# 对测试集预测应用轻度指数平滑
alpha = 0.3  # 平滑参数
test_preds_smoothed = test_preds_clipped.copy()
for i in range(1, len(test_preds_smoothed)):
    test_preds_smoothed[i] = alpha * test_preds_clipped[i] + (1 - alpha) * test_preds_smoothed[i-1]

# 4. 确保非负
test_preds_final = np.maximum(test_preds_smoothed, 0)

print(f"  验证集误差统计 - 均值: {error_mean:.4f}, 标准差: {error_std:.4f}")
print(f"  训练集目标统计 - 最小值: {train_target_stats['min']:.2f}, 最大值: {train_target_stats['max']:.2f}")
print(f"  调整后预测范围: [{np.min(test_preds_final):.2f}, {np.max(test_preds_final):.2f}]")
print(f"  调整后预测均值: {np.mean(test_preds_final):.2f}")

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
print(f"  验证集 RMSE: {val_rmse:.4f}")
print(f"  分数 = 1.0 / (1.0 + RMSE) = {score:.4f}")
print(f"  预测值范围: [{np.min(test_preds_final):.2f}, {np.max(test_preds_final):.2f}]")
print(f"  预测值统计 - 均值: {np.mean(test_preds_final):.2f}, 标准差: {np.std(test_preds_final):.2f}")
print(f"  验证集目标统计 - 均值: {np.mean(y_val):.2f}, 标准差: {np.std(y_val):.2f}")

# 计算额外的评估指标
mae = np.mean(np.abs(val_preds - y_val))
mape = np.mean(np.abs((val_preds - y_val) / (y_val + 1e-5))) * 100
print(f"  验证集 MAE: {mae:.4f}")
print(f"  验证集 MAPE: {mape:.2f}%")

print(f"\nScore= (1.0 / (1.0 + RMSE)) = {score:.4f}")