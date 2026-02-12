import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
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

# 2. 智能特征工程 - 聚焦核心指标
def create_intelligent_features(df, is_train=True):
    """创建智能特征，基于领域知识"""
    df = df.copy()
    
    # 保存id和目标列
    if is_train:
        target_col = 'tested_positive_day3'
        target = df[target_col].copy()
        df = df.drop([target_col], axis=1)
    
    id_col = df['id'] if 'id' in df.columns else None
    df = df.drop(['id'], axis=1)
    
    # 1. 核心症状特征聚合
    for day in [1, 2, 3]:
        # 症状复合指标
        if all([f'cli_day{day}' in df.columns, f'ili_day{day}' in df.columns,
                f'hh_cmnty_cli_day{day}' in df.columns, f'nohh_cmnty_cli_day{day}' in df.columns]):
            df[f'symptom_severity_day{day}'] = (
                df[f'cli_day{day}'] * 0.4 + 
                df[f'ili_day{day}'] * 0.3 + 
                df[f'hh_cmnty_cli_day{day}'] * 0.2 + 
                df[f'nohh_cmnty_cli_day{day}'] * 0.1
            )
    
    # 2. 行为风险指数 (室内活动 + 公共交通)
    for day in [1, 2, 3]:
        risk_cols = []
        weights = []
        
        if f'wrestaurant_indoors_day{day}' in df.columns:
            risk_cols.append(f'wrestaurant_indoors_day{day}')
            weights.append(0.4)
        if f'wshop_indoors_day{day}' in df.columns:
            risk_cols.append(f'wshop_indoors_day{day}')
            weights.append(0.3)
        if f'wlarge_event_indoors_day{day}' in df.columns:
            risk_cols.append(f'wlarge_event_indoors_day{day}')
            weights.append(0.2)
        if f'public_transit_day{day}' in df.columns:
            risk_cols.append(f'public_transit_day{day}')
            weights.append(0.1)
        
        if risk_cols:
            # 归一化权重
            weights = np.array(weights) / np.sum(weights)
            df[f'behavior_risk_day{day}'] = sum(df[col] * w for col, w in zip(risk_cols, weights))
    
    # 3. 防护指数 (口罩 + 社交距离)
    for day in [1, 2, 3]:
        protection_cols = []
        
        if f'wearing_mask_7d_day{day}' in df.columns:
            protection_cols.append(f'wearing_mask_7d_day{day}')
        if f'wbelief_masking_effective_day{day}' in df.columns:
            protection_cols.append(f'wbelief_masking_effective_day{day}')
        if f'wothers_masked_public_day{day}' in df.columns:
            protection_cols.append(f'wothers_masked_public_day{day}')
        if f'wothers_distanced_public_day{day}' in df.columns:
            protection_cols.append(f'wothers_distanced_public_day{day}')
        
        if protection_cols:
            df[f'protection_index_day{day}'] = df[protection_cols].mean(axis=1)
    
    # 4. 时间序列特征 - 趋势和加速度
    for base_feat in ['cli', 'ili', 'tested_positive', 'wearing_mask_7d', 'wworried_catch_covid']:
        day_cols = [f'{base_feat}_day1', f'{base_feat}_day2', f'{base_feat}_day3']
        existing_cols = [col for col in day_cols if col in df.columns]
        
        if len(existing_cols) == 3:
            # 斜率
            df[f'{base_feat}_slope'] = df[existing_cols[2]] - df[existing_cols[0]]
            
            # 曲率 (二阶差分)
            df[f'{base_feat}_curvature'] = (
                df[existing_cols[2]] - 2*df[existing_cols[1]] + df[existing_cols[0]]
            )
            
            # 近期变化
            df[f'{base_feat}_recent_change'] = df[existing_cols[2]] - df[existing_cols[1]]
    
    # 5. 状态特征组合
    # 高危状态：症状严重 + 高风险行为 + 低防护
    for day in [1, 2, 3]:
        if (f'symptom_severity_day{day}' in df.columns and 
            f'behavior_risk_day{day}' in df.columns and
            f'protection_index_day{day}' in df.columns):
            
            # 归一化处理
            symptom_norm = (df[f'symptom_severity_day{day}'] - df[f'symptom_severity_day{day}'].mean()) / df[f'symptom_severity_day{day}'].std()
            behavior_norm = (df[f'behavior_risk_day{day}'] - df[f'behavior_risk_day{day}'].mean()) / df[f'behavior_risk_day{day}'].std()
            protection_norm = (df[f'protection_index_day{day}'] - df[f'protection_index_day{day}'].mean()) / df[f'protection_index_day{day}'].std()
            
            df[f'high_risk_state_day{day}'] = symptom_norm + behavior_norm - protection_norm
    
    # 6. 疫苗接种相关的特征
    for day in [1, 2, 3]:
        if f'wcovid_vaccinated_friends_day{day}' in df.columns:
            # 疫苗接种与担忧程度的关系
            if f'wworried_catch_covid_day{day}' in df.columns:
                df[f'vax_worry_ratio_day{day}'] = df[f'wcovid_vaccinated_friends_day{day}'] / (df[f'wworried_catch_covid_day{day}'] + 1)
            
            # 疫苗接种与行为的矛盾指数
            if f'behavior_risk_day{day}' in df.columns:
                df[f'vax_behavior_contradiction_day{day}'] = df[f'wcovid_vaccinated_friends_day{day}'] * df[f'behavior_risk_day{day}'] / 100
    
    # 7. 添加地理区域特征（简单编码州名）
    state_columns = ['AL', 'AZ', 'CA', 'CO', 'CT', 'FL', 'GA', 'IL', 'IN', 'IA', 'KS', 
                    'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MO', 'NJ', 'NM', 'NY',
                    'NC', 'OH', 'OK', 'OR', 'PA', 'SC', 'TN', 'TX', 'VA', 'WA', 'WV', 'WI']
    
    # 计算每个样本属于哪个州
    df['state_id'] = -1
    for i, state in enumerate(state_columns):
        if state in df.columns:
            df.loc[df[state] == 1, 'state_id'] = i
    
    # 8. 添加滞后特征的扩展
    if 'tested_positive_day1' in df.columns and 'tested_positive_day2' in df.columns:
        # 两天的变化率
        df['positivity_growth_rate'] = (df['tested_positive_day2'] - df['tested_positive_day1']) / (df['tested_positive_day1'] + 0.01)
        
        # 动量指标
        df['positivity_momentum'] = df['tested_positive_day2'] + (df['tested_positive_day2'] - df['tested_positive_day1'])
    
    # 添加id列回数据
    if id_col is not None:
        df['id'] = id_col
    
    # 如果是训练集，添加目标列
    if is_train:
        df['tested_positive_day3'] = target
    
    return df

# 应用特征工程
print("开始智能特征工程...")
train_df_enhanced = create_intelligent_features(train_df, is_train=True)
test_df_enhanced = create_intelligent_features(test_df, is_train=False)

print(f"原始特征数: {train_df.shape[1] - 2}")
print(f"增强后特征数: {train_df_enhanced.shape[1] - 2}")

# 3. 分离特征和目标
target_col = 'tested_positive_day3'
X = train_df_enhanced.drop(['id', target_col], axis=1)
y = train_df_enhanced[target_col].values.reshape(-1, 1)
X_test = test_df_enhanced.drop(['id'], axis=1)

# 4. 对目标变量进行变换
# 检查y的分布
y_original = y.copy()
y_log = np.log1p(y)

print(f"目标变量统计:")
print(f"  原始 - 均值: {y_original.mean():.2f}, 标准差: {y_original.std():.2f}")
print(f"  对数变换后 - 均值: {y_log.mean():.2f}, 标准差: {y_log.std():.2f}")

# 5. 使用固定验证集（最后20%）
train_size = int(len(X) * 0.8)
X_train, X_val = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_val = y_log[:train_size], y_log[train_size:]

print(f"\n数据划分:")
print(f"  训练集: {len(X_train)} 样本")
print(f"  验证集: {len(X_val)} 样本")
print(f"  测试集: {len(X_test)} 样本")

# 6. 特征选择和标准化
print("\n特征处理...")

# 使用更稳健的标准化
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 使用随机森林进行特征重要性排序
rf = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
rf.fit(X_train_scaled, y_train.ravel())

# 选择重要性大于平均值的特征
importances = rf.feature_importances_
feature_names = X.columns
mean_importance = importances.mean()
selected_features_idx = importances > (mean_importance * 0.5)  # 放宽阈值

X_train_selected = X_train_scaled[:, selected_features_idx]
X_val_selected = X_val_scaled[:, selected_features_idx]
X_test_selected = X_test_scaled[:, selected_features_idx]

print(f"  原始特征数: {X_train_scaled.shape[1]}")
print(f"  选择后特征数: {X_train_selected.shape[1]}")
print(f"  特征选择阈值: {mean_importance * 0.5:.6f}")

# 7. 定义改进的神经网络模型
class EnhancedCOVIDPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64, 32], dropout_rates=[0.3, 0.25, 0.2, 0.15]):
        super(EnhancedCOVIDPredictor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层
        for i, (hidden_dim, dropout_rate) in enumerate(zip(hidden_dims, dropout_rates)):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# 8. 训练函数
def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, fold_idx=1):
    """训练模型并评估"""
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    
    # 创建DataLoader
    batch_size = 64
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedCOVIDPredictor(input_dim=X_train.shape[1]).to(device)
    
    # 损失函数和优化器
    criterion = nn.HuberLoss(delta=1.0)  # Huber损失对异常值更鲁棒
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # 训练循环
    epochs = 200
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
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_tensor.to(device)).cpu().numpy()
            val_loss = np.sqrt(np.mean((val_preds - y_val) ** 2))  # RMSE
            val_losses.append(val_loss)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'best_model_fold{fold_idx}.pth')
        else:
            patience_counter += 1
        
        if epoch % 25 == 0:
            print(f'  Epoch {epoch}: Train Loss: {train_loss:.4f}, Val RMSE: {val_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        if patience_counter >= patience:
            print(f'  Early stopping at epoch {epoch}')
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load(f'best_model_fold{fold_idx}.pth'))
    
    # 在验证集上做最终预测
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_tensor.to(device)).cpu().numpy().flatten()
    
    # 在测试集上预测
    with torch.no_grad():
        test_preds = model(X_test_tensor.to(device)).cpu().numpy().flatten()
    
    return val_preds, test_preds, best_val_loss

# 9. 模型集成策略
print("\n训练神经网络模型...")

# 训练主要模型
val_preds_log, test_preds_log, best_val_rmse = train_and_evaluate(
    X_train_selected, y_train.ravel(), 
    X_val_selected, y_val.ravel(),
    X_test_selected
)

# 将预测值转换回原始尺度
val_preds = np.expm1(val_preds_log)
val_targets = np.expm1(y_val.ravel())

# 计算单个模型分数
rmse = np.sqrt(np.mean((val_preds - val_targets) ** 2))
score = 1.0 / (1.0 + rmse)

print(f"\n神经网络模型结果:")
print(f"  Best Val RMSE (log scale): {best_val_rmse:.4f}")
print(f"  Final Val RMSE: {rmse:.4f}")
print(f"  Score: {score:.4f}")

# 10. 使用LightGBM作为第二个模型（模型融合）
print("\n训练LightGBM模型作为集成...")

# 准备数据
train_data = lgb.Dataset(X_train_scaled, label=y_train.ravel())
val_data = lgb.Dataset(X_val_scaled, label=y_val.ravel(), reference=train_data)

# LightGBM参数
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': SEED,
    'n_jobs': -1,
}

# 训练
gbm_model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)]
)

# 预测
gbm_val_preds_log = gbm_model.predict(X_val_scaled, num_iteration=gbm_model.best_iteration)
gbm_test_preds_log = gbm_model.predict(X_test_scaled, num_iteration=gbm_model.best_iteration)

# 转换回原始尺度
gbm_val_preds = np.expm1(gbm_val_preds_log)
gbm_test_preds = np.expm1(gbm_test_preds_log)

# 计算LightGBM分数
gbm_rmse = np.sqrt(np.mean((gbm_val_preds - val_targets) ** 2))
gbm_score = 1.0 / (1.0 + gbm_rmse)

print(f"\nLightGBM模型结果:")
print(f"  Val RMSE: {gbm_rmse:.4f}")
print(f"  Score: {gbm_score:.4f}")

# 11. 模型融合（加权平均）
print("\n模型融合...")

# 使用验证集性能计算权重
nn_weight = score / (score + gbm_score)
gbm_weight = gbm_score / (score + gbm_score)

print(f"  神经网络权重: {nn_weight:.3f}")
print(f"  LightGBM权重: {gbm_weight:.3f}")

# 融合验证集预测（用于评估）
ensemble_val_preds = nn_weight * val_preds + gbm_weight * gbm_val_preds
ensemble_rmse = np.sqrt(np.mean((ensemble_val_preds - val_targets) ** 2))
ensemble_score = 1.0 / (1.0 + ensemble_rmse)

print(f"\n融合模型结果:")
print(f"  融合Val RMSE: {ensemble_rmse:.4f}")
print(f"  融合Score: {ensemble_score:.4f}")

# 12. 融合测试集预测
ensemble_test_preds = nn_weight * np.expm1(test_preds_log) + gbm_weight * gbm_test_preds

# 13. 后处理优化
print("\n后处理优化...")

# 基于验证集分布进行校准
val_mean = val_targets.mean()
val_std = val_targets.std()
pred_mean = ensemble_test_preds.mean()

# 调整到验证集分布
ensemble_test_preds = ensemble_test_preds * (val_mean / pred_mean) * 0.8 + ensemble_test_preds * 0.2

# 确保非负
ensemble_test_preds = np.maximum(ensemble_test_preds, 0)

# 温和的缩尾处理（只处理极端异常值）
q01 = np.percentile(val_targets, 1)
q99 = np.percentile(val_targets, 99)

# 使用更温和的调整
ensemble_test_preds = np.clip(ensemble_test_preds, q01 * 0.5, q99 * 1.5)

print(f"  后处理后 - 均值: {ensemble_test_preds.mean():.2f}, 标准差: {ensemble_test_preds.std():.2f}")
print(f"  验证集 - 均值: {val_targets.mean():.2f}, 标准差: {val_targets.std():.2f}")

# 14. 生成提交文件
submission = pd.DataFrame({
    'id': test_df['id'],
    'tested_positive': ensemble_test_preds
})

submission.to_csv('submission.csv', index=False)
print(f"\n提交文件已保存，包含 {len(submission)} 条预测结果")

# 15. 打印最终分数
print(f"\n{'='*60}")
print("最终结果:")
print(f"  神经网络单独: Score = {score:.4f}")
print(f"  LightGBM单独: Score = {gbm_score:.4f}")
print(f"  模型融合后: Score = {ensemble_score:.4f}")
print(f"  最终预测均值: {ensemble_test_preds.mean():.2f}")

print(f"\nScore= (1.0 / (1.0 + RMSE)) = {ensemble_score:.4f}")