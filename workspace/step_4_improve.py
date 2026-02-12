import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
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

# 2. 更丰富的特征工程
def create_features(df, is_train=True):
    """创建丰富的特征"""
    df = df.copy()
    
    # 保存id和目标列
    if is_train:
        target_col = 'tested_positive_day3'
        target = df[target_col].copy()
        df = df.drop([target_col], axis=1)
    
    id_col = df['id'] if 'id' in df.columns else None
    df = df.drop(['id'], axis=1)
    
    # 基础特征列表（排除州名特征）
    base_features = [col for col in df.columns if not (col in ['AL', 'AZ', 'CA', 'CO', 'CT', 'FL', 'GA', 'IL', 'IN', 'IA', 'KS', 
                                                             'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MO', 'NJ', 'NM', 'NY',
                                                             'NC', 'OH', 'OK', 'OR', 'PA', 'SC', 'TN', 'TX', 'VA', 'WA', 'WV', 'WI'])]
    
    # 1. 时间动态特征（变化率）
    for day in [1, 2]:
        next_day = day + 1
        for feat in base_features:
            if f'_day{next_day}' in feat and f'_day{day}' in df.columns:
                base_feat_name = feat.replace(f'_day{next_day}', '')
                day1_col = f'{base_feat_name}_day{day}'
                day2_col = feat
                if day1_col in df.columns and day2_col in df.columns:
                    # 绝对变化
                    df[f'{base_feat_name}_change_{day}_to_{next_day}'] = df[day2_col] - df[day1_col]
                    # 相对变化率（避免除以0）
                    df[f'{base_feat_name}_pct_change_{day}_to_{next_day}'] = (df[day2_col] - df[day1_col]) / (df[day1_col] + 1e-8)
    
    # 2. 跨天的统计特征
    for base_feat in ['cli', 'ili', 'wnohh_cmnty_cli', 'wearing_mask_7d', 'tested_positive', 
                     'wworried_catch_covid', 'wcovid_vaccinated_friends']:
        day_cols = [f'{base_feat}_day1', f'{base_feat}_day2', f'{base_feat}_day3']
        existing_cols = [col for col in day_cols if col in df.columns]
        if len(existing_cols) >= 2:
            # 均值
            df[f'{base_feat}_mean'] = df[existing_cols].mean(axis=1)
            # 标准差
            df[f'{base_feat}_std'] = df[existing_cols].std(axis=1)
            # 变化趋势（线性回归斜率）
            def calc_slope(row, cols):
                if len(cols) < 2:
                    return 0
                x = np.arange(len(cols))
                y = row[cols].values
                return np.polyfit(x, y, 1)[0] if len(set(y)) > 1 else 0
            
            df[f'{base_feat}_trend'] = df.apply(lambda row: calc_slope(row, existing_cols), axis=1)
    
    # 3. 领域知识驱动的交互特征
    for day in [1, 2, 3]:
        # 风险暴露指标 = 症状 * 室内活动
        if f'cli_day{day}' in df.columns and f'wrestaurant_indoors_day{day}' in df.columns:
            df[f'risk_exposure_day{day}'] = df[f'cli_day{day}'] * df[f'wrestaurant_indoors_day{day}']
        
        # 防护效果 = 口罩佩戴率 * 相信口罩有效性
        if f'wearing_mask_7d_day{day}' in df.columns and f'wbelief_masking_effective_day{day}' in df.columns:
            df[f'protection_effect_day{day}'] = df[f'wearing_mask_7d_day{day}'] * df[f'wbelief_masking_effective_day{day}'] / 100
        
        # 社交距离复合指标
        if f'wbelief_distancing_effective_day{day}' in df.columns and f'wothers_distanced_public_day{day}' in df.columns:
            df[f'distancing_composite_day{day}'] = (df[f'wbelief_distancing_effective_day{day}'] + 
                                                    df[f'wothers_distanced_public_day{day}']) / 2
    
    # 4. 疫苗接种与行为的交互
    for day in [1, 2, 3]:
        if f'wcovid_vaccinated_friends_day{day}' in df.columns:
            # 疫苗接种与室内活动
            if f'wrestaurant_indoors_day{day}' in df.columns:
                df[f'vax_behavior_day{day}'] = df[f'wcovid_vaccinated_friends_day{day}'] * df[f'wrestaurant_indoors_day{day}']
            
            # 疫苗接种与担心程度
            if f'wworried_catch_covid_day{day}' in df.columns:
                df[f'vax_worry_day{day}'] = df[f'wcovid_vaccinated_friends_day{day}'] * df[f'wworried_catch_covid_day{day}']
    
    # 5. 症状传播复合指标
    for day in [1, 2, 3]:
        symptom_cols = []
        if f'cli_day{day}' in df.columns:
            symptom_cols.append(f'cli_day{day}')
        if f'ili_day{day}' in df.columns:
            symptom_cols.append(f'ili_day{day}')
        if f'hh_cmnty_cli_day{day}' in df.columns:
            symptom_cols.append(f'hh_cmnty_cli_day{day}')
        if f'nohh_cmnty_cli_day{day}' in df.columns:
            symptom_cols.append(f'nohh_cmnty_cli_day{day}')
        
        if symptom_cols:
            df[f'symptom_composite_day{day}'] = df[symptom_cols].mean(axis=1)
    
    # 6. 滞后特征（使用day1和day2预测day3）
    for feat in base_features:
        if '_day2' in feat and '_day1' in df.columns:
            base_name = feat.replace('_day2', '')
            day1_col = f'{base_name}_day1'
            day2_col = feat
            if day1_col in df.columns:
                # 二阶差分（加速度）
                if f'{base_name}_day3' in df.columns:
                    df[f'{base_name}_acceleration'] = df[f'{base_name}_day3'] - 2*df[day2_col] + df[day1_col]
    
    # 7. 添加id列回数据
    if id_col is not None:
        df['id'] = id_col
    
    # 8. 如果是训练集，添加目标列
    if is_train:
        df['tested_positive_day3'] = target
    
    return df

# 应用特征工程
print("开始特征工程...")
train_df_enhanced = create_features(train_df, is_train=True)
test_df_enhanced = create_features(test_df, is_train=False)

print(f"原始特征数: {train_df.shape[1] - 2}")
print(f"增强后特征数: {train_df_enhanced.shape[1] - 2}")

# 3. 分离特征和目标
target_col = 'tested_positive_day3'
X = train_df_enhanced.drop(['id', target_col], axis=1)
y = train_df_enhanced[target_col].values.reshape(-1, 1)
X_test = test_df_enhanced.drop(['id'], axis=1)

# 4. 对目标变量进行变换（使其更接近正态分布）
# 使用log1p变换，预测后再用expm1变换回来
y_log = np.log1p(y)

# 5. 时间序列交叉验证
tscv = TimeSeriesSplit(n_splits=5)
fold_scores = []
all_val_preds = []
all_val_targets = []

# 存储测试集预测用于集成
test_preds = []

# 6. 定义改进的神经网络模型
class COVIDPredictor(nn.Module):
    def __init__(self, input_dim):
        super(COVIDPredictor, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.25),
        )
        
        # 残差连接
        self.residual = nn.Linear(input_dim, 128) if input_dim != 128 else nn.Identity()
        
        self.block2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.15),
            
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        identity = self.residual(x)
        out = self.block1(x)
        # 残差连接
        out = out + identity if identity.shape == out.shape else out
        out = self.block2(out)
        return out

# 7. 训练函数
def train_model(X_train, y_train, X_val, y_val, fold_idx):
    """训练单个模型"""
    # 标准化
    scaler = RobustScaler()  # 对异常值更鲁棒
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 特征选择（使用互信息，对非线性关系更敏感）
    k = min(50, X_train_scaled.shape[1])
    selector = SelectKBest(score_func=mutual_info_regression, k=k)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train.ravel())
    X_val_selected = selector.transform(X_val_scaled)
    
    print(f"Fold {fold_idx}: 选择 {k} 个特征")
    
    # PCA降维提取主成分
    pca = PCA(n_components=0.95)  # 保留95%的方差
    X_train_pca = pca.fit_transform(X_train_selected)
    X_val_pca = pca.transform(X_val_selected)
    
    print(f"Fold {fold_idx}: PCA后维度 {X_train_pca.shape[1]}")
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train_pca)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val_pca)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # 创建DataLoader
    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # 时间序列不shuffle
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = COVIDPredictor(input_dim=X_train_pca.shape[1]).to(device)
    
    # 使用SmoothL1Loss (Huber Loss)，结合MAE和MSE优点
    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 学习率调度器：余弦退火
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # 训练循环
    epochs = 150
    best_val_loss = float('inf')
    patience = 25
    patience_counter = 0
    
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
            
            # 梯度裁剪防止爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
        
        train_loss /= len(train_loader.dataset)
        scheduler.step()
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_tensor.to(device)).cpu().numpy()
            val_loss = np.sqrt(np.mean((val_preds - y_val) ** 2))  # RMSE
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'best_model_fold{fold_idx}.pth')
        else:
            patience_counter += 1
        
        if epoch % 30 == 0:
            print(f'  Epoch {epoch}: Train Loss: {train_loss:.4f}, Val RMSE: {val_loss:.4f}')
        
        if patience_counter >= patience:
            if epoch > 50:  # 确保至少训练50个epoch
                print(f'  Early stopping at epoch {epoch}')
                break
    
    # 加载最佳模型
    model.load_state_dict(torch.load(f'best_model_fold{fold_idx}.pth'))
    
    # 在验证集上做最终预测
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_tensor.to(device)).cpu().numpy()
    
    return val_preds.flatten(), scaler, selector, pca, model

# 8. 执行交叉验证
print("\n开始交叉验证训练...")
fold_idx = 0
for train_idx, val_idx in tscv.split(X):
    fold_idx += 1
    print(f"\n训练 Fold {fold_idx}/{tscv.n_splits}")
    
    # 划分数据
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y_log[train_idx], y_log[val_idx]
    
    # 训练模型
    val_preds_log, scaler, selector, pca, model = train_model(
        X_train_fold, y_train_fold, X_val_fold, y_val_fold, fold_idx
    )
    
    # 将预测值转换回原始尺度
    val_preds = np.expm1(val_preds_log)
    val_targets = np.expm1(y_val_fold.flatten())
    
    # 计算分数
    rmse = np.sqrt(np.mean((val_preds - val_targets) ** 2))
    score = 1.0 / (1.0 + rmse)
    fold_scores.append(score)
    
    print(f"Fold {fold_idx} RMSE: {rmse:.4f}, Score: {score:.4f}")
    
    # 保存验证集预测
    all_val_preds.extend(val_preds)
    all_val_targets.extend(val_targets)
    
    # 在测试集上预测（使用相同的数据处理流水线）
    X_test_scaled = scaler.transform(X_test)
    X_test_selected = selector.transform(X_test_scaled)
    X_test_pca = pca.transform(X_test_selected)
    
    X_test_tensor = torch.FloatTensor(X_test_pca)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    with torch.no_grad():
        test_preds_log = model(X_test_tensor.to(device)).cpu().numpy()
        test_preds_fold = np.expm1(test_preds_log.flatten())
        test_preds.append(test_preds_fold)

# 9. 计算整体验证分数
all_val_preds = np.array(all_val_preds)
all_val_targets = np.array(all_val_targets)

overall_rmse = np.sqrt(np.mean((all_val_preds - all_val_targets) ** 2))
overall_score = 1.0 / (1.0 + overall_rmse)

print(f"\n{'='*50}")
print(f"交叉验证结果:")
for i, score in enumerate(fold_scores, 1):
    print(f"  Fold {i}: Score = {score:.4f}")

print(f"\n整体验证集 RMSE: {overall_rmse:.4f}")
print(f"整体 Score = (1.0 / (1.0 + RMSE)) = {overall_score:.4f}")

# 10. 集成测试集预测（加权平均，权重为每个fold的分数）
test_preds = np.array(test_preds)
fold_weights = np.array(fold_scores) / np.sum(fold_scores)
print(f"\nFold权重: {fold_weights}")

# 加权平均
final_test_preds = np.zeros(test_preds.shape[1])
for i, weight in enumerate(fold_weights):
    final_test_preds += test_preds[i] * weight

# 11. 后处理
# 确保预测值为非负
final_test_preds = np.maximum(final_test_preds, 0)

# 基于领域知识的调整：阳性率通常在一定范围内
# 我们可以使用温和的缩放到合理的范围，但不使用硬截断
q99 = np.percentile(all_val_targets, 99)
final_test_preds = np.minimum(final_test_preds, q99 * 1.5)  # 允许略高于训练集最大值

# 12. 生成提交文件
submission = pd.DataFrame({
    'id': test_df['id'],
    'tested_positive': final_test_preds
})

submission.to_csv('submission.csv', index=False)
print(f"\n提交文件已保存，包含 {len(submission)} 条预测结果")

# 13. 打印最终分数（按要求格式）
print(f"Score= (1.0 / (1.0 + RMSE)) = {overall_score:.4f}")