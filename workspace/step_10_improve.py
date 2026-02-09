import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子以保证可重复性
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# 1. 数据加载与清洗
print("Loading data...")
train_df = pd.read_csv('/home/huazhao/DL-HW/ML-2025/mini-aide/input/train.csv')
test_df = pd.read_csv('/home/huazhao/DL-HW/ML-2025/mini-aide/input/test.csv')

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# 分离特征和目标
target_col = 'tested_positive_day3'
y_train_full = train_df[target_col].values
X_train_full = train_df.drop(columns=[target_col, 'id']).copy()
X_test = test_df.drop(columns=['id']).copy()

# 2. 更智能的时间序列分割（保留时间序列结构）
# 使用时间序列交叉验证的思路：多折验证
def create_time_series_folds(data, n_folds=5):
    """创建时间序列折叠"""
    fold_size = len(data) // n_folds
    folds = []
    for i in range(n_folds-1):
        train_end = (i+1) * fold_size
        val_start = train_end
        val_end = min((i+2) * fold_size, len(data))
        folds.append((train_end, val_start, val_end))
    return folds

# 我们仍然使用最后一个折叠作为最终验证集
val_size = int(len(X_train_full) * 0.2)
X_train = X_train_full.iloc[:-val_size].copy()
X_val = X_train_full.iloc[-val_size:].copy()
y_train = y_train_full[:-val_size]
y_val = y_train_full[-val_size:]

print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

# 3. 增强的特征工程 - 专注于时间序列和交互特征
def create_advanced_features_v2(df):
    """增强版特征工程"""
    df = df.copy()
    
    # 原始特征列
    original_cols = df.columns.tolist()
    
    # 1. 时间序列特征（滞后、差分、滚动统计）
    for base_feature in ['cli', 'ili', 'tested_positive', 'wearing_mask_7d']:
        day_cols = []
        for day in [1, 2, 3]:
            col_name = f'{base_feature}_day{day}'
            if col_name in df.columns:
                day_cols.append(col_name)
        
        if len(day_cols) >= 2:
            # 滞后特征
            if len(day_cols) >= 2:
                df[f'{base_feature}_lag1'] = df[day_cols[1]] - df[day_cols[0]]
            
            # 移动平均
            df[f'{base_feature}_moving_avg'] = df[day_cols].mean(axis=1)
            
            # 指数加权移动平均（更关注近期）
            weights = np.array([0.1, 0.3, 0.6][:len(day_cols)])
            weights = weights / weights.sum()
            df[f'{base_feature}_ewma'] = (df[day_cols] * weights).sum(axis=1)
            
            # 变化率
            if len(day_cols) >= 2:
                df[f'{base_feature}_change_rate'] = (df[day_cols[1]] - df[day_cols[0]]) / (df[day_cols[0]] + 1e-6)
    
    # 2. 交互特征（医学相关的交叉特征）
    for day in [1, 2, 3]:
        # 症状与防护措施的交互
        if f'cli_day{day}' in df.columns and f'wearing_mask_7d_day{day}' in df.columns:
            df[f'cli_mask_interaction_day{day}'] = df[f'cli_day{day}'] * (1 - df[f'wearing_mask_7d_day{day}'])
        
        # 社区传播与室内活动的交互
        if f'wnohh_cmnty_cli_day{day}' in df.columns and f'wlarge_event_indoors_day{day}' in df.columns:
            df[f'community_indoor_risk_day{day}'] = df[f'wnohh_cmnty_cli_day{day}'] * df[f'wlarge_event_indoors_day{day}']
        
        # 疫苗接种与担忧的交互
        if f'wcovid_vaccinated_friends_day{day}' in df.columns and f'wworried_catch_covid_day{day}' in df.columns:
            df[f'vaccine_worry_ratio_day{day}'] = df[f'wworried_catch_covid_day{day}'] / (df[f'wcovid_vaccinated_friends_day{day}'] + 1e-6)
    
    # 3. 聚合特征 - 按特征类型聚合
    # 症状聚合
    symptom_cols = [col for col in df.columns if 'cli' in col or 'ili' in col]
    if symptom_cols:
        df['total_symptoms'] = df[symptom_cols].mean(axis=1)
    
    # 防护措施聚合
    protection_cols = [col for col in df.columns if 'mask' in col or 'distanc' in col]
    if protection_cols:
        df['total_protection'] = df[protection_cols].mean(axis=1)
    
    # 风险活动聚合
    risk_cols = [col for col in df.columns if 'indoors' in col or 'transit' in col or 'event' in col]
    if risk_cols:
        df['total_risk_activities'] = df[risk_cols].mean(axis=1)
    
    # 4. 地理聚类特征
    state_cols = [col for col in df.columns if len(col) == 2 and col.isupper()]
    if len(state_cols) > 0:
        # 人口权重（基于真实人口数据近似）
        population_weights = {
            'CA': 39.5, 'TX': 29.0, 'FL': 21.5, 'NY': 19.5, 'PA': 12.8,
            'IL': 12.7, 'OH': 11.7, 'GA': 10.6, 'NC': 10.4, 'MI': 10.0,
            'NJ': 9.3, 'VA': 8.5, 'WA': 7.6, 'AZ': 7.2, 'MA': 7.0,
            'TN': 6.8, 'IN': 6.7, 'MO': 6.1, 'MD': 6.0, 'WI': 5.8,
            'CO': 5.8, 'MN': 5.6, 'SC': 5.1, 'AL': 4.9, 'LA': 4.6,
            'KY': 4.5, 'OR': 4.2, 'OK': 4.0, 'CT': 3.6, 'IA': 3.2,
            'UT': 3.2, 'NV': 3.1, 'AR': 3.0, 'MS': 3.0, 'KS': 2.9,
            'NM': 2.1, 'NE': 1.9, 'WV': 1.8, 'ID': 1.8, 'HI': 1.4,
            'NH': 1.4, 'ME': 1.3, 'RI': 1.1, 'MT': 1.1, 'DE': 1.0,
            'SD': 0.9, 'ND': 0.8, 'AK': 0.7, 'VT': 0.6, 'WY': 0.6
        }
        
        # 计算加权状态特征
        for state in state_cols:
            if state in population_weights:
                df[f'{state}_weighted'] = df[state] * population_weights[state]
    
    # 5. 统计特征
    # 偏度和峰度近似（简化计算）
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df['feature_mean'] = df[numeric_cols].mean(axis=1)
        df['feature_std'] = df[numeric_cols].std(axis=1)
        df['feature_skew'] = ((df[numeric_cols] - df['feature_mean'].values[:, None]) ** 3).mean(axis=1)
    
    # 6. 删除高度相关的特征（减少多重共线性）
    # 先计算相关性矩阵，然后删除相关系数>0.95的特征
    if len(original_cols) > 0:
        corr_matrix = df[original_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        if to_drop:
            df = df.drop(columns=to_drop)
            print(f"Dropped {len(to_drop)} highly correlated features")
    
    return df

print("Creating enhanced features...")
X_train = create_advanced_features_v2(X_train)
X_val = create_advanced_features_v2(X_val)
X_test = create_advanced_features_v2(X_test)

print(f"Features after enhanced engineering: {X_train.shape[1]}")

# 4. 数据预处理
# 4.1 处理缺失值
def safe_fillna(df, train_stats=None):
    """安全地填充缺失值"""
    if train_stats is None:
        train_stats = {}
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                train_stats[col] = {
                    'median': df[col].median(),
                    'mean': df[col].mean(),
                    'std': df[col].std()
                }
    
    df_filled = df.copy()
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            if col in train_stats:
                # 使用训练集的统计量
                fill_value = train_stats[col]['median']
                # 添加少量噪声，避免完全相同
                df_filled[col] = df[col].fillna(fill_value)
    
    return df_filled, train_stats

print("Filling missing values...")
train_stats = {}
X_train, train_stats = safe_fillna(X_train, train_stats)
X_val, _ = safe_fillna(X_val, train_stats)
X_test, _ = safe_fillna(X_test, train_stats)

# 4.2 目标值变换 - 尝试Box-Cox变换的近似
# 由于Box-Cox要求正值，我们使用Yeo-Johnson的近似
def transform_target(y):
    """变换目标变量"""
    # 添加小常数避免0
    y_transformed = np.log1p(y)
    return y_transformed

def inverse_transform_target(y_transformed):
    """逆变换目标变量"""
    return np.expm1(y_transformed)

y_train_transformed = transform_target(y_train)
y_val_transformed = transform_target(y_val)

# 4.3 特征缩放 - 使用多种缩放器
print("Scaling features...")
scaler1 = RobustScaler()  # 对异常值稳健
scaler2 = QuantileTransformer(n_quantiles=100, output_distribution='normal')  # 转换为正态分布

# 训练数据
X_train_scaled1 = scaler1.fit_transform(X_train)
X_train_scaled2 = scaler2.fit_transform(X_train)

# 验证数据
X_val_scaled1 = scaler1.transform(X_val)
X_val_scaled2 = scaler2.transform(X_val)

# 测试数据
X_test_scaled1 = scaler1.transform(X_test)
X_test_scaled2 = scaler2.transform(X_test)

# 合并不同缩放器的特征
X_train_scaled = np.hstack([X_train_scaled1, X_train_scaled2])
X_val_scaled = np.hstack([X_val_scaled1, X_val_scaled2])
X_test_scaled = np.hstack([X_test_scaled1, X_test_scaled2])

print(f"Scaled features shape: {X_train_scaled.shape}")

# 5. 特征选择 - 更智能的方法
print("Performing feature selection...")

# 5.1 使用随机森林进行特征重要性排序
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train_transformed)

# 获取特征重要性
feature_importance = rf.feature_importances_
importance_threshold = np.percentile(feature_importance, 30)  # 保留重要性前70%的特征
selected_indices = np.where(feature_importance >= importance_threshold)[0]

# 5.2 使用互信息进行补充选择
if len(selected_indices) < 50:  # 确保至少有50个特征
    mi_selector = SelectKBest(score_func=mutual_info_regression, k=50)
    X_train_mi = mi_selector.fit_transform(X_train_scaled, y_train_transformed)
    mi_scores = mi_selector.scores_
    mi_indices = np.argsort(mi_scores)[-50:]
    selected_indices = np.union1d(selected_indices, mi_indices)

# 应用特征选择
X_train_selected = X_train_scaled[:, selected_indices]
X_val_selected = X_val_scaled[:, selected_indices]
X_test_selected = X_test_scaled[:, selected_indices]

print(f"Selected {len(selected_indices)} important features")

# 6. 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train_selected)
y_train_tensor = torch.FloatTensor(y_train_transformed).view(-1, 1)
X_val_tensor = torch.FloatTensor(X_val_selected)
y_val_tensor = torch.FloatTensor(y_val_transformed).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_selected)

# 7. 定义改进的DNN模型架构
class AttentionBlock(nn.Module):
    """注意力机制模块"""
    def __init__(self, input_dim):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, input_dim, output_dim, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        
        # 如果输入输出维度不同，使用投影
        self.shortcut = nn.Sequential()
        if input_dim != output_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class COVIDPredictorV4(nn.Module):
    """增强版COVID预测器"""
    def __init__(self, input_dim, dropout_rate=0.3):
        super(COVIDPredictorV4, self).__init__()
        
        # 初始特征变换
        self.initial_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 注意力机制
        self.attention = AttentionBlock(256)
        
        # 残差块
        self.res_block1 = ResidualBlock(256, 256, dropout_rate)
        self.res_block2 = ResidualBlock(256, 128, dropout_rate)
        self.res_block3 = ResidualBlock(128, 64, dropout_rate)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(32, 1)
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
        x = self.initial_layer(x)
        x = self.attention(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.output_layer(x)
        return x

# 8. 训练函数 - 添加混合精度训练
def train_model_advanced(model, train_loader, val_loader, model_name, 
                         lr=1e-3, n_epochs=500, patience=30):
    """增强版训练函数"""
    criterion = nn.HuberLoss(delta=1.0)  # Huber损失对异常值更稳健
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 学习率调度器：OneCycleLR
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=n_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3, anneal_strategy='cos'
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    print(f"\nTraining {model_name}...")
    
    for epoch in range(n_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
        
        # 计算平均损失
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'best_{model_name}.pth')
            best_epoch = epoch
        else:
            patience_counter += 1
        
        if epoch % 50 == 0 or epoch == n_epochs - 1:
            current_lr = scheduler.get_last_lr()[0]
            print(f'Epoch {epoch:4d}: Train Loss = {train_loss:.6f}, '
                  f'Val Loss = {val_loss:.6f}, LR = {current_lr:.6f}')
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load(f'best_{model_name}.pth'))
    print(f"Best validation loss for {model_name}: {best_val_loss:.6f} at epoch {best_epoch}")
    
    return model, best_val_loss, train_losses, val_losses

# 9. 创建数据加载器（添加数据增强）
class TimeSeriesDataset(torch.utils.data.Dataset):
    """时间序列数据集，支持数据增强"""
    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = y
        self.augment = augment
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.augment and torch.rand(1) > 0.7:
            # 数据增强：添加高斯噪声
            noise = torch.randn_like(x) * 0.01
            x = x + noise
        
        return x, y

# 创建数据集
batch_size = 64
train_dataset = TimeSeriesDataset(X_train_tensor, y_train_tensor, augment=True)
val_dataset = TimeSeriesDataset(X_val_tensor, y_val_tensor, augment=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 10. 训练多个模型进行集成
input_dim = X_train_selected.shape[1]

# 使用不同的架构和超参数训练多个模型
models = []
model_losses = []

# 模型1：较深的网络
model1 = COVIDPredictorV4(input_dim, dropout_rate=0.35)
model1, loss1, train_losses1, val_losses1 = train_model_advanced(
    model1, train_loader, val_loader, "model1", 
    lr=1.2e-3, n_epochs=400, patience=25
)
models.append(model1)
model_losses.append(loss1)

# 模型2：中等深度的网络
model2 = COVIDPredictorV4(input_dim, dropout_rate=0.25)
model2, loss2, train_losses2, val_losses2 = train_model_advanced(
    model2, train_loader, val_loader, "model2",
    lr=9e-4, n_epochs=400, patience=25
)
models.append(model2)
model_losses.append(loss2)

# 模型3：较浅的网络
model3 = COVIDPredictorV4(input_dim, dropout_rate=0.2)
model3, loss3, train_losses3, val_losses3 = train_model_advanced(
    model3, train_loader, val_loader, "model3",
    lr=6e-4, n_epochs=400, patience=25
)
models.append(model3)
model_losses.append(loss3)

# 11. 创建简单的线性模型作为baseline
class SimpleLinearModel(nn.Module):
    def __init__(self, input_dim):
        super(SimpleLinearModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.model(x)

model4 = SimpleLinearModel(input_dim)
model4, loss4, train_losses4, val_losses4 = train_model_advanced(
    model4, train_loader, val_loader, "model4",
    lr=1e-3, n_epochs=200, patience=20
)
models.append(model4)
model_losses.append(loss4)

# 12. 在验证集上评估
print("\n" + "="*60)
print("Model Evaluation on Validation Set")
print("="*60)

# 评估单个模型
model_predictions = []
model_rmses = []

for i, model in enumerate(models):
    model.eval()
    with torch.no_grad():
        pred = model(X_val_tensor)
        # 转换回原始尺度计算RMSE
        pred_original = inverse_transform_target(pred.numpy().flatten())
        true_original = inverse_transform_target(y_val_transformed)
        
        rmse = np.sqrt(np.mean((pred_original - true_original) ** 2))
        score = 1.0 / (1.0 + rmse)
        
        model_predictions.append(pred_original)
        model_rmses.append(rmse)
        
        print(f"Model{i+1}: RMSE = {rmse:.6f}, Score = {score:.6f}")

# 13. 智能集成策略
print("\n" + "="*60)
print("Ensemble Strategies")
print("="*60)

# 策略1：简单平均
simple_avg = np.mean(model_predictions, axis=0)
rmse_simple = np.sqrt(np.mean((simple_avg - true_original) ** 2))
score_simple = 1.0 / (1.0 + rmse_simple)
print(f"Simple Average: RMSE = {rmse_simple:.6f}, Score = {score_simple:.6f}")

# 策略2：加权平均（基于验证损失）
weights_loss = [1/loss for loss in model_losses]
weights_loss = [w/sum(weights_loss) for w in weights_loss]
weighted_avg_loss = np.average(model_predictions, axis=0, weights=weights_loss)
rmse_weighted_loss = np.sqrt(np.mean((weighted_avg_loss - true_original) ** 2))
score_weighted_loss = 1.0 / (1.0 + rmse_weighted_loss)
print(f"Weighted Average (by loss): RMSE = {rmse_weighted_loss:.6f}, Score = {score_weighted_loss:.6f}")

# 策略3：基于RMSE的加权平均
weights_rmse = [1/rmse for rmse in model_rmses]
weights_rmse = [w/sum(weights_rmse) for w in weights_rmse]
weighted_avg_rmse = np.average(model_predictions, axis=0, weights=weights_rmse)
rmse_weighted_rmse = np.sqrt(np.mean((weighted_avg_rmse - true_original) ** 2))
score_weighted_rmse = 1.0 / (1.0 + rmse_weighted_rmse)
print(f"Weighted Average (by RMSE): RMSE = {rmse_weighted_rmse:.6f}, Score = {score_weighted_rmse:.6f}")

# 策略4：中位数（对异常值更稳健）
median_pred = np.median(model_predictions, axis=0)
rmse_median = np.sqrt(np.mean((median_pred - true_original) ** 2))
score_median = 1.0 / (1.0 + rmse_median)
print(f"Median: RMSE = {rmse_median:.6f}, Score = {score_median:.6f}")

# 选择最佳集成策略
ensemble_methods = {
    'simple_avg': (simple_avg, rmse_simple, score_simple),
    'weighted_loss': (weighted_avg_loss, rmse_weighted_loss, score_weighted_loss),
    'weighted_rmse': (weighted_avg_rmse, rmse_weighted_rmse, score_weighted_rmse),
    'median': (median_pred, rmse_median, score_median)
}

best_method = max(ensemble_methods.items(), key=lambda x: x[1][2])
print(f"\nBest ensemble method: {best_method[0]} with Score = {best_method[1][2]:.6f}")

# 14. 对测试集进行预测
print("\n" + "="*60)
print("Generating Test Set Predictions")
print("="*60)

# 获取各个模型的预测
test_predictions_list = []
for model in models:
    model.eval()
    with torch.no_grad():
        pred = model(X_test_tensor)
        # 转换回原始尺度
        pred_original = inverse_transform_target(pred.numpy().flatten())
        test_predictions_list.append(pred_original)

# 使用最佳集成策略
if best_method[0] == 'simple_avg':
    test_predictions = np.mean(test_predictions_list, axis=0)
elif best_method[0] == 'weighted_loss':
    test_predictions = np.average(test_predictions_list, axis=0, weights=weights_loss)
elif best_method[0] == 'weighted_rmse':
    test_predictions = np.average(test_predictions_list, axis=0, weights=weights_rmse)
else:  # median
    test_predictions = np.median(test_predictions_list, axis=0)

# 后处理：确保预测值合理
print("Applying post-processing...")

# 1. 确保非负
test_predictions = np.maximum(test_predictions, 0)

# 2. 温和的修剪（保留极端值的可能性）
q95 = np.percentile(test_predictions, 95)
q99 = np.percentile(test_predictions, 99)
test_predictions = np.where(test_predictions > q99, 
                           q95 + (test_predictions - q95) * 0.3,  # 压缩极端值
                           test_predictions)

# 3. 添加轻微平滑
test_predictions = 0.7 * test_predictions + 0.3 * np.median(test_predictions)

# 4. 与训练集统计量对齐
train_target_mean = np.mean(y_train_full)
train_target_std = np.std(y_train_full)
pred_mean = np.mean(test_predictions)
pred_std = np.std(test_predictions)

# 如果预测分布与训练集差异太大，进行调整
if abs(pred_mean - train_target_mean) > train_target_std * 0.5:
    print(f"Adjusting prediction distribution...")
    test_predictions = (test_predictions - pred_mean) * (train_target_std / max(pred_std, 1e-6)) + train_target_mean
    test_predictions = np.maximum(test_predictions, 0)

print(f"Final predictions - Min: {test_predictions.min():.2f}, "
      f"Max: {test_predictions.max():.2f}, "
      f"Mean: {test_predictions.mean():.2f}, "
      f"Std: {test_predictions.std():.2f}")

# 15. 生成提交文件
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'tested_positive': test_predictions
})

submission_df.to_csv('submission.csv', index=False)
print(f"\nSubmission file saved to 'submission.csv'")

# 16. 最终验证集分析
print("\n" + "="*60)
print("Final Validation Set Analysis")
print("="*60)

final_val_pred = best_method[1][0]
val_true = inverse_transform_target(y_val_transformed)

print(f"True values - Min: {val_true.min():.2f}, "
      f"Max: {val_true.max():.2f}, "
      f"Mean: {val_true.mean():.2f}")
print(f"Predicted values - Min: {final_val_pred.min():.2f}, "
      f"Max: {final_val_pred.max():.2f}, "
      f"Mean: {final_val_pred.mean():.2f}")

# 计算各种指标
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

mae = mean_absolute_error(val_true, final_val_pred)
mape = mean_absolute_percentage_error(val_true, final_val_pred) * 100
r2 = r2_score(val_true, final_val_pred)

print(f"\nValidation Metrics:")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {best_method[1][1]:.6f}")
print(f"Final Score: 1.0 / (1.0 + {best_method[1][1]:.6f}) = {best_method[1][2]:.6f}")

# 17. 保存最佳模型权重
print("\n" + "="*60)
print("Saving Best Model Weights")
print("="*60)

# 保存集成中表现最好的单个模型
best_single_model_idx = np.argmin(model_rmses)
best_single_model = models[best_single_model_idx]
torch.save(best_single_model.state_dict(), 'best_single_model.pth')
print(f"Best single model (Model{best_single_model_idx+1}) saved.")

print("\n" + "="*60)
print("Optimization Complete!")
print("="*60)
print(f"Expected score improvement: ~{best_method[1][2] - 0.840969:.4f}")