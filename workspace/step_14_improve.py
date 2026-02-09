import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
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

# 2. 时间序列分割（最后5%作为验证集，不打乱）
val_size = int(len(X_train_full) * 0.05)  # 更小的验证集，更多训练数据
X_train = X_train_full.iloc[:-val_size].copy()
X_val = X_train_full.iloc[-val_size:].copy()
y_train = y_train_full[:-val_size]
y_val = y_train_full[-val_size:]

print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

# 3. 改进的特征工程
def create_optimized_features(df):
    df = df.copy()
    
    # 原始特征列
    feature_cols = [col for col in df.columns if col != 'id']
    
    # 1. 创建症状严重程度指标
    for day in ['1', '2', '3']:
        if f'cli_day{day}' in df.columns and f'ili_day{day}' in df.columns:
            df[f'symptom_severity_day{day}'] = df[f'cli_day{day}'] * 0.7 + df[f'ili_day{day}'] * 0.3
    
    # 2. 创建社交活动风险复合指标
    for day in ['1', '2', '3']:
        indoor_risk_cols = []
        if f'wlarge_event_indoors_day{day}' in df.columns:
            indoor_risk_cols.append(f'wlarge_event_indoors_day{day}')
        if f'wshop_indoors_day{day}' in df.columns:
            indoor_risk_cols.append(f'wshop_indoors_day{day}')
        if f'wrestaurant_indoors_day{day}' in df.columns:
            indoor_risk_cols.append(f'wrestaurant_indoors_day{day}')
        if f'public_transit_day{day}' in df.columns:
            indoor_risk_cols.append(f'public_transit_day{day}')
            
        if indoor_risk_cols:
            df[f'indoor_risk_composite_day{day}'] = df[indoor_risk_cols].mean(axis=1)
    
    # 3. 创建防护行为有效性指标
    for day in ['1', '2', '3']:
        if (f'wbelief_masking_effective_day{day}' in df.columns and 
            f'wearing_mask_7d_day{day}' in df.columns):
            df[f'protection_effectiveness_day{day}'] = (
                df[f'wbelief_masking_effective_day{day}'] * df[f'wearing_mask_7d_day{day}'])
    
    # 4. 创建社区传播风险指标
    for day in ['1', '2', '3']:
        community_risk_cols = []
        if f'wnohh_cmnty_cli_day{day}' in df.columns:
            community_risk_cols.append(f'wnohh_cmnty_cli_day{day}')
        if f'hh_cmnty_cli_day{day}' in df.columns:
            community_risk_cols.append(f'hh_cmnty_cli_day{day}')
        if f'nohh_cmnty_cli_day{day}' in df.columns:
            community_risk_cols.append(f'nohh_cmnty_cli_day{day}')
            
        if community_risk_cols:
            df[f'community_transmission_day{day}'] = df[community_risk_cols].mean(axis=1)
    
    # 5. 创建跨天趋势特征（重点优化）
    for feature in ['tested_positive', 'cli', 'ili', 'wearing_mask_7d']:
        cols = [f'{feature}_day{i}' for i in [1, 2, 3] if f'{feature}_day{i}' in df.columns]
        if len(cols) >= 2:
            # 最近两天变化率
            df[f'{feature}_growth_rate'] = (df[cols[1]] - df[cols[0]]) / (df[cols[0]] + 1)
            # 加权平均值（最近的天数权重更高）
            weights = [0.2, 0.3, 0.5] if len(cols) == 3 else [0.3, 0.7]
            df[f'{feature}_weighted_avg'] = sum(df[col] * weight for col, weight in zip(cols, weights[-len(cols):]))
    
    # 6. 创建交互特征
    for day in ['1', '2', '3']:
        if (f'tested_positive_day{day}' in df.columns and 
            f'wworried_catch_covid_day{day}' in df.columns):
            df[f'risk_perception_index_day{day}'] = (
                df[f'tested_positive_day{day}'] * df[f'wworried_catch_covid_day{day}'])
        
        if (f'wbelief_distancing_effective_day{day}' in df.columns and 
            f'wothers_distanced_public_day{day}' in df.columns):
            df[f'distancing_compliance_day{day}'] = (
                df[f'wbelief_distancing_effective_day{day}'] + df[f'wothers_distanced_public_day{day}']) / 2
    
    # 7. 创建区域人口权重特征（基于人口大州）
    # 美国人口最多的10个州
    top_population_states = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI']
    
    # 为这些州创建加权特征
    state_cols = [col for col in df.columns if len(col) == 2 and col.isupper()]
    important_state_cols = [state for state in state_cols if state in top_population_states]
    
    if important_state_cols:
        # 计算人口权重（简化版）
        population_weights = {
            'CA': 39.5, 'TX': 29.1, 'FL': 21.5, 'NY': 19.5, 'PA': 12.8,
            'IL': 12.7, 'OH': 11.7, 'GA': 10.6, 'NC': 10.4, 'MI': 10.0
        }
        
        weighted_state_sum = 0
        for state in important_state_cols:
            if state in population_weights:
                weighted_state_sum += df[state] * population_weights[state]
        
        if important_state_cols:
            df['population_weighted_state'] = weighted_state_sum / len(important_state_cols)
    
    # 8. 创建时间滞后特征差异
    for feature in ['tested_positive', 'cli', 'ili']:
        if (f'{feature}_day2' in df.columns and f'{feature}_day1' in df.columns):
            df[f'{feature}_day2_day1_diff'] = df[f'{feature}_day2'] - df[f'{feature}_day1']
        if (f'{feature}_day3' in df.columns and f'{feature}_day2' in df.columns):
            df[f'{feature}_day3_day2_diff'] = df[f'{feature}_day3'] - df[f'{feature}_day2']
    
    # 9. 删除低方差的州特征（保留重要的）
    state_cols = [col for col in df.columns if len(col) == 2 and col.isupper()]
    # 计算每个州特征的方差
    state_variances = {}
    for state in state_cols:
        state_variances[state] = df[state].var()
    
    # 保留方差最高的10个州
    if state_variances:
        top_states = sorted(state_variances.items(), key=lambda x: x[1], reverse=True)[:10]
        top_state_names = [state for state, var in top_states]
        cols_to_drop = [state for state in state_cols if state not in top_state_names]
        df = df.drop(columns=cols_to_drop)
    
    return df

print("Creating optimized features...")
X_train = create_optimized_features(X_train)
X_val = create_optimized_features(X_val)
X_test = create_optimized_features(X_test)

print(f"Features after optimized engineering: {X_train.shape[1]}")

# 4. 数据预处理
# 4.1 填充缺失值
X_train = X_train.fillna(X_train.median())
X_val = X_val.fillna(X_train.median())
X_test = X_test.fillna(X_train.median())

# 4.2 目标值变换（使用Box-Cox变换的近似）
y_train_transformed = np.log1p(y_train)  # log(1 + y)
y_val_transformed = np.log1p(y_val)

# 4.3 特征缩放 - 使用QuantileTransformer处理非正态分布
scaler = QuantileTransformer(n_quantiles=100, output_distribution='normal', random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"Feature scaling complete. Shape: {X_train_scaled.shape}")

# 5. 特征选择（更保守的方法）
print("Performing feature selection...")

# 5.1 基于相关性的初步筛选
correlation_threshold = 0.05
if X_train_scaled.shape[1] > 50:
    # 计算与目标的相关性（使用互信息）
    from sklearn.feature_selection import mutual_info_regression
    
    mi_scores = mutual_info_regression(X_train_scaled, y_train_transformed, random_state=42)
    mi_scores_series = pd.Series(mi_scores, index=range(X_train_scaled.shape[1]))
    
    # 选择前60个特征
    selected_indices = mi_scores_series.nlargest(60).index.tolist()
    
    X_train_selected = X_train_scaled[:, selected_indices]
    X_val_selected = X_val_scaled[:, selected_indices]
    X_test_selected = X_test_scaled[:, selected_indices]
    
    print(f"Selected {len(selected_indices)} features based on mutual information")
else:
    # 如果特征不多，直接使用所有特征
    X_train_selected = X_train_scaled
    X_val_selected = X_val_scaled
    X_test_selected = X_test_scaled
    selected_indices = list(range(X_train_scaled.shape[1]))

# 6. 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train_selected)
y_train_tensor = torch.FloatTensor(y_train_transformed).view(-1, 1)
X_val_tensor = torch.FloatTensor(X_val_selected)
y_val_tensor = torch.FloatTensor(y_val_transformed).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_selected)

# 创建数据加载器
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 7. 定义改进的DNN模型架构
class EnhancedCOVIDPredictor(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.2):
        super(EnhancedCOVIDPredictor, self).__init__()
        
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            
            # Layer 2
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate * 0.8),
            
            # Layer 3
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate * 0.6),
            
            # Layer 4
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate * 0.4),
            
            # Layer 5
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate * 0.2),
            
            # Output layer
            nn.Linear(32, 1)
        )
        
        # 添加残差连接（如果维度匹配）
        self.residual = nn.Linear(input_dim, 1) if input_dim != 1 else None
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        out = self.network(x)
        if self.residual is not None:
            out = out + self.residual(x)
        return out

# 8. 定义多样化的模型架构
class ResidualCOVIDPredictor(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.15):
        super(ResidualCOVIDPredictor, self).__init__()
        
        # 第一个残差块
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        self.shortcut1 = nn.Linear(input_dim, 128) if input_dim != 128 else nn.Identity()
        
        # 第二个残差块
        self.block2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        self.shortcut2 = nn.Linear(128, 32) if 128 != 32 else nn.Identity()
        
        # 输出层
        self.output = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        # 第一个残差块
        identity1 = self.shortcut1(x)
        out1 = self.block1(x)
        out1 = out1 + identity1
        out1 = nn.functional.relu(out1)
        
        # 第二个残差块
        identity2 = self.shortcut2(out1)
        out2 = self.block2(out1)
        out2 = out2 + identity2
        out2 = nn.functional.relu(out2)
        
        # 输出
        return self.output(out2)

# 9. 训练函数（改进版）
def train_model_advanced(model, train_loader, val_loader, model_name, lr=1e-3, n_epochs=400):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5, betas=(0.9, 0.999))
    
    # 使用余弦退火学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-6)
    
    best_val_loss = float('inf')
    patience = 25
    patience_counter = 0
    best_model_state = None
    
    print(f"\nTraining {model_name}...")
    
    # 训练历史记录
    train_losses = []
    val_losses = []
    
    for epoch in range(n_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            
            # 添加L2正则化
            l2_lambda = 1e-5
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = loss + l2_lambda * l2_norm
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
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
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
        else:
            patience_counter += 1
        
        if epoch % 40 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f'Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, LR = {current_lr:.6f}')
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"Best validation loss for {model_name}: {best_val_loss:.6f} at epoch {best_epoch}")
    
    return model, best_val_loss, train_losses, val_losses

# 10. 训练多个模型进行集成
input_dim = X_train_selected.shape[1]
print(f"Input dimension: {input_dim}")

# 模型1：深网络
model1 = EnhancedCOVIDPredictor(input_dim, dropout_rate=0.25)
model1, loss1, train_losses1, val_losses1 = train_model_advanced(
    model1, train_loader, val_loader, "model1_deep", lr=1.2e-3, n_epochs=350)

# 模型2：残差网络
model2 = ResidualCOVIDPredictor(input_dim, dropout_rate=0.2)
model2, loss2, train_losses2, val_losses2 = train_model_advanced(
    model2, train_loader, val_loader, "model2_residual", lr=9e-4, n_epochs=350)

# 模型3：中等深度的网络
class MediumCOVIDPredictor(nn.Module):
    def __init__(self, input_dim):
        super(MediumCOVIDPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)

model3 = MediumCOVIDPredictor(input_dim)
model3, loss3, train_losses3, val_losses3 = train_model_advanced(
    model3, train_loader, val_loader, "model3_medium", lr=1e-3, n_epochs=300)

# 11. 在验证集上评估和集成
def evaluate_ensemble(models, X_val_tensor, y_val_tensor, weights=None):
    if weights is None:
        # 根据验证损失分配权重（损失越小权重越大）
        losses = [loss1, loss2, loss3]
        # 使用softmax-like权重分配
        weights = [np.exp(-loss) for loss in losses]
        weights = [w/sum(weights) for w in weights]
        print(f"Model weights: {weights}")
    
    ensemble_predictions = 0
    individual_predictions = []
    
    for i, model in enumerate(models):
        model.eval()
        with torch.no_grad():
            pred = model(X_val_tensor)
            individual_predictions.append(pred)
            ensemble_predictions += weights[i] * pred
    
    # 计算RMSE
    mse = nn.MSELoss()(ensemble_predictions, y_val_tensor).item()
    rmse = np.sqrt(mse)
    
    return rmse, ensemble_predictions, individual_predictions, weights

# 评估单个模型
print("\n" + "="*60)
print("Individual model performance:")
print("="*60)

models = [model1, model2, model3]
model_names = ["Deep Network", "Residual Network", "Medium Network"]
individual_rmses = []

for name, model in zip(model_names, models):
    model.eval()
    with torch.no_grad():
        pred = model(X_val_tensor)
        mse = nn.MSELoss()(pred, y_val_tensor).item()
        rmse = np.sqrt(mse)
        individual_rmses.append(rmse)
        score = 1.0 / (1.0 + rmse)
        print(f"{name}: RMSE = {rmse:.6f}, Score = {score:.6f}")

# 集成评估
ensemble_rmse, ensemble_pred, individual_preds, ensemble_weights = evaluate_ensemble(
    models, X_val_tensor, y_val_tensor)

# 计算Score
ensemble_score = 1.0 / (1.0 + ensemble_rmse)

print("\n" + "="*60)
print("Ensemble performance:")
print("="*60)
print(f"Ensemble RMSE: {ensemble_rmse:.6f}")
print(f"Score = 1.0 / (1.0 + {ensemble_rmse:.6f}) = {ensemble_score:.6f}")
print("\n" + "="*60)

# 12. 对测试集进行预测（集成）
print("\nGenerating predictions for test set...")

# 获取各个模型的预测
test_predictions_list = []
for model in models:
    model.eval()
    with torch.no_grad():
        pred = model(X_test_tensor).numpy().flatten()
        test_predictions_list.append(pred)

# 加权集成
test_predictions_ensemble = np.zeros_like(test_predictions_list[0])
for i, pred in enumerate(test_predictions_list):
    test_predictions_ensemble += ensemble_weights[i] * pred

# 将预测值转换回原始尺度（逆对数变换）
test_predictions = np.expm1(test_predictions_ensemble)

# 13. 后处理：使用验证集统计信息校准预测
# 计算验证集上的缩放因子
val_predictions_transformed = ensemble_pred.numpy().flatten()
val_predictions = np.expm1(val_predictions_transformed)
val_true = np.expm1(y_val_transformed)

# 计算分位数映射
from scipy import stats
def quantile_mapping(predictions, source_quantiles, target_quantiles):
    """将预测值的分位数映射到目标值的分位数"""
    mapped = np.zeros_like(predictions)
    for i in range(len(predictions)):
        # 找到预测值在源分布中的位置
        idx = np.searchsorted(source_quantiles, predictions[i])
        idx = min(max(idx, 0), len(target_quantiles)-1)
        mapped[i] = target_quantiles[idx]
    return mapped

# 创建分位数映射
source_quantiles = np.percentile(val_predictions, np.linspace(0, 100, 101))
target_quantiles = np.percentile(val_true, np.linspace(0, 100, 101))

# 应用分位数映射
test_predictions_calibrated = quantile_mapping(test_predictions, source_quantiles, target_quantiles)

# 确保预测值非负
test_predictions_calibrated = np.maximum(test_predictions_calibrated, 0)

# 温和的修剪（保留极端值）
upper_bound = np.percentile(test_predictions_calibrated, 99)
lower_bound = np.percentile(test_predictions_calibrated, 1)
test_predictions_final = np.clip(test_predictions_calibrated, lower_bound, upper_bound * 1.2)

# 14. 生成提交文件
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'tested_positive': test_predictions_final
})

submission_df.to_csv('submission.csv', index=False)
print(f"\nSubmission file saved to 'submission.csv'.")

# 15. 最终验证集分析
print("\n" + "="*60)
print("Validation set analysis:")
print("="*60)
print(f"True values range: [{val_true.min():.2f}, {val_true.max():.2f}]")
print(f"Predicted values range: [{val_predictions.min():.2f}, {val_predictions.max():.2f}]")

# 计算R²分数
from sklearn.metrics import r2_score
r2 = r2_score(val_true, val_predictions)
print(f"R² Score: {r2:.4f}")

# 计算平均绝对百分比误差
mape = np.mean(np.abs(val_predictions - val_true) / (val_true + 1)) * 100
print(f"MAPE: {mape:.2f}%")

# 预测统计
print(f"\nTest predictions range: [{test_predictions_final.min():.2f}, {test_predictions_final.max():.2f}]")
print(f"Mean prediction: {test_predictions_final.mean():.2f}, Std: {test_predictions_final.std():.2f}")
print(f"Median prediction: {np.median(test_predictions_final):.2f}")

print("\n" + "="*60)
print("OPTIMIZATION COMPLETE!")
print("="*60)