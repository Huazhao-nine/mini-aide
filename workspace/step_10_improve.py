import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit
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

print(f"训练集形状: {train_df.shape}, 测试集形状: {test_df.shape}")

# 2. 改进的特征工程 - 更精确的领域知识应用
def create_advanced_features(df, is_train=True):
    """创建基于流行病学知识的特征"""
    df = df.copy()
    
    # 保存id和目标列
    if is_train:
        target_col = 'tested_positive_day3'
        target = df[target_col].copy()
        df = df.drop([target_col], axis=1)
    
    id_col = df['id'] if 'id' in df.columns else None
    df = df.drop(['id'], axis=1)
    
    # 获取州名列表
    state_columns = [col for col in df.columns if col in ['AL', 'AZ', 'CA', 'CO', 'CT', 'FL', 'GA', 'IL', 'IN', 'IA', 'KS', 
                    'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MO', 'NJ', 'NM', 'NY', 'NC', 'OH', 'OK', 'OR', 'PA', 'SC', 
                    'TN', 'TX', 'VA', 'WA', 'WV', 'WI']]
    
    # 1. 核心流行病学特征
    for day in [1, 2, 3]:
        # 暴露风险指数 = 症状指标 + 行为风险
        if all([f'cli_day{day}' in df.columns, f'wrestaurant_indoors_day{day}' in df.columns]):
            df[f'exposure_risk_day{day}'] = (
                df[f'cli_day{day}'] * 0.6 + 
                df[f'wrestaurant_indoors_day{day}'] * 0.2 +
                df[f'wshop_indoors_day{day}'] * 0.1 +
                df[f'public_transit_day{day}'] * 0.1
            )
        
        # 防护效能指数 = 口罩使用 + 口罩信念 + 社交距离
        if f'wearing_mask_7d_day{day}' in df.columns:
            df[f'protection_efficacy_day{day}'] = (
                df[f'wearing_mask_7d_day{day}'] * 0.4 +
                df[f'wbelief_masking_effective_day{day}'] * 0.3 +
                df[f'wothers_masked_public_day{day}'] * 0.2 +
                df[f'wothers_distanced_public_day{day}'] * 0.1
            )
        
        # 传染性压力指数 = 症状 + 社区传播
        if all([f'cli_day{day}' in df.columns, f'hh_cmnty_cli_day{day}' in df.columns]):
            df[f'contagion_pressure_day{day}'] = (
                df[f'cli_day{day}'] * 0.4 +
                df[f'ili_day{day}'] * 0.2 +
                df[f'hh_cmnty_cli_day{day}'] * 0.2 +
                df[f'nohh_cmnty_cli_day{day}'] * 0.2
            )
    
    # 2. 时间序列动力学特征
    for base_feat in ['cli', 'ili', 'tested_positive', 'wearing_mask_7d', 'wworried_catch_covid']:
        day_cols = [f'{base_feat}_day1', f'{base_feat}_day2', f'{base_feat}_day3']
        existing_cols = [col for col in day_cols if col in df.columns]
        
        if len(existing_cols) >= 2:
            # 简单差分
            if len(existing_cols) == 3:
                df[f'{base_feat}_diff_3_1'] = df[existing_cols[2]] - df[existing_cols[0]]
                df[f'{base_feat}_diff_3_2'] = df[existing_cols[2]] - df[existing_cols[1]]
                df[f'{base_feat}_diff_2_1'] = df[existing_cols[1]] - df[existing_cols[0]]
            
            # 增长率
            if len(existing_cols) >= 2:
                df[f'{base_feat}_growth_rate'] = (df[existing_cols[-1]] - df[existing_cols[-2]]) / (abs(df[existing_cols[-2]]) + 1)
    
    # 3. 交互特征 - 基于流行病学理论
    for day in [1, 2, 3]:
        # 风险行为与症状的交互
        if all([f'cli_day{day}' in df.columns, f'wrestaurant_indoors_day{day}' in df.columns]):
            df[f'risk_behavior_symptom_day{day}'] = df[f'cli_day{day}'] * df[f'wrestaurant_indoors_day{day}'] / 100
        
        # 疫苗接种与担忧的交互
        if all([f'wcovid_vaccinated_friends_day{day}' in df.columns, f'wworried_catch_covid_day{day}' in df.columns]):
            df[f'vax_worry_interaction_day{day}'] = df[f'wcovid_vaccinated_friends_day{day}'] * (100 - df[f'wworried_catch_covid_day{day}']) / 100
    
    # 4. 滞后特征的自相关
    if all([f'tested_positive_day1' in df.columns, f'tested_positive_day2' in df.columns]):
        df['positivity_autocorr'] = df['tested_positive_day1'] * 0.3 + df['tested_positive_day2'] * 0.7
    
    # 5. 状态特征 - 复合指标
    for day in [1, 2, 3]:
        if all([f'exposure_risk_day{day}' in df.columns, f'protection_efficacy_day{day}' in df.columns]):
            # 标准化到相似范围
            exposure_norm = (df[f'exposure_risk_day{day}'] - df[f'exposure_risk_day{day}'].min()) / (df[f'exposure_risk_day{day}'].max() - df[f'exposure_risk_day{day}'].min() + 1e-8)
            protection_norm = (df[f'protection_efficacy_day{day}'] - df[f'protection_efficacy_day{day}'].min()) / (df[f'protection_efficacy_day{day}'].max() - df[f'protection_efficacy_day{day}'].min() + 1e-8)
            
            df[f'net_risk_day{day}'] = exposure_norm * (1 - protection_norm)
    
    # 6. 添加地理特征（州作为分类变量）
    if state_columns:
        df['state_sum'] = df[state_columns].sum(axis=1)
    
    # 添加id列回数据
    if id_col is not None:
        df['id'] = id_col
    
    # 如果是训练集，添加目标列
    if is_train:
        df['tested_positive_day3'] = target
    
    return df

# 应用特征工程
print("创建高级特征...")
train_df_enhanced = create_advanced_features(train_df, is_train=True)
test_df_enhanced = create_advanced_features(test_df, is_train=False)

print(f"原始特征数: {train_df.shape[1] - 2}")
print(f"增强后特征数: {train_df_enhanced.shape[1] - 2}")

# 3. 分离特征和目标
target_col = 'tested_positive_day3'
X = train_df_enhanced.drop(['id', target_col], axis=1)
y = train_df_enhanced[target_col].values

# 获取测试集特征
X_test = test_df_enhanced.drop(['id'], axis=1)

print(f"\n特征形状: X={X.shape}, y={y.shape}, X_test={X_test.shape}")

# 4. 数据划分 - 使用时间序列分割
train_size = int(len(X) * 0.8)
X_train = X.iloc[:train_size]
y_train = y[:train_size]
X_val = X.iloc[train_size:]
y_val = y[train_size:]

print(f"训练集: {len(X_train)} 样本")
print(f"验证集: {len(X_val)} 样本")

# 5. 特征缩放 - 使用RobustScaler减少异常值影响
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 6. 改进的特征选择 - 使用互信息
print("\n执行特征选择...")
selector = SelectKBest(score_func=mutual_info_regression, k=min(40, X_train_scaled.shape[1]))
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_val_selected = selector.transform(X_val_scaled)
X_test_selected = selector.transform(X_test_scaled)

selected_features = X.columns[selector.get_support()]
print(f"选择了 {X_train_selected.shape[1]} 个最佳特征")

# 7. 定义改进的神经网络模型
class AdvancedCOVIDPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32, 16], dropout_rates=[0.3, 0.25, 0.2, 0.15]):
        super(AdvancedCOVIDPredictor, self).__init__()
        
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
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.model(x)

# 8. 训练配置
class Trainer:
    def __init__(self, model, device, criterion, optimizer, scheduler):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * batch_X.size(0)
        
        return total_loss / len(train_loader.dataset)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item() * batch_X.size(0)
                
                all_preds.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(batch_y.cpu().numpy().flatten())
        
        return total_loss / len(val_loader.dataset), np.array(all_preds), np.array(all_targets)

# 9. 训练模型
def train_model(X_train, y_train, X_val, y_val, X_test, fold_idx=1):
    """训练单个模型"""
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    
    # 创建DataLoader
    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdvancedCOVIDPredictor(input_dim=X_train.shape[1]).to(device)
    
    # 损失函数 - 使用SmoothL1Loss (Huber Loss)
    criterion = nn.SmoothL1Loss(beta=1.0)
    
    # 优化器 - 使用AdamW
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 学习率调度器 - 使用余弦退火
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    
    trainer = Trainer(model, device, criterion, optimizer, scheduler)
    
    # 训练循环
    epochs = 300
    best_val_loss = float('inf')
    patience = 40
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # 训练阶段
        train_loss = trainer.train_epoch(train_loader)
        
        # 验证阶段
        val_loss, val_preds, val_targets = trainer.validate(val_loader)
        
        # 计算RMSE
        val_rmse = np.sqrt(np.mean((val_preds - val_targets) ** 2))
        
        train_losses.append(train_loss)
        val_losses.append(val_rmse)
        
        # 更新学习率
        scheduler.step()
        
        # 早停检查
        if val_rmse < best_val_loss:
            best_val_loss = val_rmse
            patience_counter = 0
            torch.save(model.state_dict(), f'best_model_fold{fold_idx}.pth')
            best_val_preds = val_preds
        else:
            patience_counter += 1
        
        if epoch % 30 == 0:
            print(f'  Fold {fold_idx} - Epoch {epoch}: Train Loss: {train_loss:.4f}, Val RMSE: {val_rmse:.4f}')
        
        if patience_counter >= patience:
            print(f'  Fold {fold_idx} - Early stopping at epoch {epoch}')
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load(f'best_model_fold{fold_idx}.pth'))
    
    # 在验证集上做最终预测
    trainer.model.eval()
    with torch.no_grad():
        val_preds = trainer.model(X_val_tensor.to(device)).cpu().numpy().flatten()
        test_preds = trainer.model(X_test_tensor.to(device)).cpu().numpy().flatten()
    
    return val_preds, test_preds, best_val_loss

# 10. 模型集成 - 使用不同特征子集
print("\n训练集成模型...")

# 创建不同特征子集
feature_sets = []

# 第一组：全部选择的特征
feature_sets.append((X_train_selected, X_val_selected, X_test_selected, "Full"))

# 第二组：只选择症状相关特征
symptom_features = [col for col in selected_features if any(x in col for x in ['cli', 'ili', 'hh_cmnty', 'nohh_cmnty'])]
if len(symptom_features) > 0:
    symptom_idx = [i for i, col in enumerate(X.columns) if col in symptom_features]
    X_train_symptoms = X_train_scaled[:, symptom_idx]
    X_val_symptoms = X_val_scaled[:, symptom_idx]
    X_test_symptoms = X_test_scaled[:, symptom_idx]
    feature_sets.append((X_train_symptoms, X_val_symptoms, X_test_symptoms, "Symptoms"))

# 第三组：只选择行为相关特征
behavior_features = [col for col in selected_features if any(x in col for x in ['mask', 'restaurant', 'shop', 'transit', 'event'])]
if len(behavior_features) > 0:
    behavior_idx = [i for i, col in enumerate(X.columns) if col in behavior_features]
    X_train_behavior = X_train_scaled[:, behavior_idx]
    X_val_behavior = X_val_scaled[:, behavior_idx]
    X_test_behavior = X_test_scaled[:, behavior_idx]
    feature_sets.append((X_train_behavior, X_val_behavior, X_test_behavior, "Behavior"))

print(f"创建了 {len(feature_sets)} 个特征子集")

# 训练多个模型
all_val_preds = []
all_test_preds = []
val_scores = []

for i, (X_train_set, X_val_set, X_test_set, name) in enumerate(feature_sets):
    print(f"\n训练模型 {i+1}/{len(feature_sets)}: {name} 特征集")
    val_preds, test_preds, best_score = train_model(
        X_train_set, y_train, 
        X_val_set, y_val,
        X_test_set, fold_idx=i+1
    )
    
    all_val_preds.append(val_preds)
    all_test_preds.append(test_preds)
    val_scores.append(best_score)
    
    print(f"  模型 {name}: Best Val RMSE = {best_score:.4f}")

# 11. 模型融合
print("\n模型融合...")

# 使用验证集性能计算权重
weights = 1.0 / np.array(val_scores)
weights = weights / weights.sum()

print("模型权重:")
for i, (name, w) in enumerate(zip([fs[3] for fs in feature_sets], weights)):
    print(f"  模型 {name}: {w:.3f}")

# 融合验证集预测
ensemble_val_preds = np.zeros_like(all_val_preds[0])
for i, preds in enumerate(all_val_preds):
    ensemble_val_preds += weights[i] * preds

# 计算融合后的验证分数
val_rmse = np.sqrt(np.mean((ensemble_val_preds - y_val) ** 2))
val_mae = np.mean(np.abs(ensemble_val_preds - y_val))
score = 1.0 / (1.0 + val_rmse)

print(f"\n验证集结果:")
print(f"  RMSE: {val_rmse:.4f}")
print(f"  MAE: {val_mae:.4f}")
print(f"  Score: {score:.4f}")

# 12. 融合测试集预测
ensemble_test_preds = np.zeros_like(all_test_preds[0])
for i, preds in enumerate(all_test_preds):
    ensemble_test_preds += weights[i] * preds

# 13. 后处理优化
print("\n后处理优化...")

# 基于验证集分布进行校准
val_mean = y_val.mean()
val_std = y_val.std()
pred_mean = ensemble_test_preds.mean()
pred_std = ensemble_test_preds.std()

# 校准到验证集分布
scale_factor = val_std / (pred_std + 1e-8)
ensemble_test_preds = (ensemble_test_preds - pred_mean) * scale_factor + val_mean

# 确保非负
ensemble_test_preds = np.maximum(ensemble_test_preds, 0)

# 温和的缩尾处理
q01 = np.percentile(y_val, 0.5)
q99 = np.percentile(y_val, 99.5)
ensemble_test_preds = np.clip(ensemble_test_preds, q01, q99)

print(f"  校准后 - 均值: {ensemble_test_preds.mean():.2f}, 标准差: {ensemble_test_preds.std():.2f}")
print(f"  验证集 - 均值: {val_mean:.2f}, 标准差: {val_std:.2f}")

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
print(f"  验证集 RMSE: {val_rmse:.4f}")
print(f"  验证集 Score: {score:.4f}")
print(f"  测试集预测均值: {ensemble_test_preds.mean():.2f}")

print(f"\nScore= (1.0 / (1.0 + RMSE)) = {score:.4f}")