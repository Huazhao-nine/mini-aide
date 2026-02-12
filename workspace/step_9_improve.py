import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
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

# 2. 分离特征和目标
target_col = 'tested_positive_day3'
y_train = train_df[target_col].values.reshape(-1, 1)

# 提取州特征（用于固定效应）
state_cols = [col for col in train_df.columns if len(col) == 2 and col.isupper()]

# 3. 高级特征工程
def create_advanced_features(df, is_train=True):
    """创建基于流行病学知识的特征"""
    new_df = df.copy()
    
    # 基础特征分组
    symptom_cols = ['cli', 'ili', 'hh_cmnty_cli', 'nohh_cmnty_cli', 'wnohh_cmnty_cli']
    behavior_cols = ['wearing_mask_7d', 'wshop_indoors', 'wrestaurant_indoors', 
                    'public_transit', 'wlarge_event_indoors']
    belief_cols = ['wbelief_masking_effective', 'wbelief_distancing_effective']
    mental_cols = ['wworried_catch_covid', 'worried_finances']
    other_cols = ['wothers_masked_public', 'wothers_distanced_public', 
                  'wcovid_vaccinated_friends']
    
    all_features = symptom_cols + behavior_cols + belief_cols + mental_cols + other_cols
    
    # 1. 时序特征（最重要）
    for feature in all_features:
        # 确保列存在
        day_cols = []
        for day in [1, 2, 3]:
            col_name = f'{feature}_day{day}'
            if col_name in df.columns:
                day_cols.append(col_name)
        
        if len(day_cols) >= 2:
            # 创建趋势特征
            if f'{feature}_day1' in df.columns and f'{feature}_day2' in df.columns:
                new_df[f'{feature}_trend_1_2'] = df[f'{feature}_day2'] - df[f'{feature}_day1']
            
            # 创建加速度特征
            if f'{feature}_day1' in df.columns and f'{feature}_day2' in df.columns and f'{feature}_day3' in df.columns:
                new_df[f'{feature}_acceleration'] = (df[f'{feature}_day3'] - 2*df[f'{feature}_day2'] + df[f'{feature}_day1'])
            
            # 创建移动统计特征
            new_df[f'{feature}_mean_3d'] = df[day_cols].mean(axis=1)
            new_df[f'{feature}_std_3d'] = df[day_cols].std(axis=1)
            new_df[f'{feature}_max_3d'] = df[day_cols].max(axis=1)
            new_df[f'{feature}_min_3d'] = df[day_cols].min(axis=1)
            new_df[f'{feature}_range_3d'] = new_df[f'{feature}_max_3d'] - new_df[f'{feature}_min_3d']
            
            # 创建百分比变化特征
            if f'{feature}_day1' in df.columns:
                for day in [2, 3]:
                    col_day = f'{feature}_day{day}'
                    if col_day in df.columns:
                        new_df[f'{feature}_pct_change_{day}'] = (df[col_day] - df[f'{feature}_day1']) / (df[f'{feature}_day1'] + 1e-10)
    
    # 2. 流行病学特征（基于领域知识）
    for day in [1, 2]:
        # 风险暴露指数 = 室内活动 + 社区感染 - 防护措施
        indoor_risk = 0
        protection = 0
        
        for indoor in ['wrestaurant_indoors', 'wshop_indoors', 'wlarge_event_indoors', 'public_transit']:
            col = f'{indoor}_day{day}'
            if col in df.columns:
                indoor_risk += df[col]
        
        for protect in ['wearing_mask_7d', 'wothers_masked_public', 'wcovid_vaccinated_friends']:
            col = f'{protect}_day{day}'
            if col in df.columns:
                protection += df[col]
        
        if 'nohh_cmnty_cli_day{day}' in df.columns:
            infection_pressure = df[f'nohh_cmnty_cli_day{day}']
            new_df[f'risk_exposure_index_day{day}'] = indoor_risk + infection_pressure - protection/3
    
    # 3. 关键交互特征（基于因果关系）
    for day in [1, 2]:
        # 症状传播潜力 = 症状率 × 室内活动
        if f'cli_day{day}' in df.columns and f'wrestaurant_indoors_day{day}' in df.columns:
            new_df[f'spread_potential_{day}'] = df[f'cli_day{day}'] * df[f'wrestaurant_indoors_day{day}']
        
        # 防护有效性 = 口罩佩戴 × 他人戴口罩
        if f'wearing_mask_7d_day{day}' in df.columns and f'wothers_masked_public_day{day}' in df.columns:
            new_df[f'protection_effectiveness_{day}'] = df[f'wearing_mask_7d_day{day}'] * df[f'wothers_masked_public_day{day}']
        
        # 担忧行动指数 = 担心感染 × 防护行为
        if f'wworried_catch_covid_day{day}' in df.columns:
            if f'wearing_mask_7d_day{day}' in df.columns:
                new_df[f'worry_action_index_{day}'] = df[f'wworried_catch_covid_day{day}'] * df[f'wearing_mask_7d_day{day}']
    
    # 4. 目标变量的滞后特征和趋势
    if is_train:
        if 'tested_positive_day1' in df.columns and 'tested_positive_day2' in df.columns:
            new_df['tested_positive_trend'] = df['tested_positive_day2'] - df['tested_positive_day1']
            new_df['tested_positive_momentum'] = (df['tested_positive_day2'] - df['tested_positive_day1']) / (df['tested_positive_day1'] + 1e-10)
    
    # 5. 州固定效应特征（与目标变量的交互）
    # 这里我们创建州与关键特征的交互
    key_features_for_state_interaction = ['cli_day2', 'nohh_cmnty_cli_day2', 'wearing_mask_7d_day2']
    
    # 6. 添加多项式特征（二次项）
    for feature in ['cli_day2', 'nohh_cmnty_cli_day2', 'wearing_mask_7d_day2', 'wworried_catch_covid_day2']:
        if feature in df.columns:
            new_df[f'{feature}_squared'] = df[feature] ** 2
    
    return new_df

print("创建高级特征...")
X_train = create_advanced_features(train_df, is_train=True)
X_test = create_advanced_features(test_df, is_train=False)

# 删除ID列和目标列（保留州特征）
X_train = X_train.drop(['id', target_col], axis=1)
X_test = X_test.drop(['id'], axis=1)

# 确保测试集有训练集的所有列
for col in X_train.columns:
    if col not in X_test.columns:
        X_test[col] = 0

# 重新排列列顺序，确保一致
common_cols = [col for col in X_train.columns if col in X_test.columns]
X_train = X_train[common_cols]
X_test = X_test[common_cols]

print(f"特征工程后特征数量: {X_train.shape[1]}")

# 4. 时间序列验证集划分（最后20%，不shuffle）
split_idx = int(len(X_train) * 0.8)
X_val = X_train.iloc[split_idx:].copy()
y_val = y_train[split_idx:].copy()
X_train_split = X_train.iloc[:split_idx].copy()
y_train_split = y_train[:split_idx].copy()

print(f"训练集大小: {len(X_train_split)}, 验证集大小: {len(X_val)}")

# 5. 特征标准化（使用RobustScaler，对异常值更鲁棒）
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_split)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 6. 特征降维（使用PCA保留95%方差，而不是SelectKBest）
print("进行PCA降维...")
pca = PCA(n_components=0.95)  # 保留95%的方差
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"PCA后维度: {X_train_pca.shape[1]} (保留方差: {sum(pca.explained_variance_ratio_):.4f})")

# 7. 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train_pca)
y_train_tensor = torch.FloatTensor(y_train_split)
X_val_tensor = torch.FloatTensor(X_val_pca)
y_val_tensor = torch.FloatTensor(y_val)
X_test_tensor = torch.FloatTensor(X_test_pca)

# 创建DataLoader（关键：时间序列数据不shuffle）
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 8. 定义改进的神经网络模型（残差网络 + 深度）
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        
        # 如果输入输出维度不同，使用1x1卷积
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class COVIDPredictorV2(nn.Module):
    def __init__(self, input_dim):
        super(COVIDPredictorV2, self).__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 残差块
        self.res_block1 = ResidualBlock(256, 128, dropout_rate=0.25)
        self.res_block2 = ResidualBlock(128, 64, dropout_rate=0.2)
        self.res_block3 = ResidualBlock(64, 32, dropout_rate=0.15)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1)
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
        x = self.input_layer(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.output_layer(x)
        return x

# 9. 训练设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

model = COVIDPredictorV2(input_dim=X_train_pca.shape[1]).to(device)

# 使用Huber损失（对异常值更鲁棒）
criterion = nn.HuberLoss(delta=1.0)

# 使用AdamW优化器（带权重衰减）
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# 学习率调度器（余弦退火）
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=50, T_mult=1, eta_min=1e-6
)

# 梯度裁剪
grad_clip = 1.0

# 10. 训练循环
epochs = 300
best_val_loss = float('inf')
patience = 30
patience_counter = 0

print("开始训练...")
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        train_loss += loss.item() * batch_X.size(0)
    
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)
    
    # 验证阶段
    model.eval()
    val_loss = 0
    val_preds = []
    val_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_X.size(0)
            val_preds.extend(outputs.cpu().numpy())
            val_targets.extend(batch_y.cpu().numpy())
    
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)
    
    # 计算验证集RMSE
    val_preds = np.array(val_preds).flatten()
    val_targets = np.array(val_targets).flatten()
    val_rmse = np.sqrt(np.mean((val_preds - val_targets) ** 2))
    
    # 学习率调整
    scheduler.step()
    
    # 早停检查
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
        best_val_rmse = val_rmse
        best_epoch = epoch
    else:
        patience_counter += 1
    
    if epoch % 20 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}, LR: {current_lr:.6f}')
    
    if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch}')
        break

print(f'最佳模型在 epoch {best_epoch}, Val Loss: {best_val_loss:.4f}, Val RMSE: {best_val_rmse:.4f}')

# 11. 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))

# 12. 在验证集上最终评估
model.eval()
val_predictions = []
val_targets = []

with torch.no_grad():
    for batch_X, batch_y in val_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        val_predictions.extend(outputs.cpu().numpy())
        val_targets.extend(batch_y.cpu().numpy())

val_predictions = np.array(val_predictions).flatten()
val_targets = np.array(val_targets).flatten()

# 计算RMSE和MAE
rmse = np.sqrt(np.mean((val_predictions - val_targets) ** 2))
mae = np.mean(np.abs(val_predictions - val_targets))
r2 = 1 - np.sum((val_predictions - val_targets) ** 2) / np.sum((val_targets - np.mean(val_targets)) ** 2)

print(f'验证集RMSE: {rmse:.4f}')
print(f'验证集MAE: {mae:.4f}')
print(f'验证集R²: {r2:.4f}')

# 13. 模型集成（创建多个预测）
def create_predictions(model, X_tensor, device, n_iter=5):
    """创建多个预测（模拟集成）"""
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for _ in range(n_iter):
            # 启用dropout进行测试时的不确定性估计
            model.train()
            # 小技巧：对输入添加微小噪声增加鲁棒性
            noise = torch.randn_like(X_tensor) * 0.01
            X_noisy = X_tensor + noise
            preds = model(X_noisy.to(device)).cpu().numpy().flatten()
            all_preds.append(preds)
    
    # 返回平均值
    return np.mean(all_preds, axis=0)

# 14. 在测试集上进行预测（使用集成）
print("生成测试集预测...")
test_predictions = create_predictions(model, X_test_tensor, device, n_iter=5)

# 15. 基于流行病学知识的后处理
def epidemiological_postprocessing(predictions, test_df):
    """基于流行病学知识的后处理"""
    processed = predictions.copy()
    
    # 1. 确保非负
    processed = np.maximum(processed, 0)
    
    # 2. 基于州的后处理（某些州可能有基线差异）
    # 这里我们使用州特征的加权平均来调整预测
    state_cols = [col for col in test_df.columns if len(col) == 2 and col.isupper()]
    if len(state_cols) > 0:
        state_weights = {
            'CA': 0.95, 'NY': 0.97, 'TX': 1.05, 'FL': 1.03,
            # 其他州保持1.0
        }
        
        for i, row in test_df.iterrows():
            for state, weight in state_weights.items():
                if state in state_cols and row[state] == 1:
                    processed[i] *= weight
    
    # 3. 基于症状严重程度的调整
    if 'cli_day2' in test_df.columns:
        cli_mean = test_df['cli_day2'].mean()
        cli_std = test_df['cli_day2'].std()
        cli_z_scores = (test_df['cli_day2'] - cli_mean) / cli_std
        # 症状严重的地区增加预测
        processed = processed * (1 + 0.05 * np.clip(cli_z_scores, -1, 1))
    
    # 4. 基于防护措施的调整
    if 'wearing_mask_7d_day2' in test_df.columns:
        mask_mean = test_df['wearing_mask_7d_day2'].mean()
        mask_effect = (test_df['wearing_mask_7d_day2'] - mask_mean) / 100
        # 口罩佩戴率高的地区减少预测
        processed = processed * (1 - 0.1 * np.clip(mask_effect, 0, 0.3))
    
    # 5. 平滑处理（避免极端值）
    processed = np.clip(processed, 0, 50)  # 合理范围
    
    return processed

test_predictions_processed = epidemiological_postprocessing(test_predictions, test_df)

# 16. 生成提交文件
submission = pd.DataFrame({
    'id': test_df['id'],
    'tested_positive': test_predictions_processed
})

submission.to_csv('submission.csv', index=False)
print(f"提交文件已保存，包含 {len(submission)} 条预测结果")

# 17. 打印最终分数
score = 1.0 / (1.0 + rmse)
print(f"Score= (1.0 / (1.0 + RMSE)) = {score:.4f}")

# 18. 额外：特征重要性分析（使用梯度信息）
print("\n=== 特征重要性分析 ===")
model.eval()
X_sample = X_train_tensor[:100].to(device)
X_sample.requires_grad = True

output = model(X_sample)
grads = torch.autograd.grad(output.sum(), X_sample)[0]
feature_importance = grads.abs().mean(dim=0).cpu().numpy()

# 获取最重要的特征
if hasattr(pca, 'components_'):
    # 将重要性映射回原始特征
    pca_components = pca.components_
    original_feature_importance = np.abs(pca_components.T @ feature_importance)
    
    # 获取前10个最重要的原始特征
    top_indices = np.argsort(original_feature_importance)[-10:][::-1]
    top_features = X_train.columns[top_indices]
    
    print("Top 10最重要特征:")
    for i, (idx, feat) in enumerate(zip(top_indices, top_features)):
        importance = original_feature_importance[idx]
        print(f"  {i+1}. {feat}: {importance:.4f}")