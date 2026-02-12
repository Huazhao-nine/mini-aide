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

# 2. 提取州特征和创建核心特征
target_col = 'tested_positive_day3'
y_train = train_df[target_col].values

# 州特征
state_cols = [col for col in train_df.columns if len(col) == 2 and col.isupper()]

# 3. 核心特征工程 - 简化但有效
def create_simple_features(df, is_train=True):
    new_df = df.copy()
    
    # 基础特征
    base_features = [
        'cli', 'ili', 'hh_cmnty_cli', 'nohh_cmnty_cli', 'wnohh_cmnty_cli',
        'wearing_mask_7d', 'wshop_indoors', 'wrestaurant_indoors',
        'public_transit', 'wlarge_event_indoors', 'wbelief_masking_effective',
        'wbelief_distancing_effective', 'wworried_catch_covid',
        'worried_finances', 'wothers_masked_public', 'wothers_distanced_public',
        'wcovid_vaccinated_friends'
    ]
    
    # 1. 关键的时序特征
    for feat in base_features:
        for day in [1, 2, 3]:
            col = f'{feat}_day{day}'
            if col in df.columns:
                new_df[col] = df[col]
        
        # 创建3天平均值
        day_cols = [f'{feat}_day{i}' for i in [1, 2, 3] if f'{feat}_day{i}' in df.columns]
        if len(day_cols) >= 2:
            new_df[f'{feat}_mean'] = df[day_cols].mean(axis=1)
            new_df[f'{feat}_std'] = df[day_cols].std(axis=1)
    
    # 2. 关键交互特征（基于流行病学）
    for day in [2, 3]:  # 使用第2、3天数据更稳定
        # 风险暴露 = 室内活动 × 社区感染
        if f'wrestaurant_indoors_day{day}' in df.columns and f'nohh_cmnty_cli_day{day}' in df.columns:
            new_df[f'risk_exposure_day{day}'] = df[f'wrestaurant_indoors_day{day}'] * df[f'nohh_cmnty_cli_day{day}'] / 100
        
        # 防护有效性 = 口罩佩戴 × 他人戴口罩
        if f'wearing_mask_7d_day{day}' in df.columns and f'wothers_masked_public_day{day}' in df.columns:
            new_df[f'protection_day{day}'] = df[f'wearing_mask_7d_day{day}'] * df[f'wothers_masked_public_day{day}'] / 100
    
    # 3. 目标变量的滞后特征（仅在训练时）
    if is_train:
        for lag in [1, 2]:
            if f'tested_positive_day{lag}' in df.columns:
                new_df[f'target_lag{lag}'] = df[f'tested_positive_day{lag}']
    
    return new_df

print("创建特征...")
X_train = create_simple_features(train_df, is_train=True)
X_test = create_simple_features(test_df, is_train=False)

# 删除ID和目标列
X_train = X_train.drop(['id', target_col], axis=1)
X_test_id = test_df['id'].copy()
X_test = X_test.drop(['id'], axis=1)

# 确保特征一致性
for col in X_train.columns:
    if col not in X_test.columns:
        X_test[col] = 0
X_test = X_test[X_train.columns]

print(f"特征数量: {X_train.shape[1]}")

# 4. 时间序列验证集划分（最后20%）
split_idx = int(len(X_train) * 0.8)
X_val = X_train.iloc[split_idx:].copy()
y_val = y_train[split_idx:].copy()
X_train_split = X_train.iloc[:split_idx].copy()
y_train_split = y_train[:split_idx].copy()

print(f"训练集: {len(X_train_split)}, 验证集: {len(X_val)}")

# 5. 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_split)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 6. 特征选择（SelectKBest - 任务要求）
print("特征选择...")
selector = SelectKBest(score_func=f_regression, k=min(20, X_train_scaled.shape[1]))
X_train_selected = selector.fit_transform(X_train_scaled, y_train_split)
X_val_selected = selector.transform(X_val_scaled)
X_test_selected = selector.transform(X_test_scaled)

selected_indices = selector.get_support(indices=True)
selected_features = X_train.columns[selected_indices]
print(f"选择了 {len(selected_features)} 个特征")

# 7. 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train_selected)
y_train_tensor = torch.FloatTensor(y_train_split).reshape(-1, 1)
X_val_tensor = torch.FloatTensor(X_val_selected)
y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_selected)

# DataLoader - 关键：时间序列不shuffle
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 8. 定义改进的神经网络（更简单但有效）
class COVIDPredictor(nn.Module):
    def __init__(self, input_dim):
        super(COVIDPredictor, self).__init__()
        
        self.net = nn.Sequential(
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
            
            nn.Linear(32, 1)
        )
        
        # 初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.net(x)

# 9. 模型集成（创建多个模型）
class ModelEnsemble:
    def __init__(self, input_dim, n_models=3):
        self.models = []
        self.input_dim = input_dim
        self.n_models = n_models
        
    def train_models(self, train_loader, val_loader, epochs=200):
        for i in range(self.n_models):
            print(f"\n训练模型 {i+1}/{self.n_models}")
            model = COVIDPredictor(self.input_dim)
            
            # 每个模型使用不同的随机种子
            seed = SEED + i * 100
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # 训练单个模型
            model = self._train_single_model(model, train_loader, val_loader, epochs)
            self.models.append(model)
    
    def _train_single_model(self, model, train_loader, val_loader, epochs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # 使用L1Loss（MAE） - 对异常值更鲁棒
        criterion = nn.L1Loss()
        
        # 使用AdamW优化器
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        best_val_loss = float('inf')
        patience = 25
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
            
            train_loss /= len(train_loader.dataset)
            
            # 验证
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
            
            # 计算RMSE
            val_preds = np.array(val_preds).flatten()
            val_targets = np.array(val_targets).flatten()
            val_rmse = np.sqrt(np.mean((val_preds - val_targets) ** 2))
            
            # 学习率调整
            scheduler.step(val_loss)
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                best_val_rmse = val_rmse
            else:
                patience_counter += 1
            
            if epoch % 25 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch:3d}: Train MAE: {train_loss:.4f}, Val MAE: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}, LR: {current_lr:.6f}')
            
            if patience_counter >= patience:
                print(f'早停于 epoch {epoch}')
                break
        
        # 加载最佳模型
        model.load_state_dict(best_model_state)
        print(f'最佳模型验证RMSE: {best_val_rmse:.4f}')
        
        return model
    
    def predict(self, X_tensor):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        all_preds = []
        
        for model in self.models:
            model.eval()
            model.to(device)
            
            with torch.no_grad():
                # 多次预测取平均（使用dropout）
                preds_list = []
                for _ in range(5):
                    preds = model(X_tensor.to(device)).cpu().numpy().flatten()
                    preds_list.append(preds)
                
                preds_avg = np.mean(preds_list, axis=0)
                all_preds.append(preds_avg)
        
        # 集成预测（取中位数减少异常值影响）
        ensemble_preds = np.median(all_preds, axis=0)
        
        return ensemble_preds

# 10. 训练模型集成
print("\n开始训练模型集成...")
input_dim = X_train_selected.shape[1]
ensemble = ModelEnsemble(input_dim=input_dim, n_models=3)
ensemble.train_models(train_loader, val_loader, epochs=200)

# 11. 在验证集上评估
print("\n在验证集上评估...")
val_predictions = ensemble.predict(X_val_tensor)
val_targets = y_val_tensor.numpy().flatten()

# 计算指标
rmse = np.sqrt(np.mean((val_predictions - val_targets) ** 2))
mae = np.mean(np.abs(val_predictions - val_targets))
score = 1.0 / (1.0 + rmse)

print(f"验证集RMSE: {rmse:.4f}")
print(f"验证集MAE: {mae:.4f}")
print(f"Score= (1.0 / (1.0 + RMSE)) = {score:.4f}")

# 12. 在测试集上预测
print("\n生成测试集预测...")
test_predictions = ensemble.predict(X_test_tensor)

# 13. 稳健的后处理
def robust_postprocessing(predictions, X_test_df):
    """稳健的后处理，仅处理明显不合理的情况"""
    processed = predictions.copy()
    
    # 1. 确保非负
    processed = np.maximum(processed, 0)
    
    # 2. 基于测试集特征的温和调整
    # 如果有症状数据，轻微调整
    if 'cli_day2_mean' in X_test_df.columns:
        cli_mean = X_test_df['cli_day2_mean'].mean()
        # 症状高于平均值的地区轻微增加预测
        mask = X_test_df['cli_day2_mean'] > cli_mean
        processed[mask] *= 1.02
    
    # 3. 基于防护措施的轻微调整
    if 'wearing_mask_7d_mean' in X_test_df.columns:
        mask_mean = X_test_df['wearing_mask_7d_mean'].mean()
        # 口罩佩戴率高的地区轻微减少预测
        mask = X_test_df['wearing_mask_7d_mean'] > mask_mean
        processed[mask] *= 0.98
    
    # 4. 温和的平滑（使用移动平均）
    window_size = 5
    if len(processed) > window_size * 2:
        smoothed = np.copy(processed)
        for i in range(len(processed)):
            start = max(0, i - window_size)
            end = min(len(processed), i + window_size + 1)
            smoothed[i] = np.mean(processed[start:end])
        processed = smoothed
    
    # 5. 设置合理上限（基于训练数据分布）
    train_target_max = np.percentile(y_train, 99)  # 99%分位数
    processed = np.minimum(processed, train_target_max * 1.5)
    
    return processed

# 准备测试集特征DataFrame用于后处理
X_test_df = pd.DataFrame(X_test_scaled, columns=X_train.columns)
test_predictions_processed = robust_postprocessing(test_predictions, X_test_df)

# 14. 生成提交文件
submission = pd.DataFrame({
    'id': X_test_id,
    'tested_positive': test_predictions_processed
})

submission.to_csv('submission.csv', index=False)
print(f"\n提交文件已保存: submission.csv ({len(submission)} 条预测)")

# 15. 特征重要性分析
print("\n=== 特征重要性分析 ===")
print(f"选择的前10个特征:")
for i, feat in enumerate(selected_features[:10]):
    print(f"{i+1}. {feat}")

# 16. 额外验证：检查预测分布
print("\n=== 预测分布统计 ===")
print(f"训练集目标范围: [{y_train.min():.2f}, {y_train.max():.2f}]")
print(f"预测值范围: [{test_predictions_processed.min():.2f}, {test_predictions_processed.max():.2f}]")
print(f"预测均值: {test_predictions_processed.mean():.2f}, 标准差: {test_predictions_processed.std():.2f}")

# 最终输出要求的格式
print(f"\n最终验证分数:")
print(f"Score= (1.0 / (1.0 + RMSE)) = {score:.4f}")