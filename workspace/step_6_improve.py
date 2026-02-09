import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子以保证可重复性
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

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

print(f"Features after dropping ID and target: {X_train_full.shape[1]}")

# 2. 时间序列分割（最后20%作为验证集，不打乱）
val_size = int(len(X_train_full) * 0.15)  # 稍微减少验证集比例，增加训练数据
X_train = X_train_full.iloc[:-val_size].copy()
X_val = X_train_full.iloc[-val_size:].copy()
y_train = y_train_full[:-val_size]
y_val = y_train_full[-val_size:]

print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

# 3. 高级特征工程
def create_advanced_features(df):
    df = df.copy()
    
    # 原始特征列分组
    state_cols = [col for col in df.columns if col in ['AL', 'AZ', 'CA', 'CO', 'CT', 'FL', 'GA', 'IL', 'IN', 'IA', 
                                                     'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MO', 'NJ', 
                                                     'NM', 'NY', 'NC', 'OH', 'OK', 'OR', 'PA', 'SC', 'TN', 'TX', 
                                                     'VA', 'WA', 'WV', 'WI']]
    
    # 1. 构造滞后差分特征
    for day in ['1', '2']:
        next_day = str(int(day) + 1)
        if f'cli_day{day}' in df.columns and f'cli_day{next_day}' in df.columns:
            df[f'cli_diff_{day}_{next_day}'] = df[f'cli_day{next_day}'] - df[f'cli_day{day}']
        
        if f'ili_day{day}' in df.columns and f'ili_day{next_day}' in df.columns:
            df[f'ili_diff_{day}_{next_day}'] = df[f'ili_day{next_day}'] - df[f'ili_day{day}']
            
        if f'tested_positive_day{day}' in df.columns:
            df[f'tested_pos_log_{day}'] = np.log1p(df[f'tested_positive_day{day}'])
    
    # 2. 构造统计特征
    for prefix in ['cli', 'ili', 'wnohh_cmnty_cli', 'wearing_mask_7d']:
        cols = [col for col in df.columns if col.startswith(f'{prefix}_day')]
        if len(cols) >= 2:
            df[f'{prefix}_mean'] = df[cols].mean(axis=1)
            df[f'{prefix}_std'] = df[cols].std(axis=1)
            df[f'{prefix}_max'] = df[cols].max(axis=1)
            df[f'{prefix}_min'] = df[cols].min(axis=1)
    
    # 3. 构造交互特征（基于领域知识）
    for day in ['1', '2', '3']:
        # 症状与防护行为的交互
        if f'cli_day{day}' in df.columns and f'wearing_mask_7d_day{day}' in df.columns:
            df[f'cli_mask_interaction_day{day}'] = df[f'cli_day{day}'] * df[f'wearing_mask_7d_day{day}']
            df[f'cli_mask_ratio_day{day}'] = df[f'cli_day{day}'] / (df[f'wearing_mask_7d_day{day}'] + 1e-6)
        
        # 症状与担忧程度的交互
        if f'cli_day{day}' in df.columns and f'wworried_catch_covid_day{day}' in df.columns:
            df[f'cli_worried_interaction_day{day}'] = df[f'cli_day{day}'] * df[f'wworried_catch_covid_day{day}']
        
        # 阳性检测与社区症状的交互
        if f'tested_positive_day{day}' in df.columns and f'cli_day{day}' in df.columns:
            df[f'positive_cli_ratio_day{day}'] = df[f'tested_positive_day{day}'] / (df[f'cli_day{day}'] + 1e-6)
    
    # 4. 构造多项式特征（平方项）
    for day in ['1', '2']:
        if f'cli_day{day}' in df.columns:
            df[f'cli_squared_day{day}'] = df[f'cli_day{day}'] ** 2
        if f'ili_day{day}' in df.columns:
            df[f'ili_squared_day{day}'] = df[f'ili_day{day}'] ** 2
    
    # 5. 构造复合特征
    for day in ['1', '2', '3']:
        if f'cli_day{day}' in df.columns and f'ili_day{day}' in df.columns:
            df[f'cli_ili_sum_day{day}'] = df[f'cli_day{day}'] + df[f'ili_day{day}']
            df[f'cli_ili_product_day{day}'] = df[f'cli_day{day}'] * df[f'ili_day{day}']
    
    # 6. 添加时间趋势特征
    if 'tested_positive_day1' in df.columns and 'tested_positive_day2' in df.columns:
        df['positive_growth_rate'] = (df['tested_positive_day2'] - df['tested_positive_day1']) / (df['tested_positive_day1'] + 1e-6)
        df['positive_abs_growth'] = df['tested_positive_day2'] - df['tested_positive_day1']
    
    return df

print("Creating advanced features...")
X_train = create_advanced_features(X_train)
X_val = create_advanced_features(X_val)
X_test = create_advanced_features(X_test)

print(f"Features after advanced engineering: {X_train.shape[1]}")

# 4. 处理缺失值和无穷值
def handle_infinite(df):
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    # 用列中位数填充NaN
    for col in df.columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    return df

X_train = handle_infinite(X_train)
X_val = handle_infinite(X_val)
X_test = handle_infinite(X_test)

# 5. 数据标准化
# 使用RobustScaler对异常值更鲁棒
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 6. 特征选择（使用互信息，更适用于非线性关系）
print("Performing feature selection...")
selector = SelectKBest(score_func=mutual_info_regression, k=min(50, X_train_scaled.shape[1]))
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_val_selected = selector.transform(X_val_scaled)
X_test_selected = selector.transform(X_test_scaled)

print(f"Features after selection: {X_train_selected.shape[1]}")

# 7. 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train_selected)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
X_val_tensor = torch.FloatTensor(X_val_selected)
y_val_tensor = torch.FloatTensor(y_val).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_selected)

# 创建数据加载器
batch_size = 128  # 增加批大小
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 训练时可以打乱
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 8. 定义改进的DNN模型（带残差连接）
class COVIDPredictorV2(nn.Module):
    def __init__(self, input_dim):
        super(COVIDPredictorV2, self).__init__()
        
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
        
        # 第三层
        self.layer3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 第四层
        self.layer4 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.15)
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # 残差连接
        self.residual1 = nn.Linear(input_dim, 128)
        self.residual2 = nn.Linear(256, 64)
        
    def forward(self, x):
        identity1 = x
        
        # 第一层
        out = self.layer1(x)
        
        # 第二层
        out = self.layer2(out)
        
        # 第一残差连接
        if identity1.shape[1] == 128:  # 确保维度匹配
            out = out + self.residual1(identity1)
        
        identity2 = out
        
        # 第三层
        out = self.layer3(out)
        
        # 第四层
        out = self.layer4(out)
        
        # 第二残差连接
        if identity2.shape[1] == 64:  # 确保维度匹配
            out = out + self.residual2(identity2)
        
        # 输出层
        out = self.output_layer(out)
        
        return out

# 9. 定义集成模型类
class ModelEnsemble:
    def __init__(self, model_classes, input_dim):
        self.models = []
        for model_class in model_classes:
            model = model_class(input_dim)
            self.models.append(model)
    
    def train_models(self, train_loader, val_loader, num_epochs=300):
        best_models = []
        for i, model in enumerate(self.models):
            print(f"\nTraining model {i+1}/{len(self.models)}...")
            best_model = self._train_single_model(model, train_loader, val_loader, num_epochs, model_id=i)
            best_models.append(best_model)
        self.models = best_models
    
    def _train_single_model(self, model, train_loader, val_loader, num_epochs, model_id=0):
        criterion = nn.HuberLoss(delta=1.0)  # Huber损失对异常值更鲁棒
        optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        
        # 余弦退火学习率调度
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )
        
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        best_model_state = None
        
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
            
            # 早停机制
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if epoch % 30 == 0:
                print(f'Model {model_id+1}, Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}')
            
            if patience_counter >= patience:
                print(f'Model {model_id+1} early stopping at epoch {epoch}')
                break
        
        # 加载最佳模型
        model.load_state_dict(best_model_state)
        return model
    
    def predict(self, X):
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(X).numpy().flatten()
                predictions.append(pred)
        
        # 使用中位数集成（对异常值更鲁棒）
        predictions = np.stack(predictions, axis=0)
        return np.median(predictions, axis=0)

# 10. 训练集成模型
print("\n" + "="*50)
print("Training Model Ensemble")
print("="*50)

input_dim = X_train_selected.shape[1]

# 定义多个不同架构的模型
model_classes = [
    COVIDPredictorV2,
    lambda input_dim: nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Dropout(0.15),
        nn.Linear(32, 1)
    ),
    lambda input_dim: nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.35),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(64, 1)
    )
]

ensemble = ModelEnsemble(model_classes, input_dim)
ensemble.train_models(train_loader, val_loader, num_epochs=300)

# 11. 在验证集上评估
print("\n" + "="*50)
print("Ensemble Evaluation")
print("="*50)

ensemble_preds = ensemble.predict(X_val_tensor)
val_rmse = np.sqrt(np.mean((ensemble_preds - y_val) ** 2))

# 计算分数
score = 1.0 / (1.0 + val_rmse)
print(f'\nValidation Results:')
print(f'Validation RMSE: {val_rmse:.6f}')
print(f'Score = 1.0 / (1.0 + {val_rmse:.6f}) = {score:.6f}')

# 12. 对测试集进行预测
print("\nMaking predictions on test set...")
test_predictions = ensemble.predict(X_test_tensor)

# 13. 后处理优化
# 应用指数平滑处理时间序列特性
def exponential_smoothing(predictions, alpha=0.1):
    smoothed = np.zeros_like(predictions)
    smoothed[0] = predictions[0]
    for i in range(1, len(predictions)):
        smoothed[i] = alpha * predictions[i] + (1 - alpha) * smoothed[i-1]
    return smoothed

# 确保预测值非负且合理
test_predictions = np.maximum(test_predictions, 0)

# 对预测值进行温和的平滑
test_predictions = exponential_smoothing(test_predictions, alpha=0.15)

# 14. 生成提交文件
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'tested_positive': test_predictions
})

submission_df.to_csv('submission.csv', index=False)
print(f"\nSubmission file saved. Predictions range: [{test_predictions.min():.2f}, {test_predictions.max():.2f}]")

# 15. 模型保存
print("\nSaving model ensemble...")
torch.save({
    'ensemble_state_dicts': [model.state_dict() for model in ensemble.models],
    'scaler': scaler,
    'selector': selector,
    'input_dim': input_dim
}, 'covid_ensemble_model.pth')

print("Done!")