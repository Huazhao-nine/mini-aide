import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
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

print(f"Features after dropping ID and target: {X_train_full.shape[1]}")

# 2. 时间序列分割（最后20%作为验证集，不打乱）
val_size = int(len(X_train_full) * 0.2)
X_train = X_train_full.iloc[:-val_size].copy()
X_val = X_train_full.iloc[-val_size:].copy()
y_train = y_train_full[:-val_size]
y_val = y_train_full[-val_size:]

print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

# 3. 增强的特征工程
def create_advanced_features(df):
    df = df.copy()
    
    # 基础特征列表
    cli_cols = [col for col in df.columns if 'cli_day' in col]
    ili_cols = [col for col in df.columns if 'ili_day' in col]
    mask_cols = [col for col in df.columns if 'wearing_mask' in col]
    test_cols = [col for col in df.columns if 'tested_positive' in col]
    
    # 1. 创建更有意义的交互特征
    for day in [1, 2, 3]:
        if f'cli_day{day}' in df.columns and f'wearing_mask_7d_day{day}' in df.columns:
            df[f'cli_mask_ratio_day{day}'] = df[f'cli_day{day}'] / (df[f'wearing_mask_7d_day{day}'] + 1e-6)
        
        if f'ili_day{day}' in df.columns and f'wearing_mask_7d_day{day}' in df.columns:
            df[f'ili_mask_ratio_day{day}'] = df[f'ili_day{day}'] / (df[f'wearing_mask_7d_day{day}'] + 1e-6)
    
    # 2. 创建时序特征（差分）
    if 'cli_day2' in df.columns and 'cli_day1' in df.columns:
        df['cli_change_2_1'] = df['cli_day2'] - df['cli_day1']
    if 'ili_day2' in df.columns and 'ili_day1' in df.columns:
        df['ili_change_2_1'] = df['ili_day2'] - df['ili_day1']
    
    # 3. 创建状态级别的聚合特征
    state_cols = ['AL', 'AZ', 'CA', 'CO', 'CT', 'FL', 'GA', 'IL', 'IN', 'IA', 'KS', 
                 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MO', 'NJ', 'NM', 'NY', 
                 'NC', 'OH', 'OK', 'OR', 'PA', 'SC', 'TN', 'TX', 'VA', 'WA', 'WV', 'WI']
    
    # 对于每个州，计算相关特征的平均值
    for col in cli_cols + ili_cols + mask_cols:
        if col in df.columns:
            # 创建与州的交互特征
            for state in state_cols:
                if state in df.columns:
                    df[f'{col}_{state}'] = df[col] * df[state]
    
    # 4. 创建多项式特征（二次项）
    important_features = ['cli_day3', 'ili_day3', 'wearing_mask_7d_day3', 
                         'cli_day2', 'ili_day2', 'wearing_mask_7d_day2']
    
    for feat in important_features:
        if feat in df.columns:
            df[f'{feat}_squared'] = df[feat] ** 2
            df[f'{feat}_log'] = np.log1p(np.abs(df[feat]))
    
    # 5. 创建复合指标
    if 'cli_day3' in df.columns and 'ili_day3' in df.columns:
        df['symptoms_composite'] = df['cli_day3'] * 0.7 + df['ili_day3'] * 0.3
    
    # 6. 处理缺失值（如果有的话）
    df = df.fillna(df.median())
    
    return df

print("Creating advanced features...")
X_train = create_advanced_features(X_train)
X_val = create_advanced_features(X_val)
X_test = create_advanced_features(X_test)

print(f"Features after advanced engineering: {X_train.shape[1]}")

# 4. 数据标准化和变换
# 对特征使用PowerTransformer处理偏态分布
pt = PowerTransformer(method='yeo-johnson')
X_train_pt = pt.fit_transform(X_train)
X_val_pt = pt.transform(X_val)
X_test_pt = pt.transform(X_test)

# 然后进行标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_pt)
X_val_scaled = scaler.transform(X_val_pt)
X_test_scaled = scaler.transform(X_test_pt)

# 5. 智能特征选择
print("Performing feature selection...")
# 使用RandomForest进行特征重要性排序
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)

# 选择重要性大于平均重要性的特征
importances = rf.feature_importances_
avg_importance = np.mean(importances)
selected_indices = np.where(importances > avg_importance * 0.3)[0]  # 比平均重要性高30%

if len(selected_indices) < 20:
    # 如果选出的特征太少，选择前50个最重要的特征
    selected_indices = np.argsort(importances)[-50:]

X_train_selected = X_train_scaled[:, selected_indices]
X_val_selected = X_val_scaled[:, selected_indices]
X_test_selected = X_test_scaled[:, selected_indices]

print(f"Selected {len(selected_indices)} important features")

# 6. 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train_selected)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
X_val_tensor = torch.FloatTensor(X_val_selected)
y_val_tensor = torch.FloatTensor(y_val).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_selected)

# 对目标值进行log1p变换以处理偏态分布
y_train_log = torch.log1p(y_train_tensor)
y_val_log = torch.log1p(y_val_tensor)

# 创建数据加载器
batch_size = 128
train_dataset = TensorDataset(X_train_tensor, y_train_log)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(X_val_tensor, y_val_log)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 7. 定义更强大的DNN模型
class EnhancedCOVIDPredictor(nn.Module):
    def __init__(self, input_dim):
        super(EnhancedCOVIDPredictor, self).__init__()
        
        # 更深的网络结构
        self.network = nn.Sequential(
            # Block 1
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Block 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            
            # Block 3
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Block 4
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            # Block 5
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Output
            nn.Linear(16, 1)
        )
        
        # 残差连接（如果维度匹配）
        self.residual = nn.Linear(input_dim, 1) if input_dim != 1 else None
        
    def forward(self, x):
        out = self.network(x)
        if self.residual is not None:
            residual_out = self.residual(x)
            out = out + 0.1 * residual_out  # 小权重残差
        return out

# 8. 模型集成：训练多个模型
class ModelEnsemble:
    def __init__(self, input_dim, n_models=5):
        self.models = []
        self.input_dim = input_dim
        self.n_models = n_models
        
    def train_models(self, train_loader, val_loader, epochs=300):
        for i in range(self.n_models):
            print(f"\nTraining model {i+1}/{self.n_models}")
            
            # 每个模型使用不同的随机种子
            torch.manual_seed(42 + i)
            np.random.seed(42 + i)
            
            model = EnhancedCOVIDPredictor(self.input_dim)
            
            # 不同的优化器配置
            if i % 3 == 0:
                optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
            elif i % 3 == 1:
                optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
            else:
                optimizer = optim.RAdam(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
            
            criterion = nn.MSELoss()
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-5
            )
            
            # 训练单个模型
            best_val_loss = float('inf')
            patience = 20
            patience_counter = 0
            
            for epoch in range(epochs):
                # 训练阶段
                model.train()
                train_loss = 0
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    predictions = model(batch_x)
                    loss = criterion(predictions, batch_y)
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    train_loss += loss.item()
                
                # 验证阶段
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        predictions = model(batch_x)
                        loss = criterion(predictions, batch_y)
                        val_loss += loss.item()
                
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                
                # 学习率调度
                scheduler.step()
                
                # 早停机制
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f'Model {i+1}: Early stopping at epoch {epoch}')
                    break
                
                if epoch % 30 == 0:
                    print(f'Model {i+1}, Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}')
            
            # 保存最佳模型
            model.load_state_dict(best_model_state)
            self.models.append(model)
    
    def predict(self, x):
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        # 平均预测
        return torch.mean(torch.stack(predictions), dim=0)

# 9. 训练集成模型
print("\nTraining ensemble of models...")
input_dim = X_train_selected.shape[1]
ensemble = ModelEnsemble(input_dim, n_models=5)
ensemble.train_models(train_loader, val_loader, epochs=300)

# 10. 在验证集上评估
print("\nEvaluating ensemble on validation set...")
ensemble_models = ensemble.models

# 计算每个模型的验证集预测
all_val_predictions = []
for i, model in enumerate(ensemble_models):
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_tensor)
        # 反向log1p变换
        val_pred_exp = torch.expm1(val_pred)
        all_val_predictions.append(val_pred_exp)

# 集成预测（加权平均，给表现更好的模型更高权重）
val_predictions_stack = torch.stack(all_val_predictions)
weights = torch.ones(len(ensemble_models)) / len(ensemble_models)  # 可以基于验证损失调整权重
final_val_predictions = torch.sum(val_predictions_stack * weights.view(-1, 1, 1), dim=0)

# 计算RMSE（在原始尺度上）
val_rmse = torch.sqrt(torch.mean((final_val_predictions - y_val_tensor) ** 2)).item()

# 计算分数
score = 1.0 / (1.0 + val_rmse)
print(f'\nValidation Results:')
print(f'Validation RMSE: {val_rmse:.6f}')
print(f'Score = 1.0 / (1.0 + {val_rmse:.6f}) = {score:.6f}')

# 11. 对测试集进行预测
print("\nMaking predictions on test set...")
all_test_predictions = []

for model in ensemble_models:
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_tensor)
        test_pred_exp = torch.expm1(test_pred)
        all_test_predictions.append(test_pred_exp.numpy().flatten())

# 集成测试集预测
test_predictions_stack = np.array(all_test_predictions)
final_test_predictions = np.mean(test_predictions_stack, axis=0)

# 确保预测值非负
final_test_predictions = np.maximum(final_test_predictions, 0)

# 12. 生成提交文件
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'tested_positive': final_test_predictions
})

submission_df.to_csv('submission.csv', index=False)
print(f"\nSubmission file saved.")
print(f"Number of models in ensemble: {len(ensemble_models)}")
print(f"Selected features: {len(selected_indices)}")
print(f"Predictions range: [{final_test_predictions.min():.2f}, {final_test_predictions.max():.2f}]")
print(f"Mean prediction: {final_test_predictions.mean():.2f}")
print("Done!")