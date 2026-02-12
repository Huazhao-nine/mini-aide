import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子保证可重复性
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

# 1. 加载数据
train_df = pd.read_csv('/home/huazhao/DL-HW/ML-2025/mini-aide/input/train.csv')
test_df = pd.read_csv('/home/huazhao/DL-HW/ML-2025/mini-aide/input/test.csv')

# 2. 分离特征和目标
target_col = 'tested_positive_day3'
y_train = train_df[target_col].values.reshape(-1, 1)
X_train = train_df.drop(['id', target_col], axis=1)
X_test = test_df.drop(['id'], axis=1)

# 3. 高级特征工程
def create_advanced_features(df):
    """创建更丰富的特征工程"""
    new_df = df.copy()
    
    # 基础特征列表（按类别分组）
    symptom_cols = ['cli', 'ili', 'hh_cmnty_cli', 'nohh_cmnty_cli']
    behavior_cols = ['wearing_mask_7d', 'shop_indoors', 'restaurant_indoors', 
                    'public_transit', 'large_event_indoors']
    belief_cols = ['belief_masking_effective', 'belief_distancing_effective']
    mental_cols = ['worried_catch_covid', 'worried_finances']
    other_cols = ['others_masked_public', 'others_distanced_public', 
                  'covid_vaccinated_friends']
    
    # 1. 跨天的聚合特征
    for feature in symptom_cols + behavior_cols + belief_cols + mental_cols + other_cols:
        for day in [1, 2, 3]:
            if f'{feature}_day{day}' in df.columns:
                col_name = f'{feature}_day{day}'
                
                # 创建相对变化特征（第2天-第1天，第3天-第2天）
                if day > 1:
                    prev_col = f'{feature}_day{day-1}'
                    if prev_col in df.columns and col_name in df.columns:
                        new_df[f'{feature}_delta_{day}'] = df[col_name] - df[prev_col]
                
                # 创建移动平均特征
                if day == 3:
                    new_df[f'{feature}_ma_3d'] = (df[f'{feature}_day1'] + 
                                                   df[f'{feature}_day2'] + 
                                                   df[f'{feature}_day3']) / 3
    
    # 2. 症状与行为的交叉特征（更全面）
    for day in [1, 2, 3]:
        # 症状与防护行为的交互
        for symptom in ['cli', 'ili', 'nohh_cmnty_cli']:
            for behavior in ['wearing_mask_7d', 'others_masked_public']:
                if f'{symptom}_day{day}' in df.columns and f'{behavior}_day{day}' in df.columns:
                    new_df[f'{symptom}_{behavior}_interaction_day{day}'] = (
                        df[f'{symptom}_day{day}'] * df[f'{behavior}_day{day}']
                    )
        
        # 担忧程度与疫苗接种率的交互
        if f'worried_catch_covid_day{day}' in df.columns and f'covid_vaccinated_friends_day{day}' in df.columns:
            new_df[f'worried_vaccine_interaction_day{day}'] = (
                df[f'worried_catch_covid_day{day}'] * df[f'covid_vaccinated_friends_day{day}']
            )
        
        # 室内活动与口罩佩戴的交互
        for indoor_activity in ['restaurant_indoors', 'shop_indoors', 'large_event_indoors']:
            if f'{indoor_activity}_day{day}' in df.columns and f'wearing_mask_7d_day{day}' in df.columns:
                new_df[f'{indoor_activity}_mask_interaction_day{day}'] = (
                    df[f'{indoor_activity}_day{day}'] * df[f'wearing_mask_7d_day{day}']
                )
    
    # 3. 目标变量的滞后特征（仅训练集有）
    if 'tested_positive_day1' in df.columns and 'tested_positive_day2' in df.columns:
        for day in [1, 2]:
            new_df[f'tested_positive_delta_{day+1}_{day}'] = (
                df[f'tested_positive_day{day+1}'] - df[f'tested_positive_day{day}']
            )
        new_df['tested_positive_trend'] = (
            (df['tested_positive_day3'] - df['tested_positive_day1']) / 2
        )
    
    # 4. 创建风险评分特征
    for day in [1, 2, 3]:
        risk_factors = []
        if f'cli_day{day}' in df.columns:
            risk_factors.append(df[f'cli_day{day}'])
        if f'restaurant_indoors_day{day}' in df.columns:
            risk_factors.append(df[f'restaurant_indoors_day{day}'])
        if f'large_event_indoors_day{day}' in df.columns:
            risk_factors.append(df[f'large_event_indoors_day{day}'])
        
        if risk_factors:
            new_df[f'risk_score_day{day}'] = np.mean(risk_factors, axis=0)
    
    # 5. 防护评分特征
    for day in [1, 2, 3]:
        protection_factors = []
        if f'wearing_mask_7d_day{day}' in df.columns:
            protection_factors.append(df[f'wearing_mask_7d_day{day}'])
        if f'others_masked_public_day{day}' in df.columns:
            protection_factors.append(df[f'others_masked_public_day{day}'])
        if f'covid_vaccinated_friends_day{day}' in df.columns:
            protection_factors.append(df[f'covid_vaccinated_friends_day{day}'])
        
        if protection_factors:
            new_df[f'protection_score_day{day}'] = np.mean(protection_factors, axis=0)
    
    return new_df

print("创建高级特征...")
X_train = create_advanced_features(X_train)
X_test = create_advanced_features(X_test)

# 确保测试集有训练集的所有列（除了目标变量的滞后特征）
for col in X_train.columns:
    if col not in X_test.columns and 'tested_positive' not in col:
        X_test[col] = 0

# 重新排列列顺序，确保一致
common_cols = [col for col in X_train.columns if col in X_test.columns]
X_train = X_train[common_cols]
X_test = X_test[common_cols]

print(f"特征工程后特征数量: {X_train.shape[1]}")

# 4. 时间序列验证集划分（最后20%）
split_idx = int(len(X_train) * 0.8)
X_val = X_train.iloc[split_idx:].copy()
y_val = y_train[split_idx:].copy()
X_train_split = X_train.iloc[:split_idx].copy()
y_train_split = y_train[:split_idx].copy()

print(f"训练集大小: {len(X_train_split)}, 验证集大小: {len(X_val)}")

# 5. 特征标准化（使用RobustScaler对异常值更稳健）
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_split)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 6. 特征选择（使用多种方法）
print("进行特征选择...")

# 方法1：基于随机森林的特征重要性
rf = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
rf.fit(X_train_scaled, y_train_split.ravel())
feature_importances = rf.feature_importances_

# 选择重要性前30的特征
top_n = min(30, X_train_scaled.shape[1])
important_indices = np.argsort(feature_importances)[-top_n:]

X_train_selected = X_train_scaled[:, important_indices]
X_val_selected = X_val_scaled[:, important_indices]
X_test_selected = X_test_scaled[:, important_indices]

print(f"特征选择后维度: {X_train_selected.shape[1]}")

# 7. PCA降维（保留95%的方差）
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_selected)
X_val_pca = pca.transform(X_val_selected)
X_test_pca = pca.transform(X_test_selected)

print(f"PCA后维度: {X_train_pca.shape[1]}")

# 8. 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train_pca)
y_train_tensor = torch.FloatTensor(y_train_split)
X_val_tensor = torch.FloatTensor(X_val_pca)
y_val_tensor = torch.FloatTensor(y_val)
X_test_tensor = torch.FloatTensor(X_test_pca)

# 创建DataLoader（注意：时间序列数据不shuffle训练集）
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # 关键修改！
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 9. 定义更复杂的神经网络模型
class COVIDPredictorV2(nn.Module):
    def __init__(self, input_dim):
        super(COVIDPredictorV2, self).__init__()
        
        self.encoder = nn.Sequential(
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
        )
        
        # 残差连接
        self.residual = nn.Sequential(
            nn.Linear(32 + input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        self.output = nn.Linear(16, 1)
        
    def forward(self, x):
        encoded = self.encoder(x)
        # 残差连接：将原始输入与编码特征连接
        residual_input = torch.cat([encoded, x], dim=1)
        residual_out = self.residual(residual_input)
        return self.output(residual_out)

# 10. 训练设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

model = COVIDPredictorV2(input_dim=X_train_pca.shape[1]).to(device)

# 使用Huber损失（对异常值更鲁棒）
criterion = nn.HuberLoss(delta=1.0)

# 使用AdamW优化器（更好的权重衰减）
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# 使用余弦退火学习率调度
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-5
)

# 11. 训练循环
epochs = 300
best_val_loss = float('inf')
patience = 30
patience_counter = 0

train_losses = []
val_losses = []

print("开始训练...")
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
        
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        train_loss += loss.item() * batch_X.size(0)
    
    scheduler.step()
    
    train_loss /= len(train_loader.dataset)
    
    # 验证阶段
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            val_loss += criterion(outputs, batch_y).item() * batch_X.size(0)
    
    val_loss /= len(val_loader.dataset)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    # 早停检查
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model_v2.pth')
    else:
        patience_counter += 1
    
    if epoch % 30 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}')
    
    if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch}')
        break

# 12. 加载最佳模型
model.load_state_dict(torch.load('best_model_v2.pth'))

# 13. 在验证集上评估
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
score = 1.0 / (1.0 + rmse)

print(f'验证集RMSE: {rmse:.4f}')
print(f'验证集MAE: {mae:.4f}')
print(f'Score = (1.0 / (1.0 + RMSE)) = {score:.4f}')

# 14. 在测试集上进行预测
test_predictions = []
model.eval()
with torch.no_grad():
    # 分批预测以避免内存问题
    batch_size_test = 64
    X_test_tensor_device = X_test_tensor.to(device)
    for i in range(0, len(X_test_tensor_device), batch_size_test):
        batch = X_test_tensor_device[i:i+batch_size_test]
        preds = model(batch)
        test_predictions.extend(preds.cpu().numpy())

test_predictions = np.array(test_predictions).flatten()

# 15. 后处理
# 将负值设为0（目标变量为百分比，不可能为负）
test_predictions = np.maximum(test_predictions, 0)

# 16. 生成提交文件
submission = pd.DataFrame({
    'id': test_df['id'],
    'tested_positive': test_predictions
})

submission.to_csv('submission.csv', index=False)
print(f"提交文件已保存，包含 {len(submission)} 条预测结果")

# 17. 模型性能可视化（可选）
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training History')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(val_targets, val_predictions, alpha=0.5)
plt.plot([val_targets.min(), val_targets.max()], 
         [val_targets.min(), val_targets.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Validation Predictions vs Actual (RMSE={rmse:.3f})')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=100, bbox_inches='tight')
plt.show()

# 18. 打印最终分数（按要求格式）
print(f"Score= (1.0 / (1.0 + RMSE)) = {score:.4f}")