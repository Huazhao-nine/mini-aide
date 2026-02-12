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

# 2. 分离特征和目标（注意：保留 tested_positive_day1 和 tested_positive_day2）
target_col = 'tested_positive_day3'
y_train = train_df[target_col].values.reshape(-1, 1)

# 保留所有特征，包括 tested_positive_day1 和 tested_positive_day2
X_train = train_df.drop(['id', target_col], axis=1)
X_test = test_df.drop(['id'], axis=1)

# 3. 修正的特征工程（避免使用目标列 tested_positive_day3）
def create_advanced_features(df, is_train=True):
    """创建更丰富的特征工程"""
    new_df = df.copy()
    
    # 基础特征分组（使用正确的列名）
    symptom_cols = ['cli', 'ili', 'hh_cmnty_cli', 'nohh_cmnty_cli']
    behavior_cols = ['wearing_mask_7d', 'wshop_indoors', 'wrestaurant_indoors', 
                    'public_transit', 'wlarge_event_indoors']
    belief_cols = ['wbelief_masking_effective', 'wbelief_distancing_effective']
    mental_cols = ['wworried_catch_covid', 'worried_finances']
    other_cols = ['wothers_masked_public', 'wothers_distanced_public', 
                  'wcovid_vaccinated_friends']
    
    # 1. 跨天的聚合特征（只使用day1和day2的数据，避免未来信息）
    all_features = symptom_cols + behavior_cols + belief_cols + mental_cols + other_cols
    
    for feature in all_features:
        for day in [1, 2, 3]:
            # 确保列存在
            col_name = f'{feature}_day{day}'
            if col_name in df.columns:
                # 创建相对变化特征（只对day2和day3）
                if day > 1:
                    prev_col = f'{feature}_day{day-1}'
                    if prev_col in df.columns:
                        new_df[f'{feature}_delta_day{day}'] = df[col_name] - df[prev_col]
                
                # 创建移动平均特征（使用day1-day2的平均值）
                if day == 2:
                    day1_col = f'{feature}_day1'
                    if day1_col in df.columns:
                        new_df[f'{feature}_ma_2d'] = (df[day1_col] + df[col_name]) / 2
    
    # 2. 症状与行为的交叉特征
    for day in [1, 2]:
        # 症状与防护行为的交互
        for symptom in ['cli', 'ili', 'nohh_cmnty_cli']:
            for behavior in ['wearing_mask_7d', 'wothers_masked_public']:
                symptom_col = f'{symptom}_day{day}'
                behavior_col = f'{behavior}_day{day}'
                if symptom_col in df.columns and behavior_col in df.columns:
                    new_df[f'{symptom}_{behavior}_interaction_day{day}'] = (
                        df[symptom_col] * df[behavior_col]
                    )
        
        # 担忧程度与疫苗接种率的交互
        worry_col = f'wworried_catch_covid_day{day}'
        vaccine_col = f'wcovid_vaccinated_friends_day{day}'
        if worry_col in df.columns and vaccine_col in df.columns:
            new_df[f'worried_vaccine_interaction_day{day}'] = (
                df[worry_col] * df[vaccine_col]
            )
        
        # 室内活动与口罩佩戴的交互
        for indoor_activity in ['wrestaurant_indoors', 'wshop_indoors', 'wlarge_event_indoors']:
            activity_col = f'{indoor_activity}_day{day}'
            mask_col = f'wearing_mask_7d_day{day}'
            if activity_col in df.columns and mask_col in df.columns:
                new_df[f'{indoor_activity}_mask_interaction_day{day}'] = (
                    df[activity_col] * df[mask_col]
                )
    
    # 3. 目标变量的滞后特征（仅对训练集，且只使用day1和day2）
    if is_train and 'tested_positive_day1' in df.columns and 'tested_positive_day2' in df.columns:
        # 只使用day1和day2创建趋势特征
        new_df['tested_positive_trend_1_2'] = (
            df['tested_positive_day2'] - df['tested_positive_day1']
        )
    
    # 4. 创建风险评分特征（使用day2的数据）
    day2_risk_factors = []
    for risk_factor in ['cli_day2', 'wrestaurant_indoors_day2', 'wlarge_event_indoors_day2']:
        if risk_factor in df.columns:
            day2_risk_factors.append(df[risk_factor])
    
    if day2_risk_factors:
        new_df['risk_score_day2'] = np.mean(day2_risk_factors, axis=0)
    
    # 5. 防护评分特征（使用day2的数据）
    day2_protection_factors = []
    for protection_factor in ['wearing_mask_7d_day2', 'wothers_masked_public_day2', 'wcovid_vaccinated_friends_day2']:
        if protection_factor in df.columns:
            day2_protection_factors.append(df[protection_factor])
    
    if day2_protection_factors:
        new_df['protection_score_day2'] = np.mean(day2_protection_factors, axis=0)
    
    # 6. 添加简单的统计特征
    for feature in all_features:
        day_cols = [f'{feature}_day{day}' for day in [1, 2] if f'{feature}_day{day}' in df.columns]
        if len(day_cols) >= 2:
            new_df[f'{feature}_std'] = df[day_cols].std(axis=1)
            new_df[f'{feature}_max'] = df[day_cols].max(axis=1)
            new_df[f'{feature}_min'] = df[day_cols].min(axis=1)
    
    return new_df

print("创建高级特征...")
X_train = create_advanced_features(X_train, is_train=True)
X_test = create_advanced_features(X_test, is_train=False)

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

# 5. 特征标准化（使用StandardScaler）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_split)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 6. 特征选择（使用SelectKBest）
print("进行特征选择...")
selector = SelectKBest(score_func=f_regression, k=15)
X_train_selected = selector.fit_transform(X_train_scaled, y_train_split.ravel())
X_val_selected = selector.transform(X_val_scaled)
X_test_selected = selector.transform(X_test_scaled)

selected_indices = selector.get_support(indices=True)
selected_features = X_train.columns[selected_indices]
print(f"选择的特征: {list(selected_features)}")
print(f"特征选择后维度: {X_train_selected.shape[1]}")

# 7. 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train_selected)
y_train_tensor = torch.FloatTensor(y_train_split)
X_val_tensor = torch.FloatTensor(X_val_selected)
y_val_tensor = torch.FloatTensor(y_val)
X_test_tensor = torch.FloatTensor(X_test_selected)

# 创建DataLoader（关键：时间序列数据不shuffle）
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # 不shuffle！
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 8. 定义神经网络模型（按照要求：3层网络）
class COVIDPredictor(nn.Module):
    def __init__(self, input_dim):
        super(COVIDPredictor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.network(x)

# 9. 训练设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

model = COVIDPredictor(input_dim=X_train_selected.shape[1]).to(device)

# 使用MAE损失（按照要求）
criterion = nn.L1Loss()

# 使用Adam优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                 factor=0.5, patience=10, verbose=True)

# 10. 训练循环
epochs = 200
best_val_loss = float('inf')
patience = 20
patience_counter = 0

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
        optimizer.step()
        
        train_loss += loss.item() * batch_X.size(0)
    
    train_loss /= len(train_loader.dataset)
    
    # 验证阶段
    model.eval()
    val_loss = 0
    val_preds = []
    val_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            val_loss += criterion(outputs, batch_y).item() * batch_X.size(0)
            val_preds.extend(outputs.cpu().numpy())
            val_targets.extend(batch_y.cpu().numpy())
    
    val_loss /= len(val_loader.dataset)
    
    # 学习率调整
    scheduler.step(val_loss)
    
    # 计算验证集RMSE
    val_preds = np.array(val_preds).flatten()
    val_targets = np.array(val_targets).flatten()
    val_rmse = np.sqrt(np.mean((val_preds - val_targets) ** 2))
    
    # 早停检查
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
        best_val_rmse = val_rmse
    else:
        patience_counter += 1
    
    if epoch % 20 == 0:
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}')
    
    if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch}')
        break

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

# 计算RMSE
rmse = np.sqrt(np.mean((val_predictions - val_targets) ** 2))
mae = np.mean(np.abs(val_predictions - val_targets))
score = 1.0 / (1.0 + rmse)

print(f'验证集RMSE: {rmse:.4f}')
print(f'验证集MAE: {mae:.4f}')

# 13. 在测试集上进行预测
test_predictions = []
model.eval()
with torch.no_grad():
    # 分批预测
    batch_size_test = 64
    X_test_tensor_device = X_test_tensor.to(device)
    for i in range(0, len(X_test_tensor_device), batch_size_test):
        batch = X_test_tensor_device[i:i+batch_size_test]
        preds = model(batch)
        test_predictions.extend(preds.cpu().numpy())

test_predictions = np.array(test_predictions).flatten()

# 14. 后处理（只处理负值，不进行clipping上限）
test_predictions = np.maximum(test_predictions, 0)

# 15. 生成提交文件
submission = pd.DataFrame({
    'id': test_df['id'],
    'tested_positive': test_predictions
})

submission.to_csv('submission.csv', index=False)
print(f"提交文件已保存，包含 {len(submission)} 条预测结果")

# 16. 打印最终分数（按要求格式）
print(f"Score= (1.0 / (1.0 + RMSE)) = {score:.4f}")