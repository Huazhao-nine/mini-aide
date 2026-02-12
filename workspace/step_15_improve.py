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

# 1. 加载数据
train_df = pd.read_csv('/home/huazhao/DL-HW/ML-2025/mini-aide/input/train.csv')
test_df = pd.read_csv('/home/huazhao/DL-HW/ML-2025/mini-aide/input/test.csv')

# 2. 分离特征和目标
target_col = 'tested_positive_day3'
y_train = train_df[target_col].values.reshape(-1, 1)
X_train = train_df.drop(['id', target_col], axis=1)
X_test = test_df.drop(['id'], axis=1)

# 3. 简洁但有效的特征工程
def create_smart_features(df):
    """基于领域知识创建特征"""
    new_df = df.copy()
    
    # 获取州特征
    state_cols = [col for col in df.columns if len(col) == 2 and col.isupper()]
    
    # 1. 关键滞后特征
    for feature in ['tested_positive', 'cli', 'ili', 'nohh_cmnty_cli']:
        if f'{feature}_day2' in df.columns and f'{feature}_day1' in df.columns:
            new_df[f'{feature}_trend'] = df[f'{feature}_day2'] - df[f'{feature}_day1']
            new_df[f'{feature}_growth'] = (df[f'{feature}_day2'] - df[f'{feature}_day1']) / (df[f'{feature}_day1'] + 1e-8)
    
    # 2. 关键交互特征（基于流行病学知识）
    for day in [1, 2, 3]:
        # 口罩使用与社区传播的交互
        if f'wearing_mask_7d_day{day}' in df.columns and f'cli_day{day}' in df.columns:
            new_df[f'mask_protection_index_day{day}'] = df[f'cli_day{day}'] * (100 - df[f'wearing_mask_7d_day{day}']) / 100
        
        # 室内活动风险指标
        if all(col in df.columns for col in [f'wlarge_event_indoors_day{day}', 
                                            f'wrestaurant_indoors_day{day}',
                                            f'wshop_indoors_day{day}']):
            new_df[f'indoor_risk_day{day}'] = (
                df[f'wlarge_event_indoors_day{day}'] * 0.4 +
                df[f'wrestaurant_indoors_day{day}'] * 0.4 +
                df[f'wshop_indoors_day{day}'] * 0.2
            )
        
        # 疫苗接种与防护行为的协同效应
        if all(col in df.columns for col in [f'wcovid_vaccinated_friends_day{day}',
                                            f'wearing_mask_7d_day{day}']):
            new_df[f'combined_protection_day{day}'] = (
                df[f'wcovid_vaccinated_friends_day{day}'] * 0.6 +
                df[f'wearing_mask_7d_day{day}'] * 0.4
            )
    
    # 3. 时间序列聚合特征（仅使用day1和day2，避免数据泄露）
    for feature in ['cli', 'ili', 'tested_positive', 'wearing_mask_7d']:
        day1_col = f'{feature}_day1'
        day2_col = f'{feature}_day2'
        if day1_col in df.columns and day2_col in df.columns:
            new_df[f'{feature}_mean_2days'] = df[[day1_col, day2_col]].mean(axis=1)
            new_df[f'{feature}_std_2days'] = df[[day1_col, day2_col]].std(axis=1)
    
    # 4. 复合传播风险指标（使用day3的数据，因为这是我们已知的特征）
    for day in [1, 2, 3]:
        if all(col in df.columns for col in [f'cli_day{day}', f'nohh_cmnty_cli_day{day}',
                                            f'wlarge_event_indoors_day{day}']):
            new_df[f'transmission_risk_day{day}'] = (
                df[f'cli_day{day}'] * 0.4 +
                df[f'nohh_cmnty_cli_day{day}'] * 0.4 +
                df[f'wlarge_event_indoors_day{day}'] * 0.2
            )
    
    return new_df

print("原始特征数量:", X_train.shape[1])
X_train = create_smart_features(X_train)
X_test = create_smart_features(X_test)
print("特征工程后特征数量:", X_train.shape[1])

# 4. 时间序列验证集划分（最后20%）
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

# 6. 特征选择（选择最重要的15个特征）
selector = SelectKBest(score_func=f_regression, k=min(15, X_train_scaled.shape[1]))
X_train_selected = selector.fit_transform(X_train_scaled, y_train_split.ravel())
X_val_selected = selector.transform(X_val_scaled)
X_test_selected = selector.transform(X_test_scaled)

print(f"特征选择后维度: {X_train_selected.shape[1]}")
print(f"选择的特征分数: {selector.scores_[selector.get_support()]}")

# 7. 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train_selected)
y_train_tensor = torch.FloatTensor(y_train_split)
X_val_tensor = torch.FloatTensor(X_val_selected)
y_val_tensor = torch.FloatTensor(y_val)
X_test_tensor = torch.FloatTensor(X_test_selected)

# 创建DataLoader（时间序列禁止shuffle）
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 8. 定义简洁但有效的神经网络模型
class COVIDPredictor(nn.Module):
    def __init__(self, input_dim):
        super(COVIDPredictor, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            
            nn.Linear(16, 1)
        )
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.model(x)

# 9. 训练设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

model = COVIDPredictor(input_dim=X_train_selected.shape[1]).to(device)

# 使用MAE损失函数（比MSE更抗噪）
criterion = nn.L1Loss()

# 使用Adam优化器
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 学习率调度器（ReduceLROnPlateau）
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, verbose=True
)

# 10. 训练循环（带早停）
epochs = 300
best_val_rmse = float('inf')
patience = 25
patience_counter = 0
best_model_state = None

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        train_loss += loss.item() * batch_X.size(0)
    
    train_loss /= len(train_loader.dataset)
    
    # 验证阶段
    model.eval()
    val_loss = 0
    val_predictions = []
    val_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_X.size(0)
            
            val_predictions.extend(outputs.cpu().numpy())
            val_targets.extend(batch_y.cpu().numpy())
    
    val_loss /= len(val_loader.dataset)
    
    # 计算验证集RMSE
    val_predictions = np.array(val_predictions).flatten()
    val_targets = np.array(val_targets).flatten()
    val_rmse = np.sqrt(np.mean((val_predictions - val_targets) ** 2))
    
    # 更新学习率
    scheduler.step(val_rmse)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    # 早停检查
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        patience_counter = 0
        best_model_state = model.state_dict().copy()
        best_val_predictions = val_predictions.copy()
        best_val_targets = val_targets.copy()
    else:
        patience_counter += 1
    
    if epoch % 20 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Val RMSE: {val_rmse:.4f}, LR: {current_lr:.6f}, Patience: {patience_counter}')
    
    if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch}')
        break

# 11. 加载最佳模型
model.load_state_dict(best_model_state)
print(f"最佳模型验证集RMSE: {best_val_rmse:.4f}")

# 12. 计算最终验证分数
final_rmse = best_val_rmse
score = 1.0 / (1.0 + final_rmse)

print(f'最终验证集RMSE: {final_rmse:.4f}')
print(f'Score = (1.0 / (1.0 + RMSE)) = {score:.4f}')

# 13. 在测试集上进行预测
model.eval()
test_predictions = []
with torch.no_grad():
    # 分批预测
    batch_size_test = 64
    for i in range(0, len(X_test_tensor), batch_size_test):
        batch = X_test_tensor[i:i+batch_size_test].to(device)
        preds = model(batch)
        test_predictions.extend(preds.cpu().numpy())

test_predictions = np.array(test_predictions).flatten()

# 14. 后处理（基于领域知识）
# 确保没有负值
test_predictions = np.maximum(test_predictions, 0)

# 对于极端高值进行温和的平滑（仅对异常高值）
mean_val = np.mean(test_predictions)
std_val = np.std(test_predictions)
upper_bound = mean_val + 3 * std_val
test_predictions = np.where(test_predictions > upper_bound, 
                           (test_predictions + mean_val) / 2, 
                           test_predictions)

# 15. 生成提交文件
submission = pd.DataFrame({
    'id': test_df['id'],
    'tested_positive': test_predictions
})

submission.to_csv('submission.csv', index=False)
print(f"提交文件已保存，包含 {len(submission)} 条预测结果")

# 16. 特征重要性分析（可选）
feature_names = X_train.columns[selector.get_support()]
feature_scores = selector.scores_[selector.get_support()]

print("\nTop 15 特征重要性:")
for name, score in sorted(zip(feature_names, feature_scores), key=lambda x: x[1], reverse=True)[:15]:
    print(f"  {name}: {score:.2f}")

# 17. 绘制训练曲线
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss (MAE)')
plt.plot(val_losses, label='Val Loss (MAE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training History')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(best_val_targets, best_val_predictions, alpha=0.6)
plt.plot([0, max(best_val_targets)], [0, max(best_val_targets)], 'r--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title(f'Validation Set Predictions (RMSE={final_rmse:.2f})')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_results.png', dpi=100)
print("结果图已保存为 training_results.png")

# 18. 最终分数输出
print(f"\n{'='*50}")
print(f"最终结果:")
print(f"验证集RMSE: {final_rmse:.4f}")
print(f"Score= (1.0 / (1.0 + RMSE)) = {score:.4f}")
print(f"{'='*50}")