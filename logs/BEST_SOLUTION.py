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

# 2. 分离特征和目标
# 删除不需要的列：id列和测试集中不存在的目标列
target_col = 'tested_positive_day3'

# 提取训练集的目标值
y_train = train_df[target_col].values.reshape(-1, 1)

# 删除目标列和id列
X_train = train_df.drop(['id', target_col], axis=1)
X_test = test_df.drop(['id'], axis=1)

# 3. 构建交互特征
def create_interaction_features(df):
    """构建交互特征"""
    new_df = df.copy()
    
    # 症状指标和行为指标的交互
    for day in [1, 2, 3]:
        # cli与口罩佩戴率的交互
        new_df[f'cli_mask_day{day}'] = new_df[f'cli_day{day}'] * new_df[f'wearing_mask_7d_day{day}']
        # cli与室内用餐的交互
        new_df[f'cli_restaurant_day{day}'] = new_df[f'cli_day{day}'] * new_df[f'wrestaurant_indoors_day{day}']
        # 担心感染与口罩佩戴的交互
        new_df[f'worried_mask_day{day}'] = new_df[f'wworried_catch_covid_day{day}'] * new_df[f'wearing_mask_7d_day{day}']
        # 疫苗接种朋友比例与行为指标的交互
        new_df[f'vax_restaurant_day{day}'] = new_df[f'wcovid_vaccinated_friends_day{day}'] * new_df[f'wrestaurant_indoors_day{day}']
    
    return new_df

X_train = create_interaction_features(X_train)
X_test = create_interaction_features(X_test)

print(f"特征数量: {X_train.shape[1]}")

# 4. 时间序列验证集划分（最后20%）
split_idx = int(len(X_train) * 0.8)
X_val = X_train.iloc[split_idx:].copy()
y_val = y_train[split_idx:].copy()
X_train_split = X_train.iloc[:split_idx].copy()
y_train_split = y_train[:split_idx].copy()

print(f"训练集大小: {len(X_train_split)}, 验证集大小: {len(X_val)}")

# 5. 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_split)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 6. 特征选择
k = 15
selector = SelectKBest(score_func=f_regression, k=k)
X_train_selected = selector.fit_transform(X_train_scaled, y_train_split.ravel())
X_val_selected = selector.transform(X_val_scaled)
X_test_selected = selector.transform(X_test_scaled)

print(f"特征选择后维度: {X_train_selected.shape[1]}")

# 7. 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train_selected)
y_train_tensor = torch.FloatTensor(y_train_split)
X_val_tensor = torch.FloatTensor(X_val_selected)
y_val_tensor = torch.FloatTensor(y_val)
X_test_tensor = torch.FloatTensor(X_test_selected)

# 创建DataLoader
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 训练集可以shuffle
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # 验证集必须shuffle=False

# 8. 定义神经网络模型
class COVIDPredictor(nn.Module):
    def __init__(self, input_dim):
        super(COVIDPredictor, self).__init__()
        self.model = nn.Sequential(
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
        return self.model(x)

# 9. 训练设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

model = COVIDPredictor(input_dim=X_train_selected.shape[1]).to(device)
criterion = nn.L1Loss()  # MAE损失
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

# 10. 训练循环
epochs = 200
best_val_loss = float('inf')
patience = 20
patience_counter = 0

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
        optimizer.step()
        
        train_loss += loss.item() * batch_X.size(0)
    
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
    
    # 学习率调整
    scheduler.step(val_loss)
    
    # 早停检查
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
    
    if epoch % 20 == 0:
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch}')
        break

# 11. 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))

# 12. 在验证集上评估
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
score = 1.0 / (1.0 + rmse)

print(f'验证集RMSE: {rmse:.4f}')
print(f'Score = (1.0 / (1.0 + RMSE)) = {score:.4f}')

# 13. 在测试集上进行预测
test_predictions = []
model.eval()
with torch.no_grad():
    # 分批预测以避免内存问题
    batch_size_test = 64
    for i in range(0, len(X_test_tensor), batch_size_test):
        batch = X_test_tensor[i:i+batch_size_test].to(device)
        preds = model(batch)
        test_predictions.extend(preds.cpu().numpy())

test_predictions = np.array(test_predictions).flatten()

# 14. 后处理：将负值设为0（目标变量为百分比，不可能为负）
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